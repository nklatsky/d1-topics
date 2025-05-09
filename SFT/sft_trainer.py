import torch
import torch.nn.functional as F
from transformers import Trainer
from transformers import DefaultDataCollator
import random
from tqdm import tqdm
import pickle
import torch.distributed as dist


class dLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Absorbing state diffusion loss computation
        """
        labels, t, num_prompt_tokens = inputs.pop("labels"), inputs.pop("t"), inputs.pop("num_prompt_tokens")
        outputs = model(**inputs)
        logits = outputs.logits
        unscaled_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]), labels.view(-1), reduction="none"
        ).view(logits.shape[0], -1)
        if (self.state.global_step + 1) % self.args.logging_steps == 0:
            self.log({"unscaled_loss": (unscaled_loss.sum() / (labels != -100).sum()).item()})
        loss = unscaled_loss / t
        loss = loss.sum() / (inputs["input_ids"].numel() - num_prompt_tokens)
        return loss if not return_outputs else (loss, outputs)


class dLLMSFTDataset(torch.utils.data.Dataset):
    """
    Similar to AR datasets, except in inference, we keep the timsteps fixed
    """

    def __init__(self, data, tokenizer, max_length, eval=False):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.eval = eval
        if self.eval:
            self.t = torch.linspace(0, 1, len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out = self.data[idx]
        if self.eval:
            out["t"] = self.t[idx]
        return out


class dLLMDataCollator(DefaultDataCollator):
    """
    Adds the forward noising process to the batch.
    Modify forward_process to change the noise schedule
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.mask_token_id = kwargs["tokenizer"].mask_token_id
        self.tokenizer = kwargs["tokenizer"]
        if "max_length" in kwargs:
            self.max_length = kwargs["max_length"]
        if kwargs["tokenizer"].mask_token_id is None:
            assert (
                "mask_token_id" in kwargs
            ), "For dLLM models, pass a mask_token_id or set it equal to tokenizer.mask_token_id"
            self.mask_token_id = kwargs["mask_token_id"]

    def forward_process(self, batch, eps=1e-3):
        input_ids = batch["input_ids"]
        B, N = input_ids.shape
        if "t" not in batch:
            t = torch.rand((B,), device=input_ids.device)
        else:
            t = batch["t"]

        t = (1 - eps) * t + eps
        t = t[:, None].repeat(1, N)

        mask_indices = torch.rand((B, N), device=input_ids.device) < t
        noisy_batch = torch.where(mask_indices, self.mask_token_id, input_ids)
        return noisy_batch, t, mask_indices

    def __call__(self, batch):
        batch = super().__call__(batch)
        batch["labels"] = batch["input_ids"].clone()
        noisy_batch, batch["t"], mask_indices = self.forward_process(batch)
        batch["labels"][~mask_indices] = -100
        batch["num_prompt_tokens"] = 0
        if "prompt_lengths" in batch:
            prompt_lengths = batch.pop("prompt_lengths")
            prompt_length_indices = torch.arange(noisy_batch.shape[1]).unsqueeze(0)
            prompt_mask = prompt_length_indices < prompt_lengths
            noisy_batch[prompt_mask] = batch["input_ids"][prompt_mask].clone()
            batch["labels"][prompt_mask] = -100
            batch["num_prompt_tokens"] = prompt_mask.sum()
        batch["input_ids"] = noisy_batch.long()
        return batch


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
...
</answer>
"""

# New SYSTEM_PROMPT for High-Level Plan rows
SYSTEM_PROMPT_HLP = """Reponse to the question with nothing but the high-level planning steps required to solve the problem, in this format:
<high-level planning>
High-level planning steps here
</high-level planning>"""


def preprocess_dataset(data, tokenizer, max_length, test_split=0.01):
    data_df = data.to_pandas()
    data_seq = data_df.copy()

    split_1 = "; High-Level Plan ;"
    split_2 = " ; The answer is #### " # This will only be used for non-HLP rows

    # Initialize columns
    data_seq['question'] = ""
    data_seq['thinking_trajectories'] = ""
    data_seq['attempt'] = ""

    has_hlp_column = "hlp?" in data_seq.columns

    for index, row_series in data_seq.iterrows():
        full_sequence_text = str(row_series['full_sequence'])
        is_hlp_row_for_split = False
        if has_hlp_column and pd.notna(row_series["hlp?"]) and int(row_series["hlp?"]) == 1:
            is_hlp_row_for_split = True

        parts_at_split1 = full_sequence_text.split(split_1, 1)
        current_question = parts_at_split1[0].strip()
        data_seq.loc[index, 'question'] = current_question

        if len(parts_at_split1) > 1:
            text_after_split1 = parts_at_split1[1].strip()
            if is_hlp_row_for_split:
                # For HLP rows, 'thinking_trajectories' is everything after split_1
                data_seq.loc[index, 'thinking_trajectories'] = text_after_split1
                data_seq.loc[index, 'attempt'] = "" # Ensure attempt is empty for HLP
            else:
                # For non-HLP rows, use split_2 on the text after split_1
                parts_at_split2 = text_after_split1.split(split_2, 1)
                data_seq.loc[index, 'thinking_trajectories'] = parts_at_split2[0].strip()
                if len(parts_at_split2) > 1:
                    data_seq.loc[index, 'attempt'] = parts_at_split2[1].strip()
                else:
                    data_seq.loc[index, 'attempt'] = "" # No text after split_2
        else:
            # No text after split_1
            data_seq.loc[index, 'thinking_trajectories'] = ""
            data_seq.loc[index, 'attempt'] = ""
            
    # Determine which columns to select for the loop
    cols_to_select = ['question', 'thinking_trajectories', 'attempt']
    if has_hlp_column:
        cols_to_select.append('hlp?')
    
    data_for_loop = data_seq[cols_to_select]

    preprocessed_data = []
    for i in tqdm(range(len(data_for_loop)), desc="Preprocessing dataset"):
        row = data_for_loop.iloc[i]
        
        question_text_from_split = str(row["question"])
        thinking_traj_text = str(row['thinking_trajectories'])
        attempt_text = str(row['attempt'])

        user_content_for_prompt = ""
        assistant_content = ""
        
        is_hlp_row_for_prompt = False
        if has_hlp_column and pd.notna(row["hlp?"]) and int(row["hlp?"]) == 1:
            is_hlp_row_for_prompt = True

        if is_hlp_row_for_prompt:
            # High-Level Plan row
            user_content_for_prompt = SYSTEM_PROMPT_HLP + "\n\n" + question_text_from_split
            # 'thinking_traj_text' for HLP rows now directly contains the full high-level plan
            assistant_content = f"<high-level planning>\n{thinking_traj_text}\n</high-level planning>"
        else:
            # Regular row
            user_content_for_prompt = SYSTEM_PROMPT + "\n\n" + question_text_from_split
            assistant_content = f"<reasoning>\n{thinking_traj_text}\n</reasoning>\n<answer>\n{attempt_text}\n</answer>"

        user_message = {"role": "user", "content": user_content_for_prompt}
        assistant_message = {"role": "assistant", "content": assistant_content}
        
        full_conversation_text = tokenizer.apply_chat_template(
            [user_message, assistant_message], 
            tokenize=False
        )
        tokenized_input = tokenizer(
            full_conversation_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length, 
            padding="max_length"
        ).input_ids.squeeze(0)

        prompt_part_text = tokenizer.apply_chat_template(
            [user_message], 
            tokenize=False, 
            add_generation_prompt=True 
        )
        tokenized_prompt_for_length = tokenizer(
            prompt_part_text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=max_length
        )
        
        preprocessed_data.append(
            {
                "input_ids": tokenized_input,
                "prompt_lengths": tokenized_prompt_for_length.attention_mask.sum(-1),
            }
        )

    random.shuffle(preprocessed_data)
    num_test_samples = int(len(preprocessed_data) * test_split)
    test_data = preprocessed_data[:num_test_samples]
    train_data = preprocessed_data[num_test_samples:]
    
    print(f"Total preprocessed examples: {len(preprocessed_data)}")
    print(f"Train data length after split: {len(train_data)}")
    print(f"Test data length after split: {len(test_data)}")

    return train_data, test_data
