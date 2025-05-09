import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel
from datasets import load_dataset


from generate import generate
from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from countdown import CTDDataset
from sudoku import SudokuDataset
from gsm8k_fewshot import GSM8KFewShotDataset
from parsers import Parser

# Default prompts for hierarchical generation
DEFAULT_HIGH_LEVEL_PROMPT = """You are a math expert. Given a math problem, provide a concise, high-level plan to solve it. Do not solve the problem, just outline the steps.
Example:
Question: Natalia sold clips to 48 of her friends. She sold 8 clips to each friend. How many clips did she sell in total?
Plan:
1. Identify the number of friends.
2. Identify the number of clips sold to each friend.
3. Multiply the number of friends by the number of clips per friend to get the total number of clips sold."""

DEFAULT_DETAIL_PROMPT = """You are a math expert. Given a math problem and a high-level plan, solve the problem step-by-step following the plan. Wrap the final answer in \\boxed{}.
Respond in the following format:
<reasoning>
Your step-by-step reasoning here, following the plan.
</reasoning>
<answer>
\\boxed{...}
</answer>"""

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
    "gsm8k_fewshot": GSM8KFewShotDataset,
    "gsm8k_hierarchical": GSM8KHierarchicalDataset,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def evaluate_hierarchical(
    model,
    tokenizer,
    dataloader,
    high_level_system_prompt,
    detail_system_prompt,
    gen_length_plan=256,
    gen_length_detail=512,
    temperature=0.0,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times_total = []
    all_generations_hierarchical = []
    device = model.device

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        start_time_batch = time.time()
        questions = batch["questions"]
        gt_answers_raw = batch["answers_raw"]

        # Stage 1: Generate High-Level Plan
        plan_prompts = []
        for q_text in questions:
            messages_plan = [
                {"role": "user", "content": f"{high_level_system_prompt}\\n\\nQuestion: {q_text}\\nPlan:"}
            ]
            plan_prompts.append(tokenizer.apply_chat_template(messages_plan, add_generation_prompt=True, tokenize=False))

        input_ids_plan = tokenizer(plan_prompts, padding_side="left", return_tensors="pt", padding="longest").input_ids.to(device)
        
        out_plan = generate(
            model,
            input_ids_plan,
            tokenizer,
            steps=steps,
            gen_length=gen_length_plan,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking="low_confidence",
        )
        raw_decoded_plans_batch = tokenizer.batch_decode(out_plan[:, input_ids_plan.shape[1]:], skip_special_tokens=False)
        
        generated_plans = []
        for raw_plan_text in raw_decoded_plans_batch:
            plan_content = raw_plan_text.split("<|eot_id|>")[0]
            generated_plans.append(plan_content.strip())
        
        # Stage 2: Generate Detailed Solution using the Plan
        detail_prompts = []
        for i, q_text in enumerate(questions):
            current_cleaned_plan = generated_plans[i] 
            messages_detail = [
                {"role": "user", "content": f"{detail_system_prompt}\\n\\nQuestion: {q_text}\\nHigh-Level Plan: {current_cleaned_plan}\\nAnswer:"}
            ]
            full_detail_prompt_text = tokenizer.apply_chat_template(messages_detail, add_generation_prompt=True, tokenize=False) + "<reasoning>"
            detail_prompts.append(full_detail_prompt_text)

        input_ids_detail = tokenizer(detail_prompts, padding_side="left", return_tensors="pt", padding="longest").input_ids.to(device)

        out_detail = generate(
            model,
            input_ids_detail,
            tokenizer,
            steps=steps,
            gen_length=gen_length_detail,
            block_length=block_length,
            temperature=temperature,
            cfg_scale=cfg_scale,
            remasking="low_confidence",
        )
        generated_details = tokenizer.batch_decode(out_detail[:, input_ids_detail.shape[1]:], skip_special_tokens=False)

        example_results_hierarchical = [
            {
                "question": questions[j],
                "prompt_plan": plan_prompts[j],
                "generated_plan": generated_plans[j],
                "prompt_detail": detail_prompts[j],
                "generated_detail": "<reasoning>" + generated_details[j],
                "ground_truth_raw": gt_answers_raw[j],
            }
            for j in range(len(questions))
        ]
        all_generations_hierarchical.extend(example_results_hierarchical)
        total_processed += len(questions)
        wall_times_total.append(time.time() - start_time_batch)

        if dist.get_rank() == 0 and questions:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 25 + " PLAN " + "-" * 25)
            print(f"Prompt Plan: {plan_prompts[idx]}")
            print(f"Generated Plan: {generated_plans[idx]}")
            print("-" * 25 + " DETAIL " + "-" * 25)
            print(f"Prompt Detail: {detail_prompts[idx]}")
            print(f"Generated Detail: {generated_details[idx]}")
            print("-" * 50)
            print(f"Ground truth (raw): {gt_answers_raw[idx]}")


    avg_wall_time = sum(wall_times_total) / len(wall_times_total) if wall_times_total else 0
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations_hierarchical,
        "total_processed": total_processed.item(),
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


class GSM8KHierarchicalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        subsample=-1,
    ):
        self.tokenizer = tokenizer
        self.load_test_dataset()
        self.subsample_indices = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"GSM8KHierarchicalDataset: evaluating {len(self.subsample_indices)} examples")
        assert subsample <= len(self.dataset), "Subsample size is greater than dataset size"

    def __len__(self):
        return len(self.subsample_indices)

    def load_test_dataset(self):
        self.dataset = load_dataset("gsm8k", "main", split="test")

    def __getitem__(self, idx):
        actual_idx = self.subsample_indices[idx]
        question = self.dataset[actual_idx.item()]["question"]
        answer_raw = self.dataset[actual_idx.item()]["answer"]
        return {"question": question, "answer_raw": answer_raw}

    @staticmethod
    def collate_fn(batch):
        questions = [item["question"] for item in batch]
        answers_raw = [item["answer_raw"] for item in batch]
        return {"questions": questions, "answers_raw": answers_raw}


if __name__ == "__main__":
    init_seed(42)

    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data1/shared/LLaDA-8B-Instruct/")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset", type=str, choices=["gsm8k", "math", "countdown", "sudoku", "game24", "gsm8k_fewshot", "gsm8k_hierarchical"], default="gsm8k_hierarchical"
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--gen_length_plan", type=int, default=64, help="Max generation length for high-level plan.")
    parser.add_argument("--gen_length_detail", type=int, default=256, help="Max generation length for detailed solution.")
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=64)
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results_hierarchical/")
    parser.add_argument("--high_level_prompt_text", type=str, default=DEFAULT_HIGH_LEVEL_PROMPT, help="System prompt for generating high-level plan.")
    parser.add_argument("--detail_prompt_text", type=str, default=DEFAULT_DETAIL_PROMPT, help="System prompt for generating detailed solution from plan.")
    parser.add_argument("--subsample_eval", type=int, default=-1, help="Number of examples to subsample for evaluation. -1 for all.")
    
    args = parser.parse_args()

    num_evals = {
        "gsm8k": -1, "math": -1, "countdown": 256, "sudoku": 256, "gsm8k_fewshot": -1,
        "gsm8k_hierarchical": args.subsample_eval
    }

    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
        local_rank
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    if args.checkpoint_path:
        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch.bfloat16).to(
            local_rank
        )
        if dist.get_world_size() > 1:
            dist.barrier()
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            print(f"Rank {local_rank}: Parameters synchronized")

    if args.dataset == "gsm8k_hierarchical":
        dataset = GSM8KHierarchicalDataset(
            tokenizer=tokenizer,
            subsample=num_evals[args.dataset]
        )
    else:
        dataset = DATASET_MAP[args.dataset](
            tokenizer,
            subsample=num_evals.get(args.dataset, -1),
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=GSM8KHierarchicalDataset.collate_fn if args.dataset == "gsm8k_hierarchical" else dataset.collate_fn,
    )

    if len(args.checkpoint_path):
        model_name_parts = args.checkpoint_path.rstrip("/").split("/")
        if len(model_name_parts) >= 2:
            model_name = model_name_parts[-2] + "_" + model_name_parts[-1]
        else:
            model_name = model_name_parts[-1] if model_name_parts else "unknown_adapter"
    else:
        model_name = "instruct" if "Instruct" in args.model_path else "base"
        model_name = model_name.split("/")[-1]

    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.dataset == "gsm8k_hierarchical":
        filename = f"{args.output_dir}/{args.dataset}_{model_name}_plan{args.gen_length_plan}_detail{args.gen_length_detail}_{args.diffusion_steps}_{dist.get_rank()}_generations.json"
    else:
        filename = f"{args.output_dir}/{args.dataset}_{model_name}_{args.gen_length_plan}_{args.diffusion_steps}_{dist.get_rank()}_generations.json"
    
    print(f"Saving generations to {filename}")

    if args.dataset == "gsm8k_hierarchical":
        metrics = evaluate_hierarchical(
            model,
            tokenizer,
            dataloader,
            high_level_system_prompt=args.high_level_prompt_text,
            detail_system_prompt=args.detail_prompt_text,
            gen_length_plan=args.gen_length_plan,
            gen_length_detail=args.gen_length_detail,
            block_length=args.block_length,
            steps=args.diffusion_steps,
        )
    else:
        raise NotImplementedError(f"Evaluation for dataset type '{args.dataset}' is not implemented in this hierarchical script version.")

    if not args.dont_save:
        with open(filename, "w") as f:
            json.dump(
                {
                    "generations": metrics["generations"],
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": metrics["total_processed"],
                    },
                    "args": vars(args),
                },
                f,
                indent=2,
            )
    cleanup_ddp()