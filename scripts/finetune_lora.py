"""
LoRA fine-tuning for Qwen2.5-3B-Instruct on zeolite synthesis prediction.
Targets CUDA environments (single or multi-GPU via device_map="auto").

Tasks:
  success  — binary classification: success / failed
  topology — topology prediction

Usage:
  python scripts/finetune_lora.py --task success
  python scripts/finetune_lora.py --task topology --epochs 5 --lr 5e-5
"""

import argparse
import inspect
import json
import os
from pathlib import Path
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="success", choices=["success", "topology"])
    parser.add_argument("--model_dir", default="models/Qwen2.5-3B-Instruct")
    parser.add_argument("--prompt_file", default=None,
                        help="Override default prompt JSONL path")
    parser.add_argument("--output_dir", default=None,
                        help="Override default output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    # Throughput-oriented defaults for single L40.
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device train batch size")
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--dataloader_workers", type=int, default=8)
    parser.add_argument("--dataset_num_proc", type=int, default=8)
    parser.add_argument("--packing", action=argparse.BooleanOptionalAction, default=True,
                        help="Pack multiple short samples into one sequence when supported by trl")
    parser.add_argument("--do_eval", action=argparse.BooleanOptionalAction, default=True,
                        help="Run eval during training")
    parser.add_argument("--use_device_map_auto", action=argparse.BooleanOptionalAction, default=False,
                        help="Use device_map='auto'. Default off for single-GPU throughput")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True,
                        help="Enable TF32 matmul on Ampere+ GPUs for better throughput")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_and_split(prompt_file: Path, task: str, seed: int):
    rows = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    df = pd.DataFrame(rows)
    df = df[df["task"] == task].copy().reset_index(drop=True)
    print(f"Total samples for task '{task}': {len(df)}")
    print(df["output"].value_counts(dropna=False).to_string())

    # Split: 70 / 15 / 15 by group_id (no data leakage across papers)
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    train_valid_idx, test_idx = next(gss1.split(df, groups=df["group_id"]))

    train_valid_df = df.iloc[train_valid_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed)
    train_idx, valid_idx = next(gss2.split(train_valid_df, groups=train_valid_df["group_id"]))

    train_df = train_valid_df.iloc[train_idx].reset_index(drop=True)
    valid_df = train_valid_df.iloc[valid_idx].reset_index(drop=True)

    print(f"\nSplit — train: {len(train_df)}, valid: {len(valid_df)}, test: {len(test_df)}")
    return train_df, valid_df, test_df


def build_prompt(row):
    return (
        f"Instruction: {row['instruction']}\n\n"
        f"Input:\n{row['input']}\n\n"
        f"Output:\n"
    )


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(
        pd.DataFrame({
            "prompt": df.apply(build_prompt, axis=1),
            "completion": df["output"].astype(str),
        }),
        preserve_index=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    root = Path(__file__).resolve().parent.parent
    model_dir = root / args.model_dir
    prompt_file = Path(args.prompt_file) if args.prompt_file else \
        root / f"outputs/prompts/recipe_{args.task}.jsonl"
    output_dir = Path(args.output_dir) if args.output_dir else \
        root / f"outputs/lora_qwen25_3b_{args.task}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Device / dtype
    # ------------------------------------------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        print(f"Detected distributed launch (WORLD_SIZE={world_size}).")
        print("For single-card high throughput, run one process (ntasks=1).")

    if torch.cuda.is_available():
        # For single-GPU throughput, let Trainer own device placement.
        # device_map='auto' can be enabled via CLI when needed.
        device_map = "auto" if args.use_device_map_auto else None
        bf16_ok = torch.cuda.is_bf16_supported()
        model_dtype = torch.bfloat16 if bf16_ok else torch.float16
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print(f"Backend: CUDA — {torch.cuda.get_device_name(0)}")
        print(f"dtype: {model_dtype}, bf16={bf16_ok}, fp16={not bf16_ok}")
        print(f"device_map: {device_map}")
        print(f"tf32: {args.tf32}")
    elif torch.backends.mps.is_available():
        # Apple Silicon — MPS does not support bf16/fp16 training stably,
        # use float32; gradient_checkpointing will be disabled below.
        device_map = {"": "mps"}
        bf16_ok = False
        model_dtype = torch.float32
        print("Backend: MPS (Apple Silicon) — dtype: float32")
    else:
        device_map = "cpu"
        bf16_ok = False
        model_dtype = torch.float32
        print("Backend: CPU — dtype: float32")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_df, valid_df, test_df = load_and_split(prompt_file, args.task, args.seed)

    # Save test split for later evaluation
    test_save = output_dir / "test_split.jsonl"
    test_df.to_json(test_save, orient="records", lines=True, force_ascii=False)
    print(f"Test split saved → {test_save}")

    train_ds = to_hf_dataset(train_df)
    valid_ds = to_hf_dataset(valid_df)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=model_dtype,
        device_map=device_map,
    )
    model.config.use_cache = False  # required when gradient_checkpointing=True

    # ------------------------------------------------------------------
    # LoRA config
    # ------------------------------------------------------------------
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    # ------------------------------------------------------------------
    # Training args
    # ------------------------------------------------------------------
    sft_config_sig = inspect.signature(SFTConfig.__init__)
    sft_config_kwargs = dict(
        output_dir=str(output_dir),
        # Batch / steps
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        # Optimisation
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        # Precision
        bf16=bf16_ok,
        fp16=(not bf16_ok),
        # Memory (gradient checkpointing not supported on MPS/CPU)
        gradient_checkpointing=torch.cuda.is_available(),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Logging / eval / save
        logging_steps=20,
        save_steps=args.save_steps,
        save_total_limit=2,
        # Misc
        dataloader_num_workers=args.dataloader_workers,
        dataset_num_proc=args.dataset_num_proc,
        seed=args.seed,
        report_to="none",
    )
    if args.do_eval:
        # Transformers API varies by version: eval_strategy vs evaluation_strategy.
        if "eval_strategy" in sft_config_sig.parameters:
            sft_config_kwargs["eval_strategy"] = "steps"
        elif "evaluation_strategy" in sft_config_sig.parameters:
            sft_config_kwargs["evaluation_strategy"] = "steps"
        sft_config_kwargs["eval_steps"] = args.eval_steps
        sft_config_kwargs["load_best_model_at_end"] = True
        sft_config_kwargs["metric_for_best_model"] = "eval_loss"
        sft_config_kwargs["greater_is_better"] = False
    else:
        if "eval_strategy" in sft_config_sig.parameters:
            sft_config_kwargs["eval_strategy"] = "no"
        elif "evaluation_strategy" in sft_config_sig.parameters:
            sft_config_kwargs["evaluation_strategy"] = "no"
    # trl API changed across versions:
    # - some versions expect max_seq_length in SFTConfig
    # - others take sequence length on SFTTrainer only.
    if "max_seq_length" in sft_config_sig.parameters:
        sft_config_kwargs["max_seq_length"] = args.max_seq_len

    training_args = SFTConfig(**sft_config_kwargs)

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    # max_seq_length passed here for compatibility with older trl versions
    # that don't accept it in SFTConfig
    sft_trainer_sig = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=(valid_ds if args.do_eval else None),
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    if "packing" in sft_trainer_sig.parameters:
        trainer_kwargs["packing"] = args.packing
    if "max_seq_length" in sft_trainer_sig.parameters:
        trainer_kwargs["max_seq_length"] = args.max_seq_len

    trainer = SFTTrainer(**trainer_kwargs)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\n=== Starting training ===")
    trainer.train()

    # ------------------------------------------------------------------
    # Save final adapter
    # ------------------------------------------------------------------
    adapter_dir = output_dir / "final_adapter"
    trainer.model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\nFinal LoRA adapter saved → {adapter_dir}")


if __name__ == "__main__":
    main()
