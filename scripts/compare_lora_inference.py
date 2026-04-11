"""
Compare inference outputs before/after LoRA fine-tuning on CUDA.

Example:
  python scripts/compare_lora_inference.py \
    --task success \
    --model_dir models/Qwen2.5-3B-Instruct \
    --adapter_dir outputs/lora_qwen25_3b_success/final_adapter \
    --num_samples 200
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default=None, choices=["success", "topology"])
    parser.add_argument("--model_dir", default="models/Qwen2.5-3B-Instruct")
    parser.add_argument("--adapter_dir", required=True,
                        help="Path to LoRA adapter dir, e.g. outputs/lora_qwen25_3b_success/final_adapter")
    parser.add_argument("--input_jsonl", default="outputs/lora_qwen25_3b_success/test_split.jsonl",
                        help="Input jsonl. Defaults to outputs/lora_qwen25_3b_success/test_split.jsonl")
    parser.add_argument("--output_dir", default="outputs/lora_inference_compare")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of examples to evaluate. Use <=0 for all samples")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def output_constraint(task: str | None) -> str:
    if task == "success":
        return (
            "Constraint: Return exactly one word: success or failed.\n"
            "Do not output any explanation, punctuation, or extra text.\n"
        )
    if task == "topology":
        return (
            "Constraint: Return exactly one topology code (uppercase), for example AFI.\n"
            "Do not output any explanation, punctuation, or extra text.\n"
        )
    return (
        "Constraint: Return only the final label.\n"
        "Do not output any explanation, punctuation, or extra text.\n"
    )


def build_prompt(row: dict, strict_constraint: bool = False) -> str:
    constraint = output_constraint(row.get("task")) if strict_constraint else ""
    return (
        f"Instruction: {row['instruction']}\n\n"
        f"Input:\n{row['input']}\n\n"
        f"{constraint}"
        "Output:\n"
    )


def normalize_prediction(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    first = lines[0]
    low = first.lower()
    if low.startswith("output:"):
        first = first.split(":", 1)[1].strip()
    return first.strip()


def free_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_inference(
    model,
    tokenizer,
    rows: list[dict],
    max_new_tokens: int,
    strict_constraint: bool = False,
) -> tuple[list[str], float]:
    predictions: list[str] = []
    t0 = time.perf_counter()
    print(f"  start inference: {len(rows)} samples", flush=True)

    for idx, row in enumerate(rows, start=1):
        prompt = build_prompt(row, strict_constraint=strict_constraint)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.inference_mode():
            out_ids = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        gen_ids = out_ids[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = normalize_prediction(text)
        predictions.append(pred)

        if idx % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  processed {idx}/{len(rows)} | elapsed={elapsed:.1f}s | avg={elapsed/idx:.3f}s/sample",
                flush=True,
            )

    elapsed = time.perf_counter() - t0
    return predictions, elapsed


def exact_match(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return float("nan")
    correct = 0
    for a, b in zip(y_true, y_pred):
        if a.strip().lower() == b.strip().lower():
            correct += 1
    return correct / len(y_true)


def main() -> None:
    args = parse_args()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = Path(__file__).resolve().parent.parent

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU inference on CUDA.")

    task = args.task
    model_dir = (root / args.model_dir).resolve()
    adapter_dir = (root / args.adapter_dir).resolve()

    input_jsonl = Path(args.input_jsonl).resolve()

    out_dir = (root / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {model_dir}", flush=True)
    print(f"LoRA adapter: {adapter_dir}", flush=True)
    print(f"Input file: {input_jsonl}", flush=True)
    print(f"Output dir: {out_dir}", flush=True)

    rows = read_jsonl(input_jsonl)
    if task is not None:
        rows = [r for r in rows if r.get("task") == task]

    if not rows:
        raise ValueError("No samples found after filtering.")

    random.seed(args.seed)
    if args.num_samples and args.num_samples > 0 and args.num_samples < len(rows):
        rows = random.sample(rows, args.num_samples)

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"dtype: {dtype}", flush=True)
    print(f"Samples: {len(rows)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n[1/2] Running BASE model inference...", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=dtype,
        device_map={"": "cuda"},
    )
    base_model.eval()
    base_preds, base_sec = run_inference(
        base_model,
        tokenizer,
        rows,
        args.max_new_tokens,
        strict_constraint=True,
    )
    free_model(base_model)

    print("\n[2/2] Running LoRA model inference...", flush=True)
    lora_base_model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=dtype,
        device_map={"": "cuda"},
    )
    lora_model = PeftModel.from_pretrained(lora_base_model, str(adapter_dir))
    lora_model.eval()
    lora_preds, lora_sec = run_inference(
        lora_model,
        tokenizer,
        rows,
        args.max_new_tokens,
        strict_constraint=False,
    )
    free_model(lora_model)

    y_true = [str(r.get("output", "")).strip() for r in rows]
    has_gold = any(x != "" for x in y_true)

    changed = sum(1 for a, b in zip(base_preds, lora_preds) if a.strip().lower() != b.strip().lower())
    changed_rate = changed / len(rows)

    summary = {
        "num_samples": len(rows),
        "task": task,
        "max_new_tokens": args.max_new_tokens,
        "base_seconds": round(base_sec, 4),
        "lora_seconds": round(lora_sec, 4),
        "base_avg_seconds_per_sample": round(base_sec / len(rows), 6),
        "lora_avg_seconds_per_sample": round(lora_sec / len(rows), 6),
        "prediction_changed_count": changed,
        "prediction_changed_rate": round(changed_rate, 6),
    }

    if has_gold:
        summary["base_exact_match"] = round(exact_match(y_true, base_preds), 6)
        summary["lora_exact_match"] = round(exact_match(y_true, lora_preds), 6)

    records_path = out_dir / "comparison_records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for row, bp, lp in zip(rows, base_preds, lora_preds):
            out = {
                "task": row.get("task"),
                "route_id": row.get("route_id"),
                "paper_id": row.get("paper_id"),
                "gold": row.get("output"),
                "base_pred": bp,
                "lora_pred": lp,
                "changed": bp.strip().lower() != lp.strip().lower(),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    summary_path = out_dir / "comparison_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== Done ===", flush=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Per-sample results: {records_path}", flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
