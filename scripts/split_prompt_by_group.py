#!/usr/bin/env python3
"""Split prompt JSONL into train/valid/test by group_id.

This script prevents data leakage by assigning each group to exactly one split.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "outputs" / "prompts" / "recipe_success.jsonl",
    ROOT / "outputs" / "prompts" / "recipe_topology.jsonl",
]
DEFAULT_OUT_ROOT = ROOT / "outputs" / "prompts" / "splits"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{i}") from exc
    return rows


def dump_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compute_group_splits(
    groups: List[str],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Tuple[set[str], set[str], set[str]]:
    if not groups:
        return set(), set(), set()

    rng = random.Random(seed)
    groups = groups[:]
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    # Keep all groups assigned, and avoid empty train split when data exists.
    if n_train == 0:
        n_train = 1
    if n_train + n_valid > n:
        n_valid = max(0, n - n_train)

    train_groups = set(groups[:n_train])
    valid_groups = set(groups[n_train : n_train + n_valid])
    test_groups = set(groups[n_train + n_valid :])
    return train_groups, valid_groups, test_groups


def split_by_group(
    rows: List[Dict[str, Any]],
    group_col: str,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(rows):
        gid = row.get(group_col)
        if gid is None or str(gid).strip() == "":
            raise ValueError(f"Row {idx} missing non-empty '{group_col}'")
        grouped[str(gid)].append(row)

    all_groups = sorted(grouped.keys())
    train_g, valid_g, test_g = compute_group_splits(all_groups, train_ratio, valid_ratio, seed)

    splits = {"train": [], "valid": [], "test": []}
    for gid, items in grouped.items():
        if gid in train_g:
            splits["train"].extend(items)
        elif gid in valid_g:
            splits["valid"].extend(items)
        else:
            splits["test"].extend(items)

    return splits


def summarize(splits: Dict[str, List[Dict[str, Any]]], group_col: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for name, rows in splits.items():
        out[name] = {
            "rows": len(rows),
            "groups": len({str(r[group_col]) for r in rows}),
        }
    return out


def derive_dataset_name(path: Path) -> str:
    stem = path.stem
    if stem.startswith("recipe_"):
        return stem.replace("recipe_", "", 1)
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Group-aware split for prompt JSONL files")
    parser.add_argument("--inputs", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--group-col", type=str, default="group_id")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio must be between 0 and 1")
    if not (0 <= args.valid_ratio < 1):
        raise ValueError("--valid-ratio must be between 0 and 1")
    if args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("train_ratio + valid_ratio must be < 1")

    for input_path in args.inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input not found: {input_path}")

        rows = load_jsonl(input_path)
        splits = split_by_group(
            rows=rows,
            group_col=args.group_col,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )

        dataset_name = derive_dataset_name(input_path)
        dataset_out_dir = args.out_root / dataset_name

        for split_name, split_rows in splits.items():
            dump_jsonl(dataset_out_dir / f"{split_name}.jsonl", split_rows)

        stats = summarize(splits, args.group_col)
        print(f"[{dataset_name}] {input_path}")
        print(
            f"  train: {stats['train']['rows']} rows / {stats['train']['groups']} groups | "
            f"valid: {stats['valid']['rows']} rows / {stats['valid']['groups']} groups | "
            f"test: {stats['test']['rows']} rows / {stats['test']['groups']} groups"
        )
        print(f"  output dir: {dataset_out_dir}")


if __name__ == "__main__":
    main()
