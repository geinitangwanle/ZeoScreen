#!/usr/bin/env python3
"""Generate recipe prompts from ZeoScreen tables.

Build prompts using:
- raw materials & composition
- OSDA information
- process conditions

Outputs JSONL (default) or CSV for downstream LLM usage.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "database" / "csv_tables"
OUT_PATH = ROOT / "outputs" / "prompts" / "recipe_prompts.jsonl"


def clean_text(x: Any) -> Optional[str]:
    if x is None or pd.isna(x):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def clean_scalar(x: Any) -> Optional[float]:
    if x is None or pd.isna(x):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt_value(x: Any, null_token: str = "null") -> str:
    v = clean_text(x)
    return v if v is not None else null_token


def fmt_num(x: Any, ndigits: int = 2, null_token: str = "null") -> str:
    v = clean_scalar(x)
    if v is None:
        return null_token
    return f"{v:.{ndigits}f}"


def truncate_synonyms(text: Optional[str], max_items: int = 5) -> Optional[str]:
    s = clean_text(text)
    if s is None:
        return None
    # Handle Python-list-like strings from CSV.
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s.replace("'", '"'))
            if isinstance(arr, list):
                arr = [clean_text(v) for v in arr]
                arr = [v for v in arr if v]
                if not arr:
                    return None
                return "; ".join(arr[:max_items])
        except Exception:  # noqa: BLE001
            pass

    parts = [p.strip() for p in re.split(r";|,", s) if p.strip()]
    if not parts:
        return s
    return "; ".join(parts[:max_items])


def build_composition_block(row: pd.Series) -> str:
    ratio_map: Dict[str, float] = {}
    for c in row.index:
        if not c.endswith("_ratio_to_t"):
            continue
        v = clean_scalar(row.get(c))
        if v is None or v <= 0:
            continue
        elem = c.replace("_ratio_to_t", "")
        ratio_map[elem] = v

    # Fallback: when normalized ratios are sparse, use raw element columns if available.
    if len(ratio_map) < 2:
        for k in ("si", "al", "na", "k", "p", "ge", "ti", "fe", "b"):
            v = clean_scalar(row.get(k))
            if v is not None and v > 0 and k not in ratio_map:
                ratio_map[k] = v

    special_map: Dict[str, float] = {}
    for key in ("oh", "h2o"):
        v = clean_scalar(row.get(key))
        if v is not None and v > 0:
            special_map[key] = v

    osda_total = 0.0
    osda_found = False
    for key in ("sda1", "sda2", "sda3"):
        v = clean_scalar(row.get(key))
        if v is not None and v > 0:
            osda_total += v
            osda_found = True
    if osda_found:
        special_map["osda"] = osda_total

    if not ratio_map and not special_map:
        return "Gel composition: null."

    preferred = ["si", "al", "na", "oh", "h2o", "osda"]
    labels = {
        "si": "Si",
        "al": "Al",
        "na": "Na",
        "oh": "OH",
        "h2o": "H2O",
        "osda": "OSDA",
    }

    parts: List[str] = []
    used = set()
    for k in preferred:
        if k in ratio_map:
            parts.append(f"{labels.get(k, k)}={ratio_map[k]:.2f}")
            used.add(k)
        elif k in special_map:
            parts.append(f"{labels.get(k, k)}={special_map[k]:.2f}")
            used.add(k)

    for k in sorted(ratio_map):
        if k in used:
            continue
        parts.append(f"{k.upper()}={ratio_map[k]:.2f}")

    return "Gel composition: " + ", ".join(parts) + "."


def build_materials_block(row: pd.Series) -> str:
    return "\n".join(
        [
            f"Silica source: {fmt_value(row.get('silica_source'), 'none')}.",
            f"Alumina source: {fmt_value(row.get('alumina_source'), 'none')}.",
            f"Alkali source: {fmt_value(row.get('alkali_source'), 'none')}.",
            f"Mineralizer: {fmt_value(row.get('mineralizer'), 'none')}.",
            f"Seed source: {fmt_value(row.get('seed_source'), 'none')}.",
        ]
    )


def build_osda_block(row: pd.Series, verbose: bool = False) -> str:
    chunks: List[str] = []
    for i in (1, 2, 3):
        name = clean_text(row.get(f"osda{i}"))
        smiles = clean_text(row.get(f"osda{i}_smiles"))
        iupac = clean_text(row.get(f"osda{i}_iupac")) if verbose else None
        formula = clean_text(row.get(f"osda{i}_formula")) if verbose else None
        syn = truncate_synonyms(row.get(f"osda{i}_synonyms"), max_items=3) if verbose else None
        if not any([name, smiles, iupac, formula, syn]):
            continue
        extra = []
        if smiles:
            extra.append(f"SMILES={smiles}")
        if iupac:
            extra.append(f"IUPAC={iupac}")
        if formula:
            extra.append(f"Formula={formula}")
        if syn:
            extra.append(f"Synonyms={syn}")
        extra_txt = "; ".join(extra)
        if extra_txt:
            chunks.append(f"OSDA-{i}: {name or 'unknown'}; {extra_txt}.")
        else:
            chunks.append(f"OSDA-{i}: {name or 'unknown'}.")

    # Keep V1 concise: at most two OSDAs by default.
    if not verbose and len(chunks) > 2:
        chunks = chunks[:2]
    return "\n".join(chunks) if chunks else "OSDA-1: none."


def build_process_block(row: pd.Series) -> str:
    aging_temp = fmt_num(row.get("aging_temp"), ndigits=0)
    aging_time = fmt_num(row.get("aging_time"), ndigits=0)
    cryst_temp = fmt_num(row.get("cryst_temp"), ndigits=0)
    cryst_time = fmt_num(row.get("cryst_time"), ndigits=0)
    rotation = fmt_value(row.get("rotation"), "none")
    ph = fmt_value(row.get("ph"), "null")
    return "\n".join(
        [
            f"Aging: {aging_temp} C for {aging_time} h.",
            f"Crystallization: {cryst_temp} C for {cryst_time} h.",
            f"Rotation: {rotation}.",
            f"pH: {ph}.",
        ]
    )


def build_recipe_prompt(
    row: pd.Series,
    product: Optional[pd.Series] = None,
    debug_include_label: bool = False,
    osda_verbose: bool = False,
) -> str:
    lines = ["Zeolite hydrothermal synthesis recipe."]
    lines.append(build_materials_block(row))
    lines.append(build_composition_block(row))
    lines.append(build_osda_block(row, verbose=osda_verbose))
    lines.append(build_process_block(row))

    if debug_include_label and product is not None:
        target = clean_text(product.get("primary_topology")) or "null"
        fail = clean_text(product.get("failure_label")) or "null"
        lines.append(f"DEBUG_LABEL: primary_topology={target}; failure_label={fail}.")

    return "\n".join(lines)


def parse_route_ids(raw: Optional[str]) -> Optional[List[int]]:
    s = clean_text(raw)
    if s is None:
        return None
    ids = []
    for p in s.split(","):
        p = p.strip()
        if not p:
            continue
        ids.append(int(p))
    return ids if ids else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate recipe prompts from ZeoScreen CSV tables")
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows")
    parser.add_argument("--route-ids", type=str, default="", help="comma-separated route ids")
    parser.add_argument("--success-only", action="store_true", help="only keep success routes")
    parser.add_argument(
        "--debug-include-label",
        action="store_true",
        help="DEBUG only: append ground-truth labels into input text (do not use for training)",
    )
    parser.add_argument(
        "--osda-verbose",
        action="store_true",
        help="include IUPAC/formula/synonyms in OSDA block (default is concise name+SMILES)",
    )
    parser.add_argument(
        "--task",
        choices=["success", "topology", "both"],
        default="both",
        help="dataset task type",
    )
    parser.add_argument(
        "--topology-include-failed",
        action="store_true",
        help="include failed routes in topology task (default only success routes)",
    )
    args = parser.parse_args()

    routes = pd.read_csv(args.csv_dir / "synthesis_routes.csv", low_memory=False)
    products = pd.read_csv(args.csv_dir / "product_table.csv", low_memory=False)
    papers = pd.read_csv(args.csv_dir / "paper_table.csv", low_memory=False)

    merged = routes.merge(products[["route_id", "primary_topology", "failure_label"]], on="route_id", how="left")
    merged = merged.merge(papers[["paper_id", "doi", "title", "year"]], on="paper_id", how="left")

    ids = parse_route_ids(args.route_ids)
    if ids is not None:
        merged = merged[merged["route_id"].isin(ids)]

    if args.success_only:
        merged = merged[merged["failure_label"] == "success"]

    if args.max_rows > 0:
        merged = merged.head(args.max_rows)

    out_rows: List[Dict[str, Any]] = []
    for _, row in merged.iterrows():
        prompt = build_recipe_prompt(
            row,
            product=row,
            debug_include_label=args.debug_include_label,
            osda_verbose=args.osda_verbose,
        )
        route_id = int(row["route_id"])
        paper_id = None if pd.isna(row.get("paper_id")) else int(row["paper_id"])
        doi = clean_text(row.get("doi"))
        group_id = doi if doi is not None else (str(paper_id) if paper_id is not None else f"route:{route_id}")
        target_primary_topology = clean_text(row.get("primary_topology"))
        failure_label = clean_text(row.get("failure_label"))
        success_output = "success" if failure_label == "success" else "failed"

        if args.task in {"success", "both"}:
            out_rows.append(
                {
                    "task": "success",
                    "route_id": route_id,
                    "paper_id": paper_id,
                    "doi": doi,
                    "group_id": group_id,
                    "instruction": "Predict whether the zeolite synthesis will be successful.",
                    "input": prompt,
                    "output": success_output,
                    "target_failure_label": failure_label,
                }
            )

        if args.task in {"topology", "both"}:
            topo_output = target_primary_topology or "null"
            if (not args.topology_include_failed) and failure_label != "success":
                continue
            if topo_output == "null":
                continue
            out_rows.append(
                {
                    "task": "topology",
                    "route_id": route_id,
                    "paper_id": paper_id,
                    "doi": doi,
                    "group_id": group_id,
                    "instruction": "Predict the primary zeolite topology formed by this synthesis recipe.",
                    "input": prompt,
                    "output": topo_output,
                    "target_primary_topology": target_primary_topology,
                }
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "jsonl":
        with args.out.open("w", encoding="utf-8") as f:
            for r in out_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    else:
        pd.DataFrame(out_rows).to_csv(args.out, index=False)

    print(f"Generated prompts: {len(out_rows)}")
    print(f"Output: {args.out}")


if __name__ == "__main__":
    main()
