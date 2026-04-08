"""
Step 4: Append matched records into database/csv_tables/adsorption_table.csv,assign adsorption_ids, fill source columns, and de-duplicate.
        将匹配的记录追加到 database/csv_tables/adsorption_table.csv 中，分配 adsorption_ids，填充源列，并去重。
Input:  outputs/matched_records.csv
        database/csv_tables/paper_table.csv   (for DOI lookup)
        database/csv_tables/adsorption_table.csv   (existing rows)
Output: database/csv_tables/adsorption_table.csv   (updated in-place)

Usage:
    python src/build_adsorption_table.py

    # Dry-run: see what would be written without touching the table
    python src/build_adsorption_table.py --dry-run
"""

import os
import argparse

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHED_FILE = os.path.join(ROOT, "outputs", "matched_records.csv")
ADSORPTION_TABLE = os.path.join(ROOT, "database", "csv_tables", "adsorption_table.csv")
PAPER_TABLE = os.path.join(ROOT, "database", "csv_tables", "paper_table.csv")

# 必须与 adsorption_table.csv 的表头完全一致（用于重排列与补齐缺失列）
TABLE_COLS = [
    "adsorption_id", "route_id", "sample_label", "activation_condition",
    "gas", "uptake_value", "uptake_unit", "temperature_k", "pressure_bar",
    "measurement_type", "selectivity_target", "selectivity_value",
    "selectivity_basis", "pretreatment", "test_condition",
    "source", "source_doi", "notes",
]

# 去重键：这些字段完全一致时视为重复记录
DEDUP_KEYS = ["route_id", "gas", "temperature_k", "pressure_bar", "uptake_value"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched", default=MATCHED_FILE)
    parser.add_argument("--output", default=ADSORPTION_TABLE)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print a preview without writing to disk.",
    )
    args = parser.parse_args()

    new_rows = pd.read_csv(args.matched)

    # 从 paper_table 回填来源信息，便于后续追溯原文献
    if os.path.exists(PAPER_TABLE):
        papers = pd.read_csv(PAPER_TABLE, usecols=["paper_id", "doi"])
        new_rows = new_rows.merge(papers, on="paper_id", how="left")
        new_rows["source"] = new_rows["paper_id"].astype(str)
        new_rows["source_doi"] = new_rows.get("doi", pd.Series(dtype=str))

    # 读取现有吸附表，与新数据拼接后统一去重
    existing = pd.read_csv(args.output)

    # 生成连续 adsorption_id，避免与历史记录冲突
    if len(existing) and existing["adsorption_id"].notna().any():
        next_id = int(existing["adsorption_id"].max()) + 1
    else:
        next_id = 1

    new_rows = new_rows.reset_index(drop=True)
    new_rows["adsorption_id"] = range(next_id, next_id + len(new_rows))

    # 对齐目标表结构：缺的列补 None，多余列丢弃并按 TABLE_COLS 排序
    for col in TABLE_COLS:
        if col not in new_rows.columns:
            new_rows[col] = None
    new_rows = new_rows[TABLE_COLS]

    # 先合并再按业务键去重，保留首次出现的记录
    combined = pd.concat([existing, new_rows], ignore_index=True)
    before = len(combined)
    valid_keys = [k for k in DEDUP_KEYS if k in combined.columns]
    combined = combined.drop_duplicates(subset=valid_keys, keep="first")
    n_dropped = before - len(combined)

    print(f"New rows to add : {len(new_rows)}")
    print(f"Duplicates removed: {n_dropped}")
    print(f"Final table size  : {len(combined)} rows")

    if args.dry_run:
        print("\n[DRY RUN] No changes written.")
        print(new_rows.to_string(max_rows=20))
        return

    combined.to_csv(args.output, index=False)
    print(f"Updated → {args.output}")


if __name__ == "__main__":
    main()
