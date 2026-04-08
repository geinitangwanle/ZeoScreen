"""
Step 3: Join normalized adsorption records with synthesis_routes to obtain route_id.
One paper can have multiple synthesis routes.  The join is a LEFT join so that every
adsorption record is preserved even when no matching route exists (route_id = NaN).
These unmatched rows still end up in the output so they can be reviewed.
将归一化的吸附记录与合成路线连接，以获得路线 ID。
一篇论文可以有多条合成路线。连接采用左连接 (LEFT JOIN)，因此即使不存在匹配的路线（路线 ID = NaN），每个吸附记录也会被保留。
这些不匹配的行仍然会出现在输出结果中，以便进行审查。

Input:  outputs/normalized_records.csv
        database/csv_tables/synthesis_routes.csv
Output: outputs/matched_records.csv

Usage:
    python src/match_route.py
"""

import os
import argparse

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NORM_FILE = os.path.join(ROOT, "outputs", "normalized_records.csv")
ROUTES_FILE = os.path.join(ROOT, "database", "csv_tables", "synthesis_routes.csv")
OUTPUT_FILE = os.path.join(ROOT, "outputs", "matched_records.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", default=NORM_FILE)
    parser.add_argument("--routes", default=ROUTES_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    norm = pd.read_csv(args.norm)
    # 只读取匹配所需的列，避免宽表带来的额外内存开销
    routes = pd.read_csv(args.routes, usecols=["route_id", "paper_id"])

    # 以 paper_id 做左连接：保留所有吸附记录，未匹配路线的行 route_id 为 NaN
    merged = norm.merge(routes, on="paper_id", how="left")

    n_matched = int(merged["route_id"].notna().sum())
    n_unmatched = int(merged["route_id"].isna().sum())
    print(
        f"Total rows: {len(merged)}  |  "
        f"Matched to a route: {n_matched}  |  "
        f"No matching route: {n_unmatched}"
    )
    if n_unmatched:
        # 打印未匹配文献 ID，方便补齐路线数据或排查 paper_id 对齐问题
        missing_pids = sorted(merged.loc[merged["route_id"].isna(), "paper_id"].unique())
        print(f"  paper_ids with no route: {missing_pids}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged)} rows → {args.output}")


if __name__ == "__main__":
    main()
