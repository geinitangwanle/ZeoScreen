"""
Step 3: Join normalized adsorption records with synthesis_routes to obtain route_id.

One paper can have multiple synthesis routes.  The join is a LEFT join so that every
adsorption record is preserved even when no matching route exists (route_id = NaN).
These unmatched rows still end up in the output so they can be reviewed.

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
    # Only load the columns we need from the (wide) synthesis_routes table
    routes = pd.read_csv(args.routes, usecols=["route_id", "paper_id"])

    merged = norm.merge(routes, on="paper_id", how="left")

    n_matched = int(merged["route_id"].notna().sum())
    n_unmatched = int(merged["route_id"].isna().sum())
    print(
        f"Total rows: {len(merged)}  |  "
        f"Matched to a route: {n_matched}  |  "
        f"No matching route: {n_unmatched}"
    )
    if n_unmatched:
        missing_pids = sorted(merged.loc[merged["route_id"].isna(), "paper_id"].unique())
        print(f"  paper_ids with no route: {missing_pids}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Wrote {len(merged)} rows → {args.output}")


if __name__ == "__main__":
    main()
