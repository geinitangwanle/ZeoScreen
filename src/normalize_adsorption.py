"""
Step 2: Normalize raw LLM-extracted records.
        Standardizes units, gas names, measurement types, and numeric fields.

Input:  outputs/extracted_raw.jsonl
Output: outputs/normalized_records.csv

Usage:
    python src/normalize_adsorption.py
    python src/normalize_adsorption.py --input outputs/extracted_raw.jsonl
"""

import os
import re
import json
import argparse

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILE = os.path.join(ROOT, "outputs", "extracted_raw.jsonl")
OUTPUT_FILE = os.path.join(ROOT, "outputs", "normalized_records.csv")

# Columns that will be carried into the next step
OUTPUT_COLS = [
    "paper_id", "sample_label", "activation_condition", "gas",
    "uptake_value", "uptake_unit", "temperature_k", "pressure_bar",
    "measurement_type", "selectivity_target", "selectivity_value",
    "selectivity_basis", "pretreatment", "test_condition", "notes",
]

# ---------------------------------------------------------------------------
# Unit normalisation
# ---------------------------------------------------------------------------

# Each entry: (regex pattern, canonical string, numeric_factor)
# numeric_factor: multiply uptake_value by this to convert to the canonical unit.
# For most cases the LLM already gives the right number; we just rename.
UNIT_RULES = [
    (r"mmol\s*/\s*g",               "mmol/g",       1.0),
    (r"mol\s*/\s*kg",               "mmol/g",       1.0),   # 1 mol/kg = 1 mmol/g
    (r"cm3?\s*/\s*g(\s*STP)?",      "cm3/g STP",    1.0),
    (r"cc\s*/\s*g",                 "cm3/g STP",    1.0),
    (r"mL\s*/\s*g",                 "cm3/g STP",    1.0),
    (r"wt\.?\s*%",                  "wt%",          1.0),
    (r"mg\s*/\s*g",                 "mg/g",         1.0),
    (r"g\s*/\s*g",                  "g/g",          1.0),
    (r"mol\s*/\s*mol",              "mol/mol",      1.0),
]

GAS_ALIASES = {
    "carbon dioxide": "CO2",
    "co 2": "CO2",
    "nitrogen": "N2",
    "methane": "CH4",
    "water": "H2O",
    "water vapor": "H2O",
    "water vapour": "H2O",
    "hydrogen": "H2",
    "oxygen": "O2",
    "argon": "Ar",
    "helium": "He",
    "ethane": "C2H6",
    "ethylene": "C2H4",
    "propane": "C3H8",
    "propylene": "C3H6",
    "sulfur dioxide": "SO2",
    "hydrogen sulfide": "H2S",
    "ammonia": "NH3",
}


def normalize_unit(unit):
    if pd.isna(unit) or unit is None:
        return None
    for pattern, canonical, _ in UNIT_RULES:
        if re.search(pattern, unit, re.IGNORECASE):
            return canonical
    return str(unit).strip()


def normalize_gas(gas):
    if pd.isna(gas) or gas is None:
        return None
    key = str(gas).strip().lower()
    return GAS_ALIASES.get(key, str(gas).strip())


def normalize_measurement_type(mtype):
    if pd.isna(mtype) or mtype is None:
        return None
    m = str(mtype).lower()
    if "gravimetric" in m or "tga" in m:
        return "gravimetric"
    if "volumetric" in m:
        return "volumetric"
    if "breakthrough" in m:
        return "breakthrough"
    if "iast" in m:
        return "IAST"
    if "henry" in m:
        return "Henry"
    if "gc" in m or "gas chromatog" in m:
        return "GC"
    return str(mtype).strip()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        print("No records found in input file.")
        return

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} raw records.")

    # Normalize fields
    if "uptake_unit" in df.columns:
        df["uptake_unit"] = df["uptake_unit"].apply(normalize_unit)
    if "gas" in df.columns:
        df["gas"] = df["gas"].apply(normalize_gas)
    if "measurement_type" in df.columns:
        df["measurement_type"] = df["measurement_type"].apply(normalize_measurement_type)

    # Coerce numerics
    for col in ("uptake_value", "temperature_k", "pressure_bar", "selectivity_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows that have neither uptake nor selectivity data (LLM noise)
    has_uptake = df.get("uptake_value", pd.Series(dtype=float)).notna()
    has_sel = df.get("selectivity_value", pd.Series(dtype=float)).notna()
    before = len(df)
    df = df[has_uptake | has_sel]
    print(f"Dropped {before - len(df)} rows with no uptake or selectivity value.")

    # Ensure all output columns exist
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = None
    df = df[OUTPUT_COLS]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} normalized records → {args.output}")


if __name__ == "__main__":
    main()
