"""
Step 2: Normalize raw LLM-extracted records.Standardizes units, gas names, measurement types, and numeric fields.
        对 LLM 提取的原始记录进行规范化。标准化单位、气体名称、测量类型和数值字段。
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

# 进入下一步匹配流程时保留的标准列集合
OUTPUT_COLS = [
    "paper_id", "sample_label", "activation_condition", "gas",
    "uptake_value", "uptake_unit", "temperature_k", "pressure_bar",
    "measurement_type", "selectivity_target", "selectivity_value",
    "selectivity_basis", "pretreatment", "test_condition", "notes",
]

# ---------------------------------------------------------------------------
# Unit normalisation
# ---------------------------------------------------------------------------

# 每条规则: (正则模式, 规范单位, 数值换算因子)
# numeric_factor 表示 uptake_value 乘以该因子后转换到规范单位；
# 当前流程多数场景只做单位名标准化，数值通常不变。
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
    # 将同义单位映射到统一写法，降低后续统计和去重噪声
    if pd.isna(unit) or unit is None:
        return None
    for pattern, canonical, _ in UNIT_RULES:
        if re.search(pattern, unit, re.IGNORECASE):
            return canonical
    return str(unit).strip()


def normalize_gas(gas):
    # 气体名称标准化（如 carbon dioxide -> CO2）
    if pd.isna(gas) or gas is None:
        return None
    key = str(gas).strip().lower()
    return GAS_ALIASES.get(key, str(gas).strip())


def normalize_measurement_type(mtype):
    # 测试方法归一化到有限标签，避免自由文本造成类别碎片化
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

    # 逐列执行标准化：单位、气体、测试类型
    if "uptake_unit" in df.columns:
        df["uptake_unit"] = df["uptake_unit"].apply(normalize_unit)
    if "gas" in df.columns:
        df["gas"] = df["gas"].apply(normalize_gas)
    if "measurement_type" in df.columns:
        df["measurement_type"] = df["measurement_type"].apply(normalize_measurement_type)

    # 关键数值列强制转为数值，非法值记为 NaN
    for col in ("uptake_value", "temperature_k", "pressure_bar", "selectivity_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 删除既无 uptake 也无 selectivity 的噪声行（常见于 LLM 误提取）
    has_uptake = df.get("uptake_value", pd.Series(dtype=float)).notna()
    has_sel = df.get("selectivity_value", pd.Series(dtype=float)).notna()
    before = len(df)
    df = df[has_uptake | has_sel]
    print(f"Dropped {before - len(df)} rows with no uptake or selectivity value.")

    # 补齐输出 schema 并按固定列顺序导出
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = None
    df = df[OUTPUT_COLS]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} normalized records → {args.output}")


if __name__ == "__main__":
    main()
