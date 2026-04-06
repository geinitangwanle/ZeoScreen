#!/usr/bin/env python3
"""Build ZeoScreen tables in CSV format with standardized schema.

Creates four core tables:
- synthesis_routes
- osda_table
- product_table
- adsorption_table
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_datasets"
OUT_DIR = ROOT / "database" / "csv_tables"

# Use T-site basis for normalized composition ratios.
T_SITE_COLUMNS = [
    "si",
    "al",
    "p",
    "ge",
    "ti",
    "in",
    "b",
    "ga",
    "ni",
    "mn",
    "fe",
    "co",
    "cr",
    "zn",
    "nb",
    "be",
    "w",
    "sn",
    "zr",
    "v",
    "ta",
    "hf",
    "as",
]

ROLE_PATTERNS = {
    "silica_source": [
        r"\bsilica\b",
        r"aerosil",
        r"ludox",
        r"cab[- ]?o[- ]?sil",
        r"fumed silica",
        r"tetraethylorthosilicate",
        r"\bteos\b",
        r"sodium silicate",
        r"water glass",
        r"silicic",
        r"silicate",
    ],
    "alumina_source": [
        r"alumina",
        r"aluminum",
        r"aluminium",
        r"boehmite",
        r"pseudo[- ]?boehmite",
        r"sodium aluminate",
        r"al\(oh\)",
        r"alcl",
        r"al\(no3\)",
    ],
    "alkali_source": [
        r"\bnaoh\b",
        r"\bkoh\b",
        r"\blioh\b",
        r"\bcsoh\b",
        r"\brboh\b",
        r"sodium hydroxide",
        r"potassium hydroxide",
        r"alkali",
    ],
    "mineralizer": [
        r"\bhf\b",
        r"hydrofluoric",
        r"ammonium fluoride",
        r"\bnh4f\b",
        r"fluoride",
        r"\bhcl\b",
        r"\bhno3\b",
        r"\bh2so4\b",
    ],
    "seed_source": [
        r"\bseed\b",
        r"seeding",
        r"crystal seed",
    ],
}

DENSE_KEYWORDS = {
    "dense",
    "quartz",
    "cristobalite",
    "tridymite",
    "coesite",
}


def to_snake(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "unnamed"
    if s[0].isdigit():
        s = f"col_{s}"
    return s


def split_tokens(text: str) -> List[str]:
    if not text or not isinstance(text, str):
        return []
    parts = re.split(r"[,;/\\+]| and ", text, flags=re.IGNORECASE)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out


def classify_precursors(precursors_raw: Optional[str], seed: object, seed_type: object) -> Dict[str, Optional[str]]:
    tokens = split_tokens(precursors_raw or "")
    roles: Dict[str, List[str]] = {k: [] for k in ROLE_PATTERNS}

    for token in tokens:
        tk = token.lower()
        for role, patterns in ROLE_PATTERNS.items():
            if any(re.search(pat, tk) for pat in patterns):
                roles[role].append(token)

    seed_values = []
    for v in (seed, seed_type):
        if pd.isna(v):
            continue
        if isinstance(v, (int, float, np.integer, np.floating)) and float(v) == 0.0:
            continue
        s = str(v).strip()
        if s.lower() in {"", "0", "0.0", "nan", "none", "null"}:
            continue
        seed_values.append(s)

    if seed_values and not roles["seed_source"]:
        roles["seed_source"].extend(seed_values)

    result: Dict[str, Optional[str]] = {}
    for role, values in roles.items():
        dedup = []
        seen = set()
        for v in values:
            key = v.lower()
            if key not in seen:
                dedup.append(v)
                seen.add(key)
        result[role] = "; ".join(dedup) if dedup else None
    return result


def is_nonempty(v: object) -> bool:
    if pd.isna(v):
        return False
    s = str(v).strip().lower()
    return s not in {"", "0", "nan", "none", "null"}


def infer_failure_label(row: pd.Series) -> str:
    fields = [
        row.get("primary_phase"),
        row.get("secondary_phase"),
        row.get("tertiary_phase"),
        row.get("primary_product"),
        row.get("secondary_product"),
        row.get("tertiary_product"),
    ]
    text = " | ".join(str(x).lower() for x in fields if is_nonempty(x))

    if "amorph" in text:
        return "failed_amorphous"
    if any(k in text for k in DENSE_KEYWORDS):
        return "failed_dense"

    secondary_present = is_nonempty(row.get("secondary_phase")) or is_nonempty(row.get("secondary_product"))
    tertiary_present = is_nonempty(row.get("tertiary_phase")) or is_nonempty(row.get("tertiary_product"))
    if secondary_present or tertiary_present:
        return "mixed_phase"

    if is_nonempty(row.get("primary_phase")):
        return "success"

    if any(k in text for k in {"unknown", "impur", "layered"}):
        return "failed_amorphous"

    if is_nonempty(row.get("primary_product")):
        return "failed_amorphous"

    return "failed_amorphous"


def json_object_from_row(row: pd.Series, cols: Iterable[str]) -> Optional[str]:
    out = {}
    for c in cols:
        v = row.get(c)
        if pd.isna(v):
            continue
        if isinstance(v, (np.floating, float)) and np.isnan(v):
            continue
        out[c] = v.item() if isinstance(v, np.generic) else v
    return json.dumps(out, ensure_ascii=True) if out else None


def clean_text(v: object) -> Optional[str]:
    if pd.isna(v):
        return None
    s = str(v).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return None
    return s


def first_nonempty(values: Iterable[object]) -> Optional[str]:
    for v in values:
        c = clean_text(v)
        if c is not None:
            return c
    return None


def merge_unique_text(values: Iterable[object], sep: str = "; ") -> Optional[str]:
    out: List[str] = []
    seen = set()
    for v in values:
        c = clean_text(v)
        if c is None:
            continue
        key = c.lower()
        if key not in seen:
            seen.add(key)
            out.append(c)
    return sep.join(out) if out else None


def main() -> None:
    zeosyn_path = RAW / "ZEOSYN.xlsx"
    osda_desc_path = RAW / "osda_descriptors.csv"
    zeolite_desc_path = RAW / "zeolite_descriptors.csv"

    routes = pd.read_excel(zeosyn_path)

    # Drop fully empty lines.
    routes = routes.dropna(how="all").reset_index(drop=True).copy()

    rename_map = {
        "Unnamed: 0": "zeosyn_row_id",
        "Code1": "primary_phase",
        "Code2": "secondary_phase",
        "Code3": "tertiary_phase",
        "precursors": "precursors_raw",
        "product1": "primary_product",
        "product2": "secondary_product",
        "product3": "tertiary_product",
    }
    routes = routes.rename(columns=rename_map)

    # Standardize all column names to snake_case.
    routes.columns = [to_snake(c) for c in routes.columns]

    # Ensure required columns exist.
    for c in [
        "primary_phase",
        "secondary_phase",
        "tertiary_phase",
        "precursors_raw",
        "primary_product",
        "secondary_product",
        "tertiary_product",
        "seed",
        "seed_type",
    ]:
        if c not in routes.columns:
            routes[c] = np.nan

    # Route id as stable PK.
    routes.insert(0, "route_id", np.arange(1, len(routes) + 1, dtype=int))

    # Precursor role split.
    role_rows = routes.apply(
        lambda r: classify_precursors(r.get("precursors_raw"), r.get("seed"), r.get("seed_type")), axis=1
    )
    role_df = pd.DataFrame(role_rows.tolist())
    routes = pd.concat([routes, role_df], axis=1)

    # Failure labels.
    routes["failure_label"] = routes.apply(infer_failure_label, axis=1)

    # Unified composition basis: T_total = 1.
    existing_t_cols = [c for c in T_SITE_COLUMNS if c in routes.columns]
    routes[existing_t_cols] = routes[existing_t_cols].apply(pd.to_numeric, errors="coerce")
    routes["t_site_total"] = routes[existing_t_cols].fillna(0).sum(axis=1)
    routes["normalization_basis"] = "t_total"

    # Normalize all composition columns from Si..V to the T_total basis.
    all_cols = list(routes.columns)
    if "si" in all_cols and "v" in all_cols:
        comp_start = all_cols.index("si")
        comp_end = all_cols.index("v")
        comp_cols = all_cols[comp_start : comp_end + 1]
    else:
        comp_cols = existing_t_cols

    for c in comp_cols:
        routes[c] = pd.to_numeric(routes[c], errors="coerce")
        routes[f"{c}_ratio_to_t"] = np.where(routes["t_site_total"] > 0, routes[c] / routes["t_site_total"], np.nan)

    # OSDA table.
    osda_desc = pd.read_csv(osda_desc_path)
    osda_desc = osda_desc.rename(columns={"Unnamed: 0": "descriptor_row_id", "osda smiles": "osda_smiles"})
    osda_desc.columns = [to_snake(c) for c in osda_desc.columns]

    osda_records = []
    for slot in (1, 2, 3):
        name_col = f"osda{slot}"
        smiles_col = f"osda{slot}_smiles"
        if name_col in routes.columns and smiles_col in routes.columns:
            part = routes[["route_id", name_col, smiles_col]].copy()
            part = part.rename(columns={name_col: "osda_name", smiles_col: "osda_smiles"})
            part["osda_slot"] = slot
            osda_records.append(part)

    if osda_records:
        osda_long = pd.concat(osda_records, ignore_index=True)
    else:
        osda_long = pd.DataFrame(columns=["route_id", "osda_name", "osda_smiles", "osda_slot"])

    osda_long["osda_smiles"] = osda_long["osda_smiles"].astype(str).str.strip()
    osda_long = osda_long[osda_long["osda_smiles"].notna() & (osda_long["osda_smiles"] != "") & (osda_long["osda_smiles"].str.lower() != "nan")]

    osda_names = (
        osda_long.dropna(subset=["osda_name"])
        .groupby("osda_smiles")["osda_name"]
        .agg(lambda x: sorted({str(v).strip() for v in x if str(v).strip() and str(v).strip().lower() != "nan"}))
        .reset_index()
    )

    osda_master = osda_desc.copy()
    if "osda_smiles" not in osda_master.columns:
        osda_master["osda_smiles"] = np.nan

    osda_master = osda_master.merge(osda_names, on="osda_smiles", how="outer")

    # Flatten names and choose canonical name.
    if "osda_name" in osda_master.columns:
        osda_master["name_aliases"] = osda_master["osda_name"].apply(
            lambda x: "; ".join(x) if isinstance(x, list) and x else None
        )
        osda_master["osda_name"] = osda_master["osda_name"].apply(lambda x: x[0] if isinstance(x, list) and x else None)
    else:
        osda_master["name_aliases"] = None
        osda_master["osda_name"] = None

    # Keep descriptors as json for compact schema.
    descriptor_exclude = {"descriptor_row_id", "osda_smiles", "mol_weight", "formal_charge", "osda_name", "name_aliases"}
    osda_descriptor_cols = [c for c in osda_master.columns if c not in descriptor_exclude]
    osda_master["descriptor_json"] = osda_master.apply(lambda r: json_object_from_row(r, osda_descriptor_cols), axis=1)

    osda_table = osda_master[["osda_name", "osda_smiles", "mol_weight", "formal_charge", "name_aliases", "descriptor_json"]].copy()
    osda_table = osda_table.drop_duplicates(subset=["osda_smiles"], keep="first")
    osda_table = osda_table[osda_table["osda_smiles"].notna() & (osda_table["osda_smiles"].astype(str).str.strip() != "")]
    osda_table.insert(0, "osda_id", np.arange(1, len(osda_table) + 1, dtype=int))

    # Link synthesis_routes to osda_table by SMILES (stable key).
    smiles_to_id = dict(zip(osda_table["osda_smiles"], osda_table["osda_id"]))
    for slot in (1, 2, 3):
        smiles_col = f"osda{slot}_smiles"
        id_col = f"osda{slot}_id"
        if smiles_col in routes.columns:
            cleaned = routes[smiles_col].astype(str).str.strip()
            cleaned = cleaned.where(cleaned.str.lower() != "nan", np.nan)
            routes[id_col] = cleaned.map(smiles_to_id).astype("Int64")
        else:
            routes[id_col] = pd.Series([pd.NA] * len(routes), dtype="Int64")

    # Product table.
    zeo_desc = pd.read_csv(zeolite_desc_path)
    zeo_desc = zeo_desc.rename(columns={"Unnamed: 0": "framework_code"})
    zeo_desc.columns = [to_snake(c) for c in zeo_desc.columns]
    if "framework_code" not in zeo_desc.columns:
        zeo_desc["framework_code"] = np.nan

    zeo_desc = zeo_desc.drop_duplicates(subset=["framework_code"], keep="first")

    pore_metric_cols = [
        c
        for c in zeo_desc.columns
        if any(k in c for k in ["density", "sphere", "volume", "channel", "pore", "ring_size", "surface"])
    ]

    product_table = routes[
        [
            "route_id",
            "primary_phase",
            "secondary_phase",
            "tertiary_phase",
            "primary_product",
            "secondary_product",
            "tertiary_product",
            "failure_label",
        ]
    ].copy()
    product_table = product_table.rename(columns={"primary_phase": "primary_topology"})

    product_table = product_table.merge(
        zeo_desc[["framework_code"] + pore_metric_cols],
        left_on="primary_topology",
        right_on="framework_code",
        how="left",
    )

    product_table["framework_descriptor_json"] = product_table.apply(
        lambda r: json_object_from_row(r, pore_metric_cols), axis=1
    )

    for c in ["framework_density", "largest_free_sphere", "accessible_volume_izc", "largest_included_sphere"]:
        if c not in product_table.columns:
            product_table[c] = np.nan

    product_table = product_table[
        [
            "route_id",
            "primary_topology",
            "secondary_phase",
            "tertiary_phase",
            "primary_product",
            "secondary_product",
            "tertiary_product",
            "failure_label",
            "framework_density",
            "largest_free_sphere",
            "accessible_volume_izc",
            "largest_included_sphere",
            "framework_descriptor_json",
        ]
    ].copy()

    # synthesis_routes should keep only input/process variables.
    duplicate_result_cols = [
        "primary_product",
        "secondary_product",
        "tertiary_product",
        "primary_phase",
        "secondary_phase",
        "tertiary_phase",
        "failure_label",
    ]
    routes = routes.drop(columns=[c for c in duplicate_result_cols if c in routes.columns])

    # Normalize literature metadata into paper_table and link by paper_id.
    routes["_doi_clean"] = routes["doi"].map(clean_text) if "doi" in routes.columns else None
    routes["_title_clean"] = routes["title"].map(clean_text) if "title" in routes.columns else None
    routes["_year_clean"] = routes["year"]
    if "year" in routes.columns:
        routes["_year_clean"] = pd.to_numeric(routes["year"], errors="coerce").astype("Int64")
    else:
        routes["_year_clean"] = pd.Series([pd.NA] * len(routes), dtype="Int64")

    def make_paper_key(r: pd.Series) -> Optional[str]:
        doi = r.get("_doi_clean")
        title = r.get("_title_clean")
        year = r.get("_year_clean")
        if clean_text(doi):
            return f"doi:{str(doi).lower()}"
        if clean_text(title):
            y = "" if pd.isna(year) else str(int(year))
            return f"title:{str(title).lower()}|year:{y}"
        # No reliable bibliographic identity.
        return None

    routes["_paper_key"] = routes.apply(make_paper_key, axis=1)

    agg_dict = {
        "_doi_clean": first_nonempty,
        "_title_clean": first_nonempty,
        "_year_clean": lambda x: next((int(v) for v in x if not pd.isna(v)), pd.NA),
    }
    if "abstract_keywords" in routes.columns:
        agg_dict["abstract_keywords"] = merge_unique_text

    paper_src = routes[routes["_paper_key"].notna()].copy()
    paper_table = paper_src.groupby("_paper_key", dropna=False).agg(agg_dict).reset_index()
    paper_table = paper_table.rename(
        columns={
            "_doi_clean": "doi",
            "_title_clean": "title",
            "_year_clean": "year",
        }
    )
    if "abstract_keywords" not in paper_table.columns:
        paper_table["abstract_keywords"] = None
    paper_table["journal"] = None
    paper_table["authors"] = None
    paper_table.insert(0, "paper_id", np.arange(1, len(paper_table) + 1, dtype=int))

    key_to_id = dict(zip(paper_table["_paper_key"], paper_table["paper_id"]))
    routes["paper_id"] = routes["_paper_key"].map(key_to_id).astype("Int64")

    paper_table = paper_table[
        ["paper_id", "doi", "title", "year", "journal", "authors", "abstract_keywords"]
    ].copy()

    # Keep only paper_id in synthesis_routes for literature linkage.
    routes = routes.drop(
        columns=[
            c
            for c in ["doi", "title", "year", "abstract_keywords", "_doi_clean", "_title_clean", "_year_clean", "_paper_key"]
            if c in routes.columns
        ]
    )

    # Adsorption table: empty scaffold for future manual enrichment.
    adsorption_table = pd.DataFrame(
        columns=[
            "adsorption_id",
            "route_id",
            "sample_label",
            "activation_condition",
            "gas",
            "uptake_value",
            "uptake_unit",
            "temperature_k",
            "pressure_bar",
            "measurement_type",
            "selectivity_target",
            "selectivity_value",
            "selectivity_basis",
            "pretreatment",
            "test_condition",
            "source",
            "source_doi",
            "notes",
        ]
    )

    # Write CSV files.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    routes.to_csv(OUT_DIR / "synthesis_routes.csv", index=False)
    osda_table.to_csv(OUT_DIR / "osda_table.csv", index=False)
    product_table.to_csv(OUT_DIR / "product_table.csv", index=False)
    adsorption_table.to_csv(OUT_DIR / "adsorption_table.csv", index=False)
    paper_table.to_csv(OUT_DIR / "paper_table.csv", index=False)

    print(f"Built CSV tables under: {OUT_DIR}")
    print(f"synthesis_routes rows: {len(routes)}")
    print(f"osda_table rows: {len(osda_table)}")
    print(f"product_table rows: {len(product_table)}")
    print(f"adsorption_table rows: {len(adsorption_table)}")
    print(f"paper_table rows: {len(paper_table)}")


if __name__ == "__main__":
    main()
