"""
Step 1: Find adsorption-related paragraphs in a paper's text file and extract
        structured JSON via Claude LLM.

Input:  data/raw_text/{paper_id}.txt        (one file per paper)
Output: outputs/extracted_raw.jsonl         (appended, one JSON object per line)

Usage:
    # Process all txt files in data/raw_text/
    python src/extract_adsorption.py

    # Process specific paper_ids only
    python src/extract_adsorption.py --paper_ids 12 34 56
"""

import os
import re
import json
import argparse

import anthropic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_TEXT_DIR = os.path.join(ROOT, "data", "raw_text")
OUTPUT_FILE = os.path.join(ROOT, "outputs", "extracted_raw.jsonl")

ADSORPTION_KEYWORDS = re.compile(
    r"\b(adsorption|uptake|isotherm|BET|surface.area|pore.vol|CO2|CH4|N2\b|H2O|"
    r"selectivity|breakthrough|loading|capacity|henry|langmuir|freundlich|"
    r"volumetric|gravimetric|TGA|physisorption|chemisorption)\b",
    re.IGNORECASE,
)

EXTRACTION_PROMPT = """\
You are a materials science expert specializing in zeolite adsorption. Extract ALL \
adsorption / gas-uptake data from the text segments below (from a zeolite synthesis paper).

Return a JSON array. Each element represents one measurement and must contain these \
fields (use null for any field not mentioned):

  sample_label        – string  – sample name or ID (e.g. "SSZ-13", "calcined ZSM-5")
  activation_condition– string  – how the sample was activated before measurement
                                  (e.g. "calcined at 550 °C", "degassed at 300 °C under vacuum")
  gas                 – string  – adsorbate gas/vapor (e.g. "CO2", "N2", "H2O", "CH4")
  uptake_value        – number  – measured uptake amount
  uptake_unit         – string  – unit of uptake (e.g. "mmol/g", "wt%", "cm3/g STP", "mg/g")
  temperature_k       – number  – measurement temperature in Kelvin (convert from °C if needed)
  pressure_bar        – number  – pressure in bar (convert from kPa/mmHg/atm/Pa if needed)
  measurement_type    – string  – technique (e.g. "volumetric", "gravimetric", "breakthrough",
                                  "TGA", "IAST", "Henry")
  selectivity_target  – string or null  – gas pair (e.g. "CO2/N2", "CO2/CH4")
  selectivity_value   – number or null
  selectivity_basis   – string or null  – basis (e.g. "adsorption ratio", "IAST", "Henry")
  pretreatment        – string or null  – any sample pre-treatment details
  notes               – string or null  – any other relevant details

If there is no adsorption data, return an empty array [].

Text:
{text}

Return ONLY valid JSON — no explanation, no markdown fences."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_adsorption_segments(text: str, max_merged_chars: int = 8000) -> list[str]:
    """Return paragraph-level chunks that contain adsorption-related keywords."""
    paragraphs = re.split(r"\n{2,}", text)
    relevant = [p.strip() for p in paragraphs if ADSORPTION_KEYWORDS.search(p)]
    if not relevant:
        return []

    # Greedily merge nearby short paragraphs into one context window
    merged, buf = [], relevant[0]
    for seg in relevant[1:]:
        if len(buf) + len(seg) + 4 <= max_merged_chars:
            buf += "\n\n" + seg
        else:
            merged.append(buf)
            buf = seg
    merged.append(buf)
    return merged


def call_llm(text: str, client: anthropic.Anthropic) -> list[dict]:
    message = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        messages=[{"role": "user", "content": EXTRACTION_PROMPT.format(text=text)}],
    )
    raw = message.content[0].text.strip()
    # Strip accidental markdown code fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw.strip())
    records = json.loads(raw)
    if not isinstance(records, list):
        records = [records]
    return records


def extract_from_paper(paper_id: int, client: anthropic.Anthropic) -> list[dict]:
    txt_path = os.path.join(RAW_TEXT_DIR, f"{paper_id}.txt")
    if not os.path.exists(txt_path):
        print(f"[SKIP] No text file for paper_id={paper_id}")
        return []

    text = open(txt_path, encoding="utf-8").read()
    segments = find_adsorption_segments(text)
    if not segments:
        print(f"[SKIP] No adsorption segments found in paper_id={paper_id}")
        return []

    combined = "\n\n---\n\n".join(segments)
    print(f"[INFO] paper_id={paper_id}: {len(segments)} segment(s), {len(combined)} chars")

    try:
        records = call_llm(combined, client)
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse failed for paper_id={paper_id}: {e}")
        return []

    for r in records:
        r["paper_id"] = paper_id
    return records


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract adsorption data from raw paper text via Claude LLM."
    )
    parser.add_argument(
        "--paper_ids", nargs="+", type=int,
        help="Paper IDs to process (default: all *.txt files in data/raw_text/)",
    )
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    if args.paper_ids:
        paper_ids = args.paper_ids
    else:
        paper_ids = sorted(
            int(f.name[:-4])
            for f in os.scandir(RAW_TEXT_DIR)
            if f.name.endswith(".txt") and f.name[:-4].isdigit()
        )

    if not paper_ids:
        print(f"No .txt files found in {RAW_TEXT_DIR}. Add paper text files first.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as out:
        for pid in paper_ids:
            records = extract_from_paper(pid, client)
            for r in records:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  → {len(records)} record(s) written for paper_id={pid}")


if __name__ == "__main__":
    main()
