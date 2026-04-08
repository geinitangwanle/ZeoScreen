"""
Step 1: Extract adsorption data from paper files via local LLM (Qwen/Ollama).
        先读取论文文件（优先 TXT，或 PDF），再抽取与吸附相关的数据并输出 JSONL。

Input:  data/raw_text/{paper_id}.txt or data/raw_text/{paper_id}.pdf
Output: outputs/extracted_raw.jsonl (appended, one JSON object per line)

Usage:
    # Process all numeric txt/pdf files in data/raw_text/
    python src/extract_adsorption.py

    # Process specific paper_ids only
    python src/extract_adsorption.py --paper_ids 12 34 56

    # Customize local model endpoint
    python src/extract_adsorption.py --model qwen2.5:14b --api-url http://localhost:11434/api/generate
"""

import os
import re
import json
import argparse

import pdfplumber
import requests

# 项目根目录（src/ 的上一级）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_TEXT_DIR = os.path.join(ROOT, "data", "raw_text")
OUTPUT_FILE = os.path.join(ROOT, "outputs", "extracted_raw.jsonl")

DEFAULT_MODEL = "qwen2.5:14b"
DEFAULT_API_URL = "http://localhost:11434/api/generate"
DEFAULT_MAX_TEXT_CHARS = 12000

# 用于初筛段落的关键词正则：只有命中这些词的段落才会送给 LLM
# 目的是降低调用成本，避免把无关段落（如详细合成步骤）发送给模型
ADSORPTION_KEYWORDS = re.compile(
    r"\b(adsorption|uptake|isotherm|BET|surface.area|pore.vol|CO2|CH4|N2\b|H2O|"
    r"selectivity|breakthrough|loading|capacity|henry|langmuir|freundlich|"
    r"volumetric|gravimetric|TGA|physisorption|chemisorption)\b",
    re.IGNORECASE,
)

# 提示词保持与下游表结构一致
EXTRACTION_PROMPT = """\
You are a materials science expert specializing in zeolite adsorption. Extract ALL \
adsorption / gas-uptake data from the text segments below (from a zeolite synthesis paper).

Return a JSON array. Each element represents one measurement and must contain these \
fields (use null for any field not mentioned):

  sample_label        - string  - sample name or ID (e.g. "SSZ-13", "calcined ZSM-5")
  activation_condition- string  - how the sample was activated before measurement
                                  (e.g. "calcined at 550 C", "degassed at 300 C under vacuum")
  gas                 - string  - adsorbate gas/vapor (e.g. "CO2", "N2", "H2O", "CH4")
  uptake_value        - number  - measured uptake amount
  uptake_unit         - string  - unit of uptake (e.g. "mmol/g", "wt%", "cm3/g STP", "mg/g")
  temperature_k       - number  - measurement temperature in Kelvin (convert from C if needed)
  pressure_bar        - number  - pressure in bar (convert from kPa/mmHg/atm/Pa if needed)
  measurement_type    - string  - technique (e.g. "volumetric", "gravimetric", "breakthrough",
                                  "TGA", "IAST", "Henry")
  selectivity_target  - string or null  - gas pair (e.g. "CO2/N2", "CO2/CH4")
  selectivity_value   - number or null
  selectivity_basis   - string or null  - basis (e.g. "adsorption ratio", "IAST", "Henry")
  pretreatment        - string or null  - any sample pre-treatment details
  notes               - string or null  - any other relevant details

If there is no adsorption data, return an empty array [].

Text:
{text}

Return ONLY valid JSON - no explanation, no markdown fences."""


def read_pdf(pdf_path: str) -> str:
    """从 PDF 提取文本。"""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def find_adsorption_segments(text: str, max_merged_chars: int = 8000) -> list[str]:
    """返回文本中包含吸附关键词的段落，并贪心合并成较少 chunk。"""
    paragraphs = re.split(r"\n{2,}", text)
    relevant = [p.strip() for p in paragraphs if ADSORPTION_KEYWORDS.search(p)]
    if not relevant:
        return []

    merged, buf = [], relevant[0]
    for seg in relevant[1:]:
        if len(buf) + len(seg) + 4 <= max_merged_chars:
            buf += "\n\n" + seg
        else:
            merged.append(buf)
            buf = seg
    merged.append(buf)
    return merged


def clean_markdown_fence(raw: str) -> str:
    """清理模型可能返回的 markdown 代码块围栏。"""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw.lstrip("`")
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    return raw.strip()


def call_llm(text: str, api_url: str, model: str) -> list[dict]:
    """调用本地 Ollama 接口并解析为记录列表。"""
    response = requests.post(
        api_url,
        json={
            "model": model,
            "prompt": EXTRACTION_PROMPT.format(text=text),
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 2048,
            },
        },
        timeout=180,
    )
    response.raise_for_status()

    raw = response.json().get("response", "")
    raw = clean_markdown_fence(raw)

    records = json.loads(raw)
    if not isinstance(records, list):
        records = [records]
    return records


def read_paper_text(paper_id: int, input_dir: str) -> str:
    """优先读 txt；若不存在则读同名 pdf。"""
    txt_path = os.path.join(input_dir, f"{paper_id}.txt")
    pdf_path = os.path.join(input_dir, f"{paper_id}.pdf")

    if os.path.exists(txt_path):
        with open(txt_path, encoding="utf-8") as f:
            return f.read()
    if os.path.exists(pdf_path):
        return read_pdf(pdf_path)
    return ""


def extract_from_paper(
    paper_id: int,
    input_dir: str,
    api_url: str,
    model: str,
    max_text_chars: int,
) -> list[dict]:
    """处理单篇文献：读取文本 -> 关键词初筛 -> 调用本地 LLM -> 附加 paper_id。"""
    text = read_paper_text(paper_id, input_dir)
    if not text:
        print(f"[SKIP] No txt/pdf file for paper_id={paper_id}")
        return []

    segments = find_adsorption_segments(text)
    if not segments:
        print(f"[SKIP] No adsorption segments found in paper_id={paper_id}")
        return []

    combined = "\n\n---\n\n".join(segments)
    if len(combined) > max_text_chars:
        combined = combined[:max_text_chars]

    print(f"[INFO] paper_id={paper_id}: {len(segments)} segment(s), {len(combined)} chars")

    try:
        records = call_llm(combined, api_url=api_url, model=model)
    except requests.RequestException as e:
        print(f"[ERROR] API request failed for paper_id={paper_id}: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parse failed for paper_id={paper_id}: {e}")
        return []

    for r in records:
        r["paper_id"] = paper_id
    return records


def discover_paper_ids(input_dir: str) -> list[int]:
    """自动发现目录下以数字命名的 .txt/.pdf 文件。"""
    ids = set()
    for f in os.scandir(input_dir):
        if not f.is_file():
            continue
        stem, ext = os.path.splitext(f.name)
        if ext.lower() not in {".txt", ".pdf"}:
            continue
        if stem.isdigit():
            ids.add(int(stem))
    return sorted(ids)


def main():
    parser = argparse.ArgumentParser(
        description="Extract adsorption data from txt/pdf paper files via local LLM (Ollama)."
    )
    parser.add_argument(
        "--paper_ids", nargs="+", type=int,
        help="Paper IDs to process (default: auto-scan numeric *.txt/*.pdf in input-dir)",
    )
    parser.add_argument("--input-dir", default=RAW_TEXT_DIR)
    parser.add_argument("--output", default=OUTPUT_FILE)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-text-chars", type=int, default=DEFAULT_MAX_TEXT_CHARS)
    parser.add_argument("--pdf-path", help="Process one specific PDF file directly.")
    parser.add_argument("--paper-id", type=int, help="paper_id used with --pdf-path.")
    args = parser.parse_args()

    if args.pdf_path:
        if args.paper_id is not None:
            paper_ids = [args.paper_id]
        else:
            stem = os.path.splitext(os.path.basename(args.pdf_path))[0]
            if stem.isdigit():
                paper_ids = [int(stem)]
            else:
                paper_ids = [0]
                print("[WARN] Non-numeric pdf filename; fallback paper_id=0. "
                      "Use --paper-id to set an explicit ID.")
    elif args.paper_ids:
        paper_ids = args.paper_ids
    else:
        paper_ids = discover_paper_ids(args.input_dir)

    if not paper_ids:
        print(f"No numeric .txt/.pdf files found in {args.input_dir}.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "a", encoding="utf-8") as out:
        for pid in paper_ids:
            if args.pdf_path:
                text = read_pdf(args.pdf_path)
                segments = find_adsorption_segments(text)
                if not segments:
                    print(f"[SKIP] No adsorption segments found in {args.pdf_path}")
                    records = []
                else:
                    combined = "\n\n---\n\n".join(segments)
                    if len(combined) > args.max_text_chars:
                        combined = combined[:args.max_text_chars]
                    print(f"[INFO] file={args.pdf_path}: {len(segments)} segment(s), {len(combined)} chars")
                    try:
                        records = call_llm(combined, api_url=args.api_url, model=args.model)
                    except requests.RequestException as e:
                        print(f"[ERROR] API request failed for {args.pdf_path}: {e}")
                        records = []
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] JSON parse failed for {args.pdf_path}: {e}")
                        records = []
                    for r in records:
                        r["paper_id"] = pid
            else:
                records = extract_from_paper(
                    paper_id=pid,
                    input_dir=args.input_dir,
                    api_url=args.api_url,
                    model=args.model,
                    max_text_chars=args.max_text_chars,
                )
            for r in records:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  -> {len(records)} record(s) written for paper_id={pid}")


if __name__ == "__main__":
    main()
