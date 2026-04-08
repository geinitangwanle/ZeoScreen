"""
单篇论文信息提取
用法: python extract_single.py your_paper.pdf
"""

import sys
import json
import requests
import pdfplumber


def read_pdf(pdf_path: str) -> str:
    """从 PDF 提取文本"""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n".join(text_parts)


def extract_info(text: str) -> dict:
    """调用 Qwen 2.5 提取结构化信息"""

    # 如果文本太长，截取前 12000 字符（约覆盖一篇论文的核心部分）
    if len(text) > 12000:
        text = text[:12000]

    prompt = f"""You are a research paper analysis assistant.
Read the following paper text and extract information in **valid JSON format only**.
Do not include any text outside the JSON.

Required JSON structure:
{{
    "title": "paper title",
    "authors": ["author1", "author2"],
    "key_findings": [
        "finding 1",
        "finding 2",
        "finding 3"
    ],
    "methods": [
        "method 1",
        "method 2"
    ],
    "numerical_results": [
        {{
            "metric": "metric name",
            "value": "value",
            "context": "brief context"
        }}
    ],
    "dataset": "dataset(s) used",
    "limitations": ["limitation 1"]
}}

Paper text:
---
{text}
---

Respond ONLY with valid JSON, no markdown fences, no explanation."""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:14b",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,      # 低温度 = 更精确的提取
                "num_predict": 2048,      # 最大输出 token 数
            }
        },
        timeout=120
    )

    raw = response.json()["response"].strip()

    # 清理可能的 markdown 代码块标记
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "JSON 解析失败", "raw_response": raw}


def main():
    if len(sys.argv) < 2:
        print("用法: python extract_single.py <pdf_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    print(f"📄 正在读取: {pdf_path}")
    text = read_pdf(pdf_path)
    print(f"📝 提取到 {len(text)} 个字符")

    print("🤖 正在调用 Qwen 2.5 分析...")
    result = extract_info(text)

    # 输出结果
    print("\n" + "=" * 60)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 保存到文件
    output_path = pdf_path.rsplit(".", 1)[0] + "_extracted.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 结果已保存到: {output_path}")


if __name__ == "__main__":
    main()