# ZeoScreen

构建标准化数据表（CSV）：

```bash
python scripts/build_database.py
```

输出目录：

- `database/csv_tables/synthesis_routes.csv`
- `database/csv_tables/osda_table.csv`
- `database/csv_tables/product_table.csv`
- `database/csv_tables/adsorption_table.csv`
- `database/csv_tables/paper_table.csv`

说明：

- `paper_table` 采用“每篇文献一行”规范化（`doi` 优先，否则 `title+year`）。
- `synthesis_routes.paper_id` 允许为空；当原始数据缺失 DOI 且缺失标题时，不强行映射到文献表。

复现 ZeoSyn baseline（Random Forest）：

```bash
python scripts/reproduce_zeosyn_baseline.py
```

输出目录：

- `outputs/baseline_rf/metrics.json`
- `outputs/baseline_rf/classification_report.txt`
- `outputs/baseline_rf/rf_feature_importance.csv`
- `outputs/baseline_rf/shap_global_importance.csv`
- `outputs/baseline_rf/shap_class_sensitivity_top20.csv`
- `outputs/baseline_rf/shap_summary.png`

用 DeepSeek API 抽取吸附数据（候选定位 → 分块 → LLM 抽取 → 标准化 → route 匹配）：

```bash
export DEEPSEEK_API_KEY=your_key
python scripts/extract_adsorption.py \
  --input path/to/paper.pdf \
  --out-dir outputs/adsorption_review
```

主要产物：

- `outputs/adsorption_review/candidates.jsonl`
- `outputs/adsorption_review/chunks.jsonl`
- `outputs/adsorption_review/llm_raw.jsonl`
- `outputs/adsorption_review/normalized_adsorption_records.jsonl`
- `database/csv_tables/adsorption_table.csv`（默认追加写入）

说明：`--input` 支持 `.pdf`、`.json`、`.jsonl`，也支持目录（可递归读取其中 PDF/JSON 文件）。
