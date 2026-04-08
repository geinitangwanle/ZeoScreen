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
