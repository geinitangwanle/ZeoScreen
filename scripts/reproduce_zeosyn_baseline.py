#!/usr/bin/env python3
"""Reproduce ZeoSyn baseline RF framework classification.

Baseline definition (from user requirement):
- Sample unit: one synthesis route
- Input: gel composition + crystallization time/temperature + descriptors of the
  largest-volume OSDA in each route
- Output: final zeolite framework; all non-success routes collapsed to "Failed"
- Split: random 80/20 train/test
- Model: sklearn RandomForestClassifier with default parameters
- Metrics: held-out accuracy (+ report)
- Interpretation: SHAP global and class-wise sensitivity summaries
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "database" / "csv_tables"
OUT_DIR = ROOT / "outputs" / "baseline_rf"


def load_data(csv_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # 读取三张核心表：合成路线、产物标签、OSDA分子信息
    routes = pd.read_csv(csv_dir / "synthesis_routes.csv", low_memory=False)
    products = pd.read_csv(csv_dir / "product_table.csv", low_memory=False)
    osda = pd.read_csv(csv_dir / "osda_table.csv", low_memory=False)
    return routes, products, osda


def parse_osda_descriptors(osda_table: pd.DataFrame) -> pd.DataFrame:
    # 将 osda_table 中的 descriptor_json 展开成可直接建模的数值列
    rows: List[Dict[str, float]] = []
    for _, r in osda_table.iterrows():
        d: Dict[str, float] = {}
        raw = r.get("descriptor_json")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                for k, v in parsed.items():
                    d[k] = v
            except json.JSONDecodeError:
                pass

        # 同时保留显式标量列，避免仅靠 JSON 时丢失关键描述符
        d["mol_weight"] = r.get("mol_weight")
        d["formal_charge"] = r.get("formal_charge")
        d["osda_id"] = r.get("osda_id")
        rows.append(d)

    desc = pd.DataFrame(rows)
    desc = desc.rename(columns={c: f"osda_{c}" for c in desc.columns if c != "osda_id"})
    desc["osda_id"] = pd.to_numeric(desc["osda_id"], errors="coerce").astype("Int64")
    return desc


def choose_largest_volume_osda_id(route_row: pd.Series, osda_meta: Dict[int, Dict[str, float]]) -> Optional[int]:
    # 从一条路线最多 3 个 OSDA 候选中，选体积最大的那个
    candidates: List[int] = []
    for col in ("osda1_id", "osda2_id", "osda3_id"):
        v = route_row.get(col)
        if pd.isna(v):
            continue
        try:
            candidates.append(int(v))
        except (TypeError, ValueError):
            continue

    if not candidates:
        return None

    best_id: Optional[int] = None
    best_volume = -np.inf

    for oid in candidates:
        meta = osda_meta.get(oid)
        if not meta:
            continue
        vol = meta.get("osda_volume_mean_0")
        vol_val = float(vol) if vol is not None and pd.notna(vol) else np.nan
        if pd.notna(vol_val) and vol_val > best_volume:
            best_volume = vol_val
            best_id = oid

    if best_id is not None:
        return best_id

    # 兜底策略：如果体积不可用，取第一个能在描述符表中找到的 OSDA
    for oid in candidates:
        if oid in osda_meta:
            return oid
    return None


def build_features(
    routes: pd.DataFrame,
    products: pd.DataFrame,
    osda_desc: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    # 标签定义：成功样本用 primary_topology，失败样本统一折叠为 "Failed"
    prod = products[["route_id", "primary_topology", "failure_label"]].copy()
    merged = routes.merge(prod, on="route_id", how="inner")

    success_mask = (merged["failure_label"] == "success") & merged["primary_topology"].notna()
    y = pd.Series(np.where(success_mask, merged["primary_topology"], "Failed"), index=merged.index, name="target")

    # 凝胶组成特征：所有 *_ratio_to_t 归一化配比列
    gel_cols = sorted([c for c in merged.columns if c.endswith("_ratio_to_t")])

    # 结晶过程特征：时间与温度（若存在）
    process_cols = [c for c in ["cryst_time", "cryst_temp"] if c in merged.columns]

    # 拼接体积最大 OSDA 的分子描述符
    osda_desc = osda_desc.copy()
    osda_desc["osda_id"] = pd.to_numeric(osda_desc["osda_id"], errors="coerce").astype("Int64")
    osda_feature_cols = [c for c in osda_desc.columns if c != "osda_id"]
    osda_meta = osda_desc.set_index("osda_id")[osda_feature_cols].to_dict(orient="index")

    largest_ids = merged.apply(lambda r: choose_largest_volume_osda_id(r, osda_meta), axis=1)
    merged["largest_osda_id"] = largest_ids

    osda_feat_rows: List[Dict[str, float]] = []
    for oid in largest_ids:
        if oid is None or pd.isna(oid):
            osda_feat_rows.append({})
            continue
        osda_feat_rows.append(osda_meta.get(int(oid), {}))

    osda_feat = pd.DataFrame(osda_feat_rows, index=merged.index)

    X = pd.concat([merged[gel_cols + process_cols], osda_feat], axis=1)

    # RF 仅接收数值输入：全列强制转数值，无法转换的置为 NaN
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    aux = merged[["route_id", "largest_osda_id", "primary_topology", "failure_label"]].copy()
    return X, y, aux


def compute_shap_summaries(
    model: RandomForestClassifier,
    X_sample: pd.DataFrame,
    class_names: Iterable[str],
) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    """Return global mean|SHAP| and per-class mean|SHAP| if available.

    Returns:
    - global_importance: (n_features,)
    - class_importance: (n_classes, n_features) or None
    - shape_note: small note about shap output shape
    """
    # 计算 SHAP，并兼容不同版本/任务下的输出形状差异
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    class_importance: Optional[np.ndarray] = None
    shape_note = ""

    if isinstance(shap_values, list):
        # 旧格式：list[n_classes]，每个元素形状为 (n_samples, n_features)
        stacked = np.stack(shap_values, axis=0)
        # 堆叠后形状：(n_classes, n_samples, n_features)
        class_importance = np.mean(np.abs(stacked), axis=1)
        global_importance = np.mean(class_importance, axis=0)
        shap_plot_values = np.mean(np.abs(stacked), axis=0)
        shape_note = f"list format -> stacked shape={stacked.shape}"
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # 可能是 (n_samples, n_features, n_classes) 或 (n_classes, n_samples, n_features)
        if shap_values.shape[0] == len(X_sample):
            # 样本在第一维：(n_samples, n_features, n_classes)
            class_importance = np.mean(np.abs(shap_values), axis=0).T
            global_importance = np.mean(class_importance, axis=0)
            shap_plot_values = np.mean(np.abs(shap_values), axis=2)
            shape_note = f"array format samples-first shape={shap_values.shape}"
        else:
            # 否则按类别在第一维处理：(n_classes, n_samples, n_features)
            class_importance = np.mean(np.abs(shap_values), axis=1)
            global_importance = np.mean(class_importance, axis=0)
            shap_plot_values = np.mean(np.abs(shap_values), axis=0)
            shape_note = f"array format classes-first shape={shap_values.shape}"
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        global_importance = np.mean(np.abs(shap_values), axis=0)
        shap_plot_values = shap_values
        shape_note = f"binary/2d shape={shap_values.shape}"
    else:
        raise RuntimeError(f"Unsupported SHAP output format: type={type(shap_values)}")

    # 输出 SHAP 全局摘要图（按平均绝对值展示特征影响强度）
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_plot_values, X_sample, show=False, max_display=20)
    plt.tight_layout()

    return global_importance, class_importance, shape_note


def main() -> None:
    # 命令行参数：输入输出路径、数据划分、随机种子与解释参数
    parser = argparse.ArgumentParser(description="Reproduce ZeoSyn baseline RF framework classifier")
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--shap-sample-size", type=int, default=256)
    parser.add_argument("--min-class-size", type=int, default=1, help="Drop classes with count < this threshold")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    routes, products, osda = load_data(args.csv_dir)
    osda_desc = parse_osda_descriptors(osda)
    X, y, aux = build_features(routes, products, osda_desc)

    # 过滤过小类别，避免极端小样本类别造成评估噪声
    class_counts = y.value_counts()
    keep_classes = class_counts[class_counts >= args.min_class_size].index
    keep_mask = y.isin(keep_classes)
    X = X.loc[keep_mask].reset_index(drop=True)
    y = y.loc[keep_mask].reset_index(drop=True)
    aux = aux.loc[keep_mask].reset_index(drop=True)

    # 80/20 随机划分（默认），并进行中位数填补
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        shuffle=True,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # 使用 sklearn 默认参数训练随机森林分类器
    model = RandomForestClassifier(random_state=args.random_state)
    model.fit(X_train_imp, y_train)

    y_pred = model.predict(X_test_imp)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, digits=4, zero_division=0)

    # 保存核心评估信息
    metrics = {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_classes": int(y.nunique()),
        "test_size": args.test_size,
        "random_state": args.random_state,
        "held_out_accuracy": float(acc),
        "class_distribution": y.value_counts().to_dict(),
    }

    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=True, indent=2), encoding="utf-8")
    (args.out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    # 保存 RF 内置特征重要性
    fi = pd.DataFrame({
        "feature": X.columns,
        "rf_importance": model.feature_importances_,
    }).sort_values("rf_importance", ascending=False)
    fi.to_csv(args.out_dir / "rf_feature_importance.csv", index=False)

    # 在测试集上抽样做 SHAP，控制解释计算成本
    sample_size = min(args.shap_sample_size, len(X_test))
    sample_idx = np.random.RandomState(args.random_state).choice(len(X_test), size=sample_size, replace=False)
    X_test_imp_df = pd.DataFrame(X_test_imp, columns=X.columns)
    X_shap = X_test_imp_df.iloc[sample_idx].reset_index(drop=True)

    global_imp, class_imp, shape_note = compute_shap_summaries(model, X_shap, model.classes_)

    plt.savefig(args.out_dir / "shap_summary.png", dpi=180)
    plt.close()

    # 保存全局 SHAP 重要性
    shap_global = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": global_imp,
    }).sort_values("mean_abs_shap", ascending=False)
    shap_global.to_csv(args.out_dir / "shap_global_importance.csv", index=False)

    if class_imp is not None:
        # 若可用，额外输出按类别统计的 Top20 敏感特征
        rows = []
        classes = list(model.classes_)
        for ci, cname in enumerate(classes):
            vec = class_imp[ci]
            top_idx = np.argsort(vec)[::-1][:20]
            for fi_idx in top_idx:
                rows.append(
                    {
                        "class": str(cname),
                        "feature": X.columns[fi_idx],
                        "mean_abs_shap": float(vec[fi_idx]),
                    }
                )
        pd.DataFrame(rows).to_csv(args.out_dir / "shap_class_sensitivity_top20.csv", index=False)

    run_summary = {
        "held_out_accuracy": acc,
        "n_samples": len(X),
        "n_classes": int(y.nunique()),
        "n_features": int(X.shape[1]),
        "shap_sample_size": sample_size,
        "shap_shape_note": shape_note,
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(run_summary, ensure_ascii=True, indent=2), encoding="utf-8")

    # 终端打印关键结果，便于快速确认运行状态
    print(f"Saved outputs to: {args.out_dir}")
    print(f"Held-out accuracy: {acc:.4f}")
    print(f"Samples={len(X)}, Features={X.shape[1]}, Classes={y.nunique()}")


if __name__ == "__main__":
    main()
