"""
benchmark_eval.py — Evaluate a model against benchmark splits.

Per CLAUDE.md:
  - Benchmark results are for AUDIT ONLY
  - They must NEVER influence modeling, feature, or threshold decisions
  - B1 every 5 iterations, B2 every 10, B3 once per phase, B4 once ever
  - Results are appended to experiments/benchmark_audit.jsonl

This script evaluates the best accepted model against a specified
benchmark split and produces a structured audit entry.
"""

import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
AUDIT_PATH = EXPERIMENTS_DIR / "benchmark_audit.jsonl"

FEATURE_COLUMNS = [
    "segment_crash_rate",
    "segment_fatal_rate",
    "segment_injury_rate",
    "segment_pdo_rate",
    "month",
    "is_winter",
    "is_weekend_heavy",
    "segment_type_is_grid",
]


def evaluate_on_benchmark(
    model_iter: str,
    benchmark_split: str,
    model_type: str = "rf",
):
    """Evaluate a saved model against a benchmark split.

    Args:
        model_iter: iteration directory name (e.g. "iter_004")
        benchmark_split: split name (e.g. "b1", "b2", "b3", "b4")
        model_type: "lr" for logistic regression (needs scaler), "rf" for random forest
    """
    print("=" * 60)
    print(f"BENCHMARK EVALUATION")
    print(f"Model: {model_iter}")
    print(f"Benchmark: {benchmark_split}")
    print(f"WARNING: Results are for AUDIT ONLY.")
    print(f"Do NOT use these results to influence any modeling decisions.")
    print("=" * 60)

    # Load model
    iter_dir = EXPERIMENTS_DIR / model_iter
    model = joblib.load(iter_dir / "model.pkl")

    scaler = None
    if model_type == "lr":
        scaler = joblib.load(iter_dir / "scaler.pkl")

    # Load benchmark featured data
    bench_path = SPLITS_DIR / f"{benchmark_split}_featured.parquet"
    if not bench_path.exists():
        raise FileNotFoundError(f"Featured benchmark not found: {bench_path}")

    bench_df = pd.read_parquet(bench_path)
    X_bench = bench_df[FEATURE_COLUMNS]
    y_bench = bench_df["label"]

    print(f"\n  Benchmark rows: {len(bench_df):,}")
    print(f"  Positive rate:  {y_bench.mean():.4f}")

    # Predict
    if scaler is not None:
        X_scaled = scaler.transform(X_bench)
        y_pred = model.predict(X_scaled)
        y_proba = model.predict_proba(X_scaled)[:, 1]
    else:
        y_pred = model.predict(X_bench)
        y_proba = model.predict_proba(X_bench)[:, 1]

    # Metrics
    combined_recall = recall_score(y_bench, y_pred, zero_division=0)
    combined_precision = precision_score(y_bench, y_pred, zero_division=0)
    combined_f1 = f1_score(y_bench, y_pred, zero_division=0)
    flag_rate = y_pred.mean()

    fatal_mask = bench_df["has_fatal"] == 1
    injury_only_mask = (bench_df["has_injury"] == 1) & (bench_df["has_fatal"] == 0)

    fatal_recall = float(y_pred[fatal_mask.values].mean()) if fatal_mask.sum() > 0 else None
    injury_recall = float(y_pred[injury_only_mask.values].mean()) if injury_only_mask.sum() > 0 else None

    tn, fp, fn, tp = confusion_matrix(y_bench, y_pred).ravel()

    # Load val metrics for comparison
    metrics_path = iter_dir / "metrics.json"
    val_recall = None
    val_fatal_recall = None
    if metrics_path.exists():
        with open(metrics_path) as f:
            saved_metrics = json.load(f)
        val_recall = saved_metrics.get("val", {}).get("recall_combined")
        val_fatal_recall = saved_metrics.get("val", {}).get("recall_fatal")

    # Compute gaps
    benchmark_gap = None
    fatal_benchmark_gap = None
    if val_recall is not None:
        benchmark_gap = round(val_recall - combined_recall, 4)
    if val_fatal_recall is not None and fatal_recall is not None:
        fatal_benchmark_gap = round(val_fatal_recall - fatal_recall, 4)

    # Print results
    print(f"\n  --- {benchmark_split.upper()} BENCHMARK ---")
    print(f"  Recall (combined):  {combined_recall:.4f}")
    if fatal_recall is not None:
        print(f"  Recall (fatal):     {fatal_recall:.4f}")
    if injury_recall is not None:
        print(f"  Recall (injury):    {injury_recall:.4f}")
    print(f"  Precision:          {combined_precision:.4f}")
    print(f"  F1:                 {combined_f1:.4f}")
    print(f"  Flag rate:          {flag_rate:.4f}")
    print(f"  TP: {tp:,}  FP: {fp:,}  FN: {fn:,}  TN: {tn:,}")
    print(f"  Fatal windows:      {int(fatal_mask.sum())}")

    if benchmark_gap is not None:
        print(f"\n  --- GAPS (val vs benchmark) ---")
        print(f"  Combined recall gap: {benchmark_gap:.4f} (val {val_recall:.4f} vs bench {combined_recall:.4f})")
        if benchmark_gap > 0.05:
            print(f"  WARNING: Benchmark gap exceeds 0.05 threshold")

    if fatal_benchmark_gap is not None:
        print(f"  Fatal recall gap:    {fatal_benchmark_gap:.4f} (val {val_fatal_recall:.4f} vs bench {fatal_recall:.4f})")

    # Build audit entry
    audit_entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_iter": model_iter,
        "benchmark_split": benchmark_split,
        "model_class": type(model).__name__,
        "purpose": "audit only — do not use for modeling decisions",
        "metrics": {
            "recall_combined": round(combined_recall, 4),
            "recall_fatal": round(fatal_recall, 4) if fatal_recall is not None else None,
            "recall_injury": round(injury_recall, 4) if injury_recall is not None else None,
            "precision": round(combined_precision, 4),
            "f1": round(combined_f1, 4),
            "flag_rate": round(flag_rate, 4),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "support_positive": int(y_bench.sum()),
            "support_total": int(len(y_bench)),
            "fatal_windows_total": int(fatal_mask.sum()),
        },
        "gaps": {
            "val_recall_combined": val_recall,
            "bench_recall_combined": round(combined_recall, 4),
            "benchmark_gap": benchmark_gap,
            "val_recall_fatal": val_fatal_recall,
            "bench_recall_fatal": round(fatal_recall, 4) if fatal_recall is not None else None,
            "fatal_benchmark_gap": fatal_benchmark_gap,
            "gap_warning": bool(benchmark_gap > 0.05) if benchmark_gap is not None else None,
        },
    }

    # Append to audit log
    with open(AUDIT_PATH, "a") as f:
        f.write(json.dumps(audit_entry) + "\n")

    print(f"\n  Audit entry appended to {AUDIT_PATH}")
    print(f"\n  REMINDER: These results are for audit only.")
    print(f"  Do NOT tune the model or features based on benchmark performance.")
    print("=" * 60)

    return audit_entry


if __name__ == "__main__":
    import sys

    # Default: evaluate iter_004 on b1
    model_iter = "iter_004"
    benchmark_split = "b1"
    model_type = "rf"

    if len(sys.argv) >= 2:
        model_iter = sys.argv[1]
    if len(sys.argv) >= 3:
        benchmark_split = sys.argv[2]
    if len(sys.argv) >= 4:
        model_type = sys.argv[3]

    evaluate_on_benchmark(model_iter, benchmark_split, model_type)