"""
naive_baseline.py — Naive baselines to contextualize model performance.

Two baselines:
  1. Historical Top-K: flag the K segments with the highest historical
     crash rate. Same segments every month, no model involved.
  2. Random baseline: flag segments randomly at the same flag rate
     as the current model.

If the ML model cannot beat "just flag the historically worst segments,"
it adds no value beyond a sorted list.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

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


def evaluate_predictions(y_true, y_pred, featured_df, name):
    """Compute standard metrics for a set of predictions."""
    combined_recall = recall_score(y_true, y_pred, zero_division=0)
    combined_precision = precision_score(y_true, y_pred, zero_division=0)
    combined_f1 = f1_score(y_true, y_pred, zero_division=0)
    flag_rate = y_pred.mean()

    fatal_mask = featured_df["has_fatal"] == 1
    injury_only_mask = (featured_df["has_injury"] == 1) & (featured_df["has_fatal"] == 0)

    fatal_recall = float(y_pred[fatal_mask.values].mean()) if fatal_mask.sum() > 0 else None
    injury_recall = float(y_pred[injury_only_mask.values].mean()) if injury_only_mask.sum() > 0 else None

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return {
        "name": name,
        "recall_combined": round(combined_recall, 4),
        "recall_fatal": round(float(fatal_recall), 4) if fatal_recall is not None else None,
        "recall_injury": round(float(injury_recall), 4) if injury_recall is not None else None,
        "precision": round(combined_precision, 4),
        "f1": round(combined_f1, 4),
        "flag_rate": round(flag_rate, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "fatal_windows_total": int(fatal_mask.sum()),
    }


def run_top_k_baseline(val_df, flag_rates):
    """Flag the top-K segments by historical crash rate.

    For each target flag rate, find the crash_rate threshold that
    produces approximately that flag rate, then flag all windows
    for segments above the threshold.

    This is the simplest possible "model" — no temporal awareness,
    no seasonality, just historically dangerous locations flagged
    permanently.
    """
    results = []

    for target_rate in flag_rates:
        # Find the crash rate threshold that produces this flag rate
        threshold = np.quantile(
            val_df["segment_crash_rate"],
            1 - target_rate
        )

        y_pred = (val_df["segment_crash_rate"] >= threshold).astype(int).values

        actual_rate = y_pred.mean()
        name = f"Top-K (flag_rate={actual_rate:.3f})"

        metrics = evaluate_predictions(
            val_df["label"].values, y_pred, val_df, name
        )
        metrics["threshold_crash_rate"] = round(float(threshold), 4)
        metrics["target_flag_rate"] = target_rate
        results.append(metrics)

    return results


def run_top_k_fatal_baseline(val_df, flag_rates):
    """Flag segments by historical FATAL rate specifically.

    Same as top-K but ranks by segment_fatal_rate instead of
    segment_crash_rate. Tests whether fatal-specific history
    is a better naive predictor than general crash history.
    """
    results = []

    for target_rate in flag_rates:
        threshold = np.quantile(
            val_df["segment_fatal_rate"],
            1 - target_rate
        )

        y_pred = (val_df["segment_fatal_rate"] >= threshold).astype(int).values

        actual_rate = y_pred.mean()
        name = f"Top-K Fatal (flag_rate={actual_rate:.3f})"

        metrics = evaluate_predictions(
            val_df["label"].values, y_pred, val_df, name
        )
        metrics["threshold_fatal_rate"] = round(float(threshold), 6)
        metrics["target_flag_rate"] = target_rate
        results.append(metrics)

    return results


def run_random_baseline(val_df, flag_rate, n_trials=10):
    """Flag segments randomly at a given flag rate.

    Averages over multiple trials to get a stable estimate.
    This is the absolute floor — if the model cannot beat random
    flagging, it has learned nothing.
    """
    rng = np.random.RandomState(42)
    recalls = []
    fatal_recalls = []
    precisions = []

    fatal_mask = val_df["has_fatal"] == 1

    for trial in range(n_trials):
        y_pred = rng.binomial(1, flag_rate, size=len(val_df))
        r = recall_score(val_df["label"], y_pred, zero_division=0)
        p = precision_score(val_df["label"], y_pred, zero_division=0)
        recalls.append(r)
        precisions.append(p)
        if fatal_mask.sum() > 0:
            fatal_recalls.append(y_pred[fatal_mask.values].mean())

    return {
        "name": f"Random (flag_rate={flag_rate:.3f})",
        "recall_combined": round(float(np.mean(recalls)), 4),
        "recall_fatal": round(float(np.mean(fatal_recalls)), 4) if fatal_recalls else None,
        "precision": round(float(np.mean(precisions)), 4),
        "flag_rate": flag_rate,
    }


def main():
    print("=" * 60)
    print("NAIVE BASELINE EVALUATION")
    print("Purpose: contextualize whether the ML model adds value")
    print("beyond simple heuristic rules")
    print("=" * 60)

    # Load val featured data
    val_df = pd.read_parquet(SPLITS_DIR / "val_featured.parquet")

    print(f"\n  Val rows: {len(val_df):,}")
    print(f"  Positive rate: {val_df['label'].mean():.4f}")
    print(f"  Fatal windows: {(val_df['has_fatal'] == 1).sum()}")

    # Flag rates to test (including our model's actual flag rates)
    flag_rates = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]

    # Run Top-K by crash rate
    print(f"\n{'='*60}")
    print("BASELINE 1: Top-K by Historical Crash Rate")
    print(f"{'='*60}")
    print(f"  {'Flag%':>7s} {'Recall':>7s} {'Fatal_R':>7s} {'Inj_R':>7s} {'Prec':>7s} {'F1':>7s}")
    print("  " + "-" * 50)

    topk_results = run_top_k_baseline(val_df, flag_rates)
    for r in topk_results:
        print(f"  {r['flag_rate']:7.3f} {r['recall_combined']:7.4f} "
              f"{r['recall_fatal']:7.4f} {r['recall_injury']:7.4f} "
              f"{r['precision']:7.4f} {r['f1']:7.4f}")

    # Run Top-K by fatal rate
    print(f"\n{'='*60}")
    print("BASELINE 2: Top-K by Historical Fatal Rate")
    print(f"{'='*60}")
    print(f"  {'Flag%':>7s} {'Recall':>7s} {'Fatal_R':>7s} {'Inj_R':>7s} {'Prec':>7s} {'F1':>7s}")
    print("  " + "-" * 50)

    topk_fatal_results = run_top_k_fatal_baseline(val_df, flag_rates)
    for r in topk_fatal_results:
        print(f"  {r['flag_rate']:7.3f} {r['recall_combined']:7.4f} "
              f"{r['recall_fatal']:7.4f} {r['recall_injury']:7.4f} "
              f"{r['precision']:7.4f} {r['f1']:7.4f}")

    # Run random baselines
    print(f"\n{'='*60}")
    print("BASELINE 3: Random Flagging")
    print(f"{'='*60}")

    for fr in [0.15, 0.30, 0.50]:
        rand = run_random_baseline(val_df, fr)
        print(f"  {rand['name']}: recall={rand['recall_combined']}, "
              f"fatal_recall={rand['recall_fatal']}, precision={rand['precision']}")

    # Comparison table
    print(f"\n{'='*60}")
    print("COMPARISON: ML MODEL vs NAIVE BASELINES")
    print("At comparable flag rates")
    print(f"{'='*60}")

    print(f"\n  iter_001 LogReg (flag_rate=0.143):")
    print(f"    Combined recall: 0.6330 | Fatal recall: 0.6598")

    # Find Top-K at ~15% flag rate
    topk_15 = [r for r in topk_results if abs(r["flag_rate"] - 0.15) < 0.05]
    if topk_15:
        r = topk_15[0]
        print(f"  Top-K crash rate (flag_rate={r['flag_rate']:.3f}):")
        print(f"    Combined recall: {r['recall_combined']} | Fatal recall: {r['recall_fatal']}")

    print(f"\n  iter_004 RF (flag_rate=0.474):")
    print(f"    Combined recall: 0.6909 | Fatal recall: 0.5760")

    topk_50 = [r for r in topk_results if abs(r["flag_rate"] - 0.50) < 0.05]
    if topk_50:
        r = topk_50[0]
        print(f"  Top-K crash rate (flag_rate={r['flag_rate']:.3f}):")
        print(f"    Combined recall: {r['recall_combined']} | Fatal recall: {r['recall_fatal']}")

    # Save results
    output = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "purpose": "Naive baseline comparison to contextualize ML model value",
        "val_rows": len(val_df),
        "val_positive_rate": round(float(val_df["label"].mean()), 4),
        "topk_crash_rate": topk_results,
        "topk_fatal_rate": topk_fatal_results,
    }

    output_path = EXPERIMENTS_DIR / "naive_baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()