"""
threshold_search.py — Finds optimal classification threshold for iteration 002.

Per CLAUDE.md:
  - Maximize fatal recall
  - Subject to: precision >= 0.05, flag rate <= 0.30
  - Threshold change counts as one iteration change
  - Must generate precision-recall curve

Uses the trained model from iter_001 (no retraining).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    confusion_matrix, precision_recall_curve
)
import joblib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

# Constraints from CLAUDE.md
PRECISION_FLOOR = 0.05
MAX_FLAG_RATE = 0.30


def search_threshold():
    """Search for optimal threshold using iter_001 model on validation data."""
    iter_id = "iter_002"
    iter_dir = EXPERIMENTS_DIR / iter_id
    iter_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ITERATION 002 — THRESHOLD OPTIMIZATION")
    print("Model: LogisticRegression (from iter_001, no retraining)")
    print("Change: threshold adjustment only")
    print("=" * 60)

    # Load iter_001 model and scaler
    prev_dir = EXPERIMENTS_DIR / "iter_001"
    model = joblib.load(prev_dir / "model.pkl")
    scaler = joblib.load(prev_dir / "scaler.pkl")

    # Feature columns (same as iter_001)
    feature_columns = [
        "segment_crash_rate",
        "segment_fatal_rate",
        "segment_injury_rate",
        "segment_pdo_rate",
        "month",
        "is_winter",
        "is_weekend_heavy",
        "segment_type_is_grid",
    ]

    # Load featured splits
    print("\nLoading featured splits...")
    train_df = pd.read_parquet(SPLITS_DIR / "train_featured.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "val_featured.parquet")

    X_train = train_df[feature_columns]
    y_train = train_df["label"]
    X_val = val_df[feature_columns]
    y_val = val_df["label"]

    # Get probabilities (no retraining — using existing model)
    X_val_scaled = scaler.transform(X_val)
    y_proba = model.predict_proba(X_val_scaled)[:, 1]

    X_train_scaled = scaler.transform(X_train)
    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]

    # Sweep thresholds
    thresholds = np.arange(0.05, 0.95, 0.01)
    results = []

    print("\nSweeping thresholds...")
    print(f"  {'Thresh':>7s} {'Recall':>7s} {'Fatal_R':>7s} {'Inj_R':>7s} "
          f"{'Prec':>7s} {'F1':>7s} {'Flag%':>7s} {'Status':>10s}")
    print("  " + "-" * 65)

    fatal_mask = val_df["has_fatal"] == 1
    injury_only_mask = (val_df["has_injury"] == 1) & (val_df["has_fatal"] == 0)

    best_fatal_recall = 0
    best_threshold = 0.5
    best_row = None

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)

        recall_combined = recall_score(y_val, y_pred, zero_division=0)
        precision = precision_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        flag_rate = y_pred.mean()

        # Fatal recall
        if fatal_mask.sum() > 0:
            fatal_recall = y_pred[fatal_mask.values].mean()
        else:
            fatal_recall = 0.0

        # Injury recall
        if injury_only_mask.sum() > 0:
            injury_recall = y_pred[injury_only_mask.values].mean()
        else:
            injury_recall = 0.0

        # Check constraints
        meets_precision = precision >= PRECISION_FLOOR
        meets_flag_rate = flag_rate <= MAX_FLAG_RATE

        if meets_precision and meets_flag_rate:
            status = "FEASIBLE"
        else:
            reasons = []
            if not meets_precision:
                reasons.append("prec")
            if not meets_flag_rate:
                reasons.append("flag")
            status = "FAIL:" + "+".join(reasons)

        row = {
            "threshold": round(t, 2),
            "recall_combined": round(recall_combined, 4),
            "recall_fatal": round(fatal_recall, 4),
            "recall_injury": round(injury_recall, 4),
            "precision": round(precision, 4),
            "f1": round(f1, 4),
            "flag_rate": round(flag_rate, 4),
            "feasible": meets_precision and meets_flag_rate,
        }
        results.append(row)

        # Track best feasible threshold by fatal recall
        if meets_precision and meets_flag_rate and fatal_recall > best_fatal_recall:
            best_fatal_recall = fatal_recall
            best_threshold = round(t, 2)
            best_row = row

        # Print select thresholds (every 5th + any that are interesting)
        if int(round(t * 100)) % 5 == 0 or (meets_precision and meets_flag_rate and fatal_recall > 0.50):
            print(f"  {t:7.2f} {recall_combined:7.4f} {fatal_recall:7.4f} "
                  f"{injury_recall:7.4f} {precision:7.4f} {f1:7.4f} "
                  f"{flag_rate:7.4f} {status:>10s}")

    # Summary
    print(f"\n{'='*60}")
    print("THRESHOLD SEARCH RESULTS")
    print(f"{'='*60}")

    if best_row is not None:
        print(f"  Best feasible threshold: {best_threshold}")
        print(f"  Fatal recall:            {best_row['recall_fatal']}")
        print(f"  Combined recall:         {best_row['recall_combined']}")
        print(f"  Injury recall:           {best_row['recall_injury']}")
        print(f"  Precision:               {best_row['precision']}")
        print(f"  F1:                      {best_row['f1']}")
        print(f"  Flag rate:               {best_row['flag_rate']}")
    else:
        print("  NO FEASIBLE THRESHOLD FOUND")
        print("  All thresholds violate precision floor or flag rate maximum")

    # Compare to iter_001 (threshold 0.5)
    print(f"\n  Comparison to iter_001 (threshold=0.50):")
    iter001_row = [r for r in results if r["threshold"] == 0.50][0]
    print(f"    Fatal recall: {iter001_row['recall_fatal']} → {best_row['recall_fatal'] if best_row else 'N/A'}")
    print(f"    Combined recall: {iter001_row['recall_combined']} → {best_row['recall_combined'] if best_row else 'N/A'}")
    print(f"    Precision: {iter001_row['precision']} → {best_row['precision'] if best_row else 'N/A'}")
    print(f"    Flag rate: {iter001_row['flag_rate']} → {best_row['flag_rate'] if best_row else 'N/A'}")

    # --- EVALUATE ON TRAIN WITH BEST THRESHOLD ---
    if best_row is not None:
        print(f"\n{'='*60}")
        print(f"TRAIN METRICS AT THRESHOLD {best_threshold}")
        print(f"{'='*60}")

        y_train_pred = (y_train_proba >= best_threshold).astype(int)
        train_recall = recall_score(y_train, y_train_pred, zero_division=0)
        train_precision = precision_score(y_train, y_train_pred, zero_division=0)
        train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        train_flag_rate = y_train_pred.mean()

        train_fatal_mask = train_df["has_fatal"] == 1
        train_injury_mask = (train_df["has_injury"] == 1) & (train_df["has_fatal"] == 0)
        train_fatal_recall = y_train_pred[train_fatal_mask.values].mean() if train_fatal_mask.sum() > 0 else 0
        train_injury_recall = y_train_pred[train_injury_mask.values].mean() if train_injury_mask.sum() > 0 else 0

        tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()

        print(f"  Recall (combined):  {train_recall:.4f}")
        print(f"  Recall (fatal):     {train_fatal_recall:.4f}")
        print(f"  Recall (injury):    {train_injury_recall:.4f}")
        print(f"  Precision:          {train_precision:.4f}")
        print(f"  F1:                 {train_f1:.4f}")
        print(f"  Flag rate:          {train_flag_rate:.4f}")
        print(f"  TP: {tp:,}  FP: {fp:,}  FN: {fn:,}  TN: {tn:,}")

        gen_gap = train_recall - best_row["recall_combined"]
        print(f"\n  Generalization gap (recall): {gen_gap:.4f}")
        if gen_gap > 0.10:
            print("  WARNING: Generalization gap exceeds 0.10 threshold")

        # --- SAVE ARTIFACTS ---

        # Metrics
        val_tn, val_fp, val_fn, val_tp = confusion_matrix(
            y_val, (y_proba >= best_threshold).astype(int)
        ).ravel()

        all_metrics = {
            "_schema_version": "1.0",
            "iter_id": iter_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "train": {
                "split": "train",
                "recall_combined": round(train_recall, 4),
                "recall_fatal": round(float(train_fatal_recall), 4),
                "recall_injury": round(float(train_injury_recall), 4),
                "precision": round(train_precision, 4),
                "f1": round(train_f1, 4),
                "flag_rate": round(train_flag_rate, 4),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "true_negatives": int(tn),
                "false_negatives": int(fn),
                "support_positive": int(y_train.sum()),
                "support_total": int(len(y_train)),
                "fatal_windows_total": int(train_fatal_mask.sum()),
            },
            "val": {
                "split": "val",
                "recall_combined": round(float(best_row["recall_combined"]), 4),
                "recall_fatal": round(float(best_row["recall_fatal"]), 4),
                "recall_injury": round(float(best_row["recall_injury"]), 4),
                "precision": round(float(best_row["precision"]), 4),
                "f1": round(float(best_row["f1"]), 4),
                "flag_rate": round(float(best_row["flag_rate"]), 4),
                "true_positives": int(val_tp),
                "false_positives": int(val_fp),
                "true_negatives": int(val_tn),
                "false_negatives": int(val_fn),
                "support_positive": int(y_val.sum()),
                "support_total": int(len(y_val)),
                "fatal_windows_total": int(fatal_mask.sum()),
            },
            "benchmark": {
                "set": None,
                "audit_only": True,
                "note": "Benchmark not run at iteration 002 (per protocol: B1 at iteration 5)"
            },
            "diagnostics": {
                "generalization_gap": round(gen_gap, 4),
                "overfitting_warning": bool(gen_gap > 0.10),
                "fatal_recall_floor_check": bool(best_row["recall_fatal"] >= 0.50),
            },
        }
        with open(iter_dir / "metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=2)

        # Config snapshot
        config_snapshot = {
            "iter_id": iter_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "model_class": "LogisticRegression",
            "hyperparameters": {
                "C": 1.0,
                "max_iter": 1000,
                "solver": "lbfgs",
                "class_weight": "balanced",
                "random_state": 42,
            },
            "random_seed": 42,
            "feature_set_version": "1.1",
            "active_features": feature_columns,
            "threshold": best_threshold,
            "threshold_change_from": 0.50,
            "threshold_change_to": best_threshold,
            "splits_config_version": "1.1",
            "thresholds_config_version": "1.0",
            "predecessor": "iter_001",
        }
        with open(iter_dir / "config_snapshot.yaml", "w") as f:
            yaml.dump(config_snapshot, f, default_flow_style=False)

        # Save threshold sweep results
        sweep_df = pd.DataFrame(results)
        sweep_df.to_csv(iter_dir / "threshold_sweep.csv", index=False)

        print(f"\n  All artifacts saved to {iter_dir}")
        print(f"  Threshold sweep saved to {iter_dir / 'threshold_sweep.csv'}")

    print("=" * 60)

    return best_threshold, best_row, results


if __name__ == "__main__":
    best_threshold, best_row, results = search_threshold()