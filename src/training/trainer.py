"""
trainer.py — Trains and evaluates models for the FTFLFD pipeline.

Iteration 001: Logistic Regression baseline with 8 features.

Per CLAUDE.md:
  - class_weight="balanced" mandatory
  - random_state=42
  - Report recall separately for Fatal, Injury, Combined
  - Log all metrics to experiments/iter_NNN/
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
CONFIG_DIR = REPO_ROOT / "config"


def load_featured_split(split_name: str) -> pd.DataFrame:
    """Load a featured split from parquet."""
    path = SPLITS_DIR / f"{split_name}_featured.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Featured split not found: {path}")
    return pd.read_parquet(path)


def evaluate(
    model,
    scaler,
    X: pd.DataFrame,
    y: pd.Series,
    featured_df: pd.DataFrame,
    split_name: str,
    feature_columns: list,
) -> dict:
    """Evaluate model on a split. Reports combined, fatal, and injury recall."""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    # Combined metrics
    combined_recall = recall_score(y, y_pred, zero_division=0)
    combined_precision = precision_score(y, y_pred, zero_division=0)
    combined_f1 = f1_score(y, y_pred, zero_division=0)

    # Fatal-specific recall
    # A window is "fatal" if has_fatal == 1
    fatal_mask = featured_df["has_fatal"] == 1
    fatal_actual = fatal_mask.astype(int)
    fatal_predicted = pd.Series(y_pred, index=y.index)

    if fatal_mask.sum() > 0:
        # Among windows that actually had fatals, how many did we predict as positive?
        fatal_recall = fatal_predicted[fatal_mask].mean()
    else:
        fatal_recall = None

    # Injury-specific recall (has_injury but NOT fatal)
    injury_only_mask = (featured_df["has_injury"] == 1) & (featured_df["has_fatal"] == 0)
    if injury_only_mask.sum() > 0:
        injury_recall = fatal_predicted[injury_only_mask].mean()
    else:
        injury_recall = None

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    # Flag rate (% of observations predicted positive)
    flag_rate = y_pred.mean()

    metrics = {
        "split": split_name,
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
        "support_positive": int(y.sum()),
        "support_negative": int(len(y) - y.sum()),
        "support_total": int(len(y)),
        "fatal_windows_total": int(fatal_mask.sum()),
        "injury_only_windows_total": int(injury_only_mask.sum()),
    }

    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print(f"\n  --- {metrics['split'].upper()} ---")
    print(f"  Recall (combined):  {metrics['recall_combined']:.4f}")
    if metrics['recall_fatal'] is not None:
        print(f"  Recall (fatal):     {metrics['recall_fatal']:.4f}")
    if metrics['recall_injury'] is not None:
        print(f"  Recall (injury):    {metrics['recall_injury']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  F1:                 {metrics['f1']:.4f}")
    print(f"  Flag rate:          {metrics['flag_rate']:.4f}")
    print(f"  TP: {metrics['true_positives']:,}  FP: {metrics['false_positives']:,}  "
          f"FN: {metrics['false_negatives']:,}  TN: {metrics['true_negatives']:,}")
    if metrics['fatal_windows_total'] > 0:
        print(f"  Fatal windows: {metrics['fatal_windows_total']:,}")


def run_iteration_001():
    """Run the baseline iteration: Logistic Regression with 8 features."""
    iter_id = "iter_001"
    iter_dir = EXPERIMENTS_DIR / iter_id
    iter_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"ITERATION 001 — BASELINE")
    print(f"Model: LogisticRegression")
    print(f"Features: 8 (baseline set)")
    print(f"class_weight: balanced")
    print(f"random_state: 42")
    print("=" * 60)

    # Feature columns (from baseline_features.py)
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

    # Load data
    print("\nLoading featured splits...")
    train_df = load_featured_split("train")
    val_df = load_featured_split("val")

    X_train = train_df[feature_columns]
    y_train = train_df["label"]
    X_val = val_df[feature_columns]
    y_val = val_df["label"]

    print(f"  Train: {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]:,} rows, {X_val.shape[1]} features")

    # Scale features (fit on training only — leakage safe)
    print("\nScaling features (fit on training only)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train
    print("\nTraining LogisticRegression...")
    model = LogisticRegression(
        random_state=42,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
    )
    model.fit(X_train_scaled, y_train)
    print("  Training complete.")

    # Evaluate on train
    train_metrics = evaluate(
        model, scaler, X_train, y_train, train_df, "train", feature_columns
    )
    print_metrics(train_metrics)

    # Evaluate on val
    val_metrics = evaluate(
        model, scaler, X_val, y_val, val_df, "val", feature_columns
    )
    print_metrics(val_metrics)

    # Generalization gap
    gen_gap = train_metrics["recall_combined"] - val_metrics["recall_combined"]
    print(f"\n  Generalization gap (recall): {gen_gap:.4f}")
    if gen_gap > 0.10:
        print("  WARNING: Generalization gap exceeds 0.10 threshold")

    # Fatal recall floor check
    if val_metrics["recall_fatal"] is not None and val_metrics["recall_fatal"] < 0.50:
        print(f"  CRITICAL: Fatal recall ({val_metrics['recall_fatal']:.4f}) "
              f"below 0.50 floor")

    # Feature importance (coefficients for logistic regression)
    coef_importance = dict(zip(
        feature_columns,
        [round(float(c), 4) for c in np.abs(model.coef_[0])]
    ))
    print("\n  Feature importance (|coefficient|):")
    for feat, imp in sorted(coef_importance.items(), key=lambda x: -x[1]):
        print(f"    {feat:30s} {imp:.4f}")

    # --- SAVE EXPERIMENT ARTIFACTS ---

    # 1. Metrics
    all_metrics = {
        "_schema_version": "1.0",
        "_documentation": "Iteration 001 baseline. See CLAUDE.md for metric definitions.",
        "iter_id": iter_id,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "train": train_metrics,
        "val": val_metrics,
        "benchmark": {
            "set": None,
            "recall_combined": None,
            "recall_fatal": None,
            "recall_injury": None,
            "precision": None,
            "f1": None,
            "audit_only": True,
            "note": "Benchmark not run at iteration 001 (per protocol: B1 at iteration 5)"
        },
        "diagnostics": {
            "generalization_gap": round(gen_gap, 4),
            "benchmark_gap": None,
            "overfitting_warning": bool(gen_gap > 0.10),
            "fatal_recall_floor_check": bool(
                val_metrics["recall_fatal"] >= 0.50
            ) if val_metrics["recall_fatal"] is not None else None,
        },
    }
    with open(iter_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    # 2. Config snapshot
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
        "threshold": 0.5,
        "splits_config_version": "1.1",
        "thresholds_config_version": "1.0",
    }
    with open(iter_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(config_snapshot, f, default_flow_style=False)

    # 3. Model and scaler
    joblib.dump(model, iter_dir / "model.pkl")
    joblib.dump(scaler, iter_dir / "scaler.pkl")

    # 4. Feature importance
    pd.DataFrame(
        list(coef_importance.items()),
        columns=["feature", "abs_coefficient"]
    ).sort_values("abs_coefficient", ascending=False).to_csv(
        iter_dir / "feature_importance.csv", index=False
    )

    print(f"\n  All artifacts saved to {iter_dir}")
    print("=" * 60)

    return model, scaler, all_metrics


if __name__ == "__main__":
    model, scaler, metrics = run_iteration_001()