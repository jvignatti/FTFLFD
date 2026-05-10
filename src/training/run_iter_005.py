"""
Iteration 005 — RF hyperparameter tuning: reduce max_depth from 10 to 6.

One change only: max_depth=10 -> max_depth=6
All other hyperparameters identical to iter_004.
Same 8 features.

Target: reduce flag rate below 0.30, maintain combined recall > 0.60,
        improve fatal recall toward iter_001 LogReg level (0.660).
"""

import pandas as pd
import numpy as np
import json
import yaml
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
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

ITER_ID = "iter_005"


def evaluate_model(model, X, y, featured_df, split_name):
    y_pred = model.predict(X)
    combined_recall = recall_score(y, y_pred, zero_division=0)
    combined_precision = precision_score(y, y_pred, zero_division=0)
    combined_f1 = f1_score(y, y_pred, zero_division=0)
    flag_rate = y_pred.mean()

    fatal_mask = featured_df["has_fatal"] == 1
    injury_only_mask = (featured_df["has_injury"] == 1) & (featured_df["has_fatal"] == 0)
    fatal_recall = float(y_pred[fatal_mask.values].mean()) if fatal_mask.sum() > 0 else None
    injury_recall = float(y_pred[injury_only_mask.values].mean()) if injury_only_mask.sum() > 0 else None

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    return {
        "split": split_name,
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
        "support_positive": int(y.sum()),
        "support_total": int(len(y)),
        "fatal_windows_total": int(fatal_mask.sum()),
    }


def main():
    print("=" * 60)
    print("ITERATION 005 — RF DEPTH TUNING")
    print("Change: max_depth=10 -> max_depth=6")
    print("All other parameters unchanged from iter_004")
    print("=" * 60)

    print("\nLoading featured splits...")
    train_df = pd.read_parquet(SPLITS_DIR / "train_featured.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "val_featured.parquet")

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["label"]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df["label"]

    print(f"  Train: {X_train.shape[0]:,} rows")
    print(f"  Val:   {X_val.shape[0]:,} rows")

    print("\nTraining RandomForestClassifier (max_depth=6)...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=50,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("  Training complete.")

    train_metrics = evaluate_model(model, X_train, y_train, train_df, "train")
    val_metrics = evaluate_model(model, X_val, y_val, val_df, "val")

    gen_gap = train_metrics["recall_combined"] - val_metrics["recall_combined"]

    print(f"\n  --- TRAIN ---")
    print(f"  Recall (combined):  {train_metrics['recall_combined']}")
    print(f"  Recall (fatal):     {train_metrics['recall_fatal']}")
    print(f"  Recall (injury):    {train_metrics['recall_injury']}")
    print(f"  Precision:          {train_metrics['precision']}")
    print(f"  F1:                 {train_metrics['f1']}")
    print(f"  Flag rate:          {train_metrics['flag_rate']}")

    print(f"\n  --- VAL ---")
    print(f"  Recall (combined):  {val_metrics['recall_combined']}")
    print(f"  Recall (fatal):     {val_metrics['recall_fatal']}")
    print(f"  Recall (injury):    {val_metrics['recall_injury']}")
    print(f"  Precision:          {val_metrics['precision']}")
    print(f"  F1:                 {val_metrics['f1']}")
    print(f"  Flag rate:          {val_metrics['flag_rate']}")

    print(f"\n  Generalization gap: {gen_gap:.4f}")
    if gen_gap > 0.10:
        print("  WARNING: Generalization gap exceeds 0.10")

    if val_metrics["recall_fatal"] is not None and val_metrics["recall_fatal"] < 0.50:
        print(f"  CRITICAL: Fatal recall below 0.50 floor")

    print(f"\n  --- COMPARISON ---")
    print(f"  {'Metric':<25s} {'iter_001 LR':>12s} {'iter_004 RF10':>12s} {'iter_005 RF6':>12s}")
    print(f"  {'-'*61}")
    print(f"  {'Fatal recall':<25s} {'0.5080':>12s} {'0.5760':>12s} {str(val_metrics['recall_fatal']):>12s}")
    print(f"  {'Combined recall':<25s} {'0.5688':>12s} {'0.6909':>12s} {str(val_metrics['recall_combined']):>12s}")
    print(f"  {'Precision':<25s} {'0.0521':>12s} {'0.0424':>12s} {str(val_metrics['precision']):>12s}")
    print(f"  {'Flag rate':<25s} {'0.1430':>12s} {'0.4743':>12s} {str(val_metrics['flag_rate']):>12s}")
    print(f"  {'F1':<25s} {'0.1112':>12s} {'0.0798':>12s} {str(val_metrics['f1']):>12s}")
    print(f"  {'Gen gap':<25s} {'0.0642':>12s} {'0.1114':>12s} {str(round(gen_gap, 4)):>12s}")

    # Feature importance
    importance = dict(zip(
        FEATURE_COLUMNS,
        [round(float(i), 4) for i in model.feature_importances_]
    ))
    print(f"\n  Feature importance (Gini):")
    for feat, imp in sorted(importance.items(), key=lambda x: -x[1]):
        print(f"    {feat:30s} {imp:.4f}")

    # Save
    iter_dir = EXPERIMENTS_DIR / ITER_ID
    iter_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {
        "_schema_version": "1.0",
        "iter_id": ITER_ID,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "train": train_metrics,
        "val": val_metrics,
        "benchmark": {"set": None, "audit_only": True, "note": "B1 already run at iter_004"},
        "diagnostics": {
            "generalization_gap": round(gen_gap, 4),
            "overfitting_warning": bool(gen_gap > 0.10),
            "fatal_recall_floor_check": bool(val_metrics["recall_fatal"] >= 0.50) if val_metrics["recall_fatal"] is not None else None,
        },
    }
    with open(iter_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    config_snapshot = {
        "iter_id": ITER_ID,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model_class": "RandomForestClassifier",
        "change": "max_depth reduced from 10 to 6",
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 6,
            "min_samples_leaf": 50,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        },
        "random_seed": 42,
        "feature_set_version": "1.1",
        "active_features": FEATURE_COLUMNS,
        "threshold": 0.5,
        "splits_config_version": "1.1",
        "thresholds_config_version": "1.0",
        "predecessor": "iter_004",
    }
    with open(iter_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(config_snapshot, f, default_flow_style=False)

    joblib.dump(model, iter_dir / "model.pkl")

    pd.DataFrame(
        list(importance.items()),
        columns=["feature", "gini_importance"]
    ).sort_values("gini_importance", ascending=False).to_csv(
        iter_dir / "feature_importance.csv", index=False
    )

    print(f"\n  Artifacts saved to {iter_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()