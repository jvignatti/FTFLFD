"""
Iteration 003 — Add segment_impairment_ratio feature.

Steps:
  1. Compute feature from training crashes only
  2. Run 3-gate test
  3. If passed: train model with 9 features, evaluate, save artifacts
"""

import pandas as pd
import numpy as np
import json
import yaml
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

CURRENT_FEATURES = [
    "segment_crash_rate",
    "segment_fatal_rate",
    "segment_injury_rate",
    "segment_pdo_rate",
    "month",
    "is_winter",
    "is_weekend_heavy",
    "segment_type_is_grid",
]

NEW_FEATURE = "segment_impairment_ratio"
ITER_ID = "iter_003"


def compute_impairment_ratio(train_crashes):
    """Compute per-segment impairment ratio from training data only."""
    df = train_crashes[train_crashes["segment_id"].notna()].copy()

    # Define impairment as any non-None value
    df["is_impaired"] = df["Impairment"].isin(["Alcohol", "Drugs", "Alcohol and Drugs"]).astype(int)

    segment_counts = df.groupby("segment_id").agg(
        total=("OBJECTID", "count"),
        impaired=("is_impaired", "sum"),
    ).reset_index()

    segment_counts[NEW_FEATURE] = segment_counts["impaired"] / segment_counts["total"]
    return segment_counts[["segment_id", NEW_FEATURE]]


def add_feature_to_split(windows_df, impairment_rates):
    """Join impairment ratio to window observations."""
    df = windows_df.merge(impairment_rates, on="segment_id", how="left")
    df[NEW_FEATURE] = df[NEW_FEATURE].fillna(0.0)
    return df


def gate_1(feature_values):
    """Availability >= 95%."""
    coverage = feature_values.notna().mean()
    passed = coverage >= 0.95
    print(f"\n  Gate 1: Availability")
    print(f"    Coverage: {coverage:.4f} (threshold: 0.95)")
    print(f"    Result: {'PASS' if passed else 'FAIL'}")
    return passed


def gate_2(feature_values, target, sample_size=100000):
    """Mutual info > 0.01 OR Spearman > 0.05."""
    idx = np.random.RandomState(42).choice(len(feature_values), min(sample_size, len(feature_values)), replace=False)
    feat_sample = feature_values.iloc[idx].fillna(0)
    target_sample = target.iloc[idx]

    mi = mutual_info_classif(feat_sample.values.reshape(-1, 1), target_sample, random_state=42)[0]
    rho = abs(spearmanr(feat_sample, target_sample).correlation)

    passed = (mi > 0.01) or (rho > 0.05)
    print(f"\n  Gate 2: Signal")
    print(f"    Mutual info: {mi:.6f} (threshold: 0.01)")
    print(f"    Spearman:    {rho:.6f} (threshold: 0.05)")
    print(f"    Result: {'PASS' if passed else 'FAIL'}")
    return passed


def gate_3(X_train, y_train, X_val, y_val):
    """Adding feature must improve val recall by >= 0.005."""
    # Baseline (current features)
    scaler_base = StandardScaler()
    model_base = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
    X_tr_base = scaler_base.fit_transform(X_train[CURRENT_FEATURES])
    model_base.fit(X_tr_base, y_train)
    X_va_base = scaler_base.transform(X_val[CURRENT_FEATURES])
    recall_base = recall_score(y_val, model_base.predict(X_va_base), zero_division=0)

    # With new feature
    new_features = CURRENT_FEATURES + [NEW_FEATURE]
    scaler_new = StandardScaler()
    model_new = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
    X_tr_new = scaler_new.fit_transform(X_train[new_features])
    model_new.fit(X_tr_new, y_train)
    X_va_new = scaler_new.transform(X_val[new_features])
    recall_new = recall_score(y_val, model_new.predict(X_va_new), zero_division=0)

    gain = recall_new - recall_base
    passed = gain >= 0.005

    print(f"\n  Gate 3: Incremental Gain")
    print(f"    Recall without: {recall_base:.4f}")
    print(f"    Recall with:    {recall_new:.4f}")
    print(f"    Gain:           {gain:.4f} (threshold: 0.005)")
    print(f"    Result: {'PASS' if passed else 'FAIL'}")
    return passed


def evaluate_model(model, scaler, X, y, featured_df, split_name, feature_columns):
    """Full evaluation with fatal/injury/combined recall."""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    combined_recall = recall_score(y, y_pred, zero_division=0)
    combined_precision = precision_score(y, y_pred, zero_division=0)
    combined_f1 = f1_score(y, y_pred, zero_division=0)
    flag_rate = y_pred.mean()

    fatal_mask = featured_df["has_fatal"] == 1
    injury_only_mask = (featured_df["has_injury"] == 1) & (featured_df["has_fatal"] == 0)

    fatal_recall = y_pred[fatal_mask.values].mean() if fatal_mask.sum() > 0 else None
    injury_recall = y_pred[injury_only_mask.values].mean() if injury_only_mask.sum() > 0 else None

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
    print(f"ITERATION 003 — {NEW_FEATURE}")
    print("=" * 60)

    # Step 1: Compute feature from training crashes
    print("\nStep 1: Computing impairment ratio from training crashes...")
    train_crashes = pd.read_parquet(SPLITS_DIR / "train.parquet")
    impairment_rates = compute_impairment_ratio(train_crashes)
    print(f"  Segments with any impairment: {(impairment_rates[NEW_FEATURE] > 0).sum():,}")
    print(f"  Segments total: {len(impairment_rates):,}")
    print(f"  Mean ratio: {impairment_rates[NEW_FEATURE].mean():.4f}")
    print(f"  Max ratio:  {impairment_rates[NEW_FEATURE].max():.4f}")

    # Step 2: Add feature to train and val
    print("\nStep 2: Adding feature to featured splits...")
    train_df = pd.read_parquet(SPLITS_DIR / "train_featured.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "val_featured.parquet")

    train_df = add_feature_to_split(train_df, impairment_rates)
    val_df = add_feature_to_split(val_df, impairment_rates)

    # Step 3: Run 3-gate test
    print("\n" + "=" * 60)
    print("3-GATE TEST")
    print("=" * 60)

    g1 = gate_1(train_df[NEW_FEATURE])
    if not g1:
        print("\nREJECTED at Gate 1.")
        return

    g2 = gate_2(train_df[NEW_FEATURE], train_df["label"])
    if not g2:
        print("\nREJECTED at Gate 2.")
        return

    g3 = gate_3(train_df, train_df["label"], val_df, val_df["label"])
    if not g3:
        print("\nREJECTED at Gate 3.")
        return

    print("\n  ALL GATES PASSED. Proceeding to full training.")

    # Step 4: Train with 9 features
    feature_columns = CURRENT_FEATURES + [NEW_FEATURE]
    print("\n" + "=" * 60)
    print(f"TRAINING WITH {len(feature_columns)} FEATURES")
    print("=" * 60)

    X_train = train_df[feature_columns]
    y_train = train_df["label"]
    X_val = val_df[feature_columns]
    y_val = val_df["label"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        random_state=42,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
    )
    model.fit(X_train_scaled, y_train)

    # Step 5: Evaluate
    train_metrics = evaluate_model(model, scaler, X_train, y_train, train_df, "train", feature_columns)
    val_metrics = evaluate_model(model, scaler, X_val, y_val, val_df, "val", feature_columns)

    gen_gap = train_metrics["recall_combined"] - val_metrics["recall_combined"]

    print(f"\n  --- TRAIN ---")
    print(f"  Recall (combined):  {train_metrics['recall_combined']}")
    print(f"  Recall (fatal):     {train_metrics['recall_fatal']}")
    print(f"  Recall (injury):    {train_metrics['recall_injury']}")
    print(f"  Precision:          {train_metrics['precision']}")
    print(f"  Flag rate:          {train_metrics['flag_rate']}")

    print(f"\n  --- VAL ---")
    print(f"  Recall (combined):  {val_metrics['recall_combined']}")
    print(f"  Recall (fatal):     {val_metrics['recall_fatal']}")
    print(f"  Recall (injury):    {val_metrics['recall_injury']}")
    print(f"  Precision:          {val_metrics['precision']}")
    print(f"  Flag rate:          {val_metrics['flag_rate']}")

    print(f"\n  Generalization gap: {gen_gap:.4f}")

    # Compare to iter_001
    print(f"\n  --- COMPARISON TO ITER_001 ---")
    print(f"  Fatal recall:    0.5080 -> {val_metrics['recall_fatal']}")
    print(f"  Combined recall: 0.5688 -> {val_metrics['recall_combined']}")
    print(f"  Precision:       0.0521 -> {val_metrics['precision']}")
    print(f"  Flag rate:       0.3172 -> {val_metrics['flag_rate']}")

    # Feature importance
    coef_importance = dict(zip(
        feature_columns,
        [round(float(c), 4) for c in np.abs(model.coef_[0])]
    ))
    print(f"\n  Feature importance (|coefficient|):")
    for feat, imp in sorted(coef_importance.items(), key=lambda x: -x[1]):
        print(f"    {feat:30s} {imp:.4f}")

    # Step 6: Save artifacts
    iter_dir = EXPERIMENTS_DIR / ITER_ID
    iter_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {
        "_schema_version": "1.0",
        "iter_id": ITER_ID,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "train": train_metrics,
        "val": val_metrics,
        "benchmark": {
            "set": None,
            "audit_only": True,
            "note": "Benchmark not run at iteration 003"
        },
        "diagnostics": {
            "generalization_gap": round(gen_gap, 4),
            "overfitting_warning": bool(gen_gap > 0.10),
            "fatal_recall_floor_check": bool(
                val_metrics["recall_fatal"] >= 0.50
            ) if val_metrics["recall_fatal"] is not None else None,
        },
    }
    with open(iter_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    config_snapshot = {
        "iter_id": ITER_ID,
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
        "feature_set_version": "1.2",
        "active_features": feature_columns,
        "new_feature": NEW_FEATURE,
        "threshold": 0.5,
        "splits_config_version": "1.1",
        "thresholds_config_version": "1.0",
        "predecessor": "iter_001",
    }
    with open(iter_dir / "config_snapshot.yaml", "w") as f:
        yaml.dump(config_snapshot, f, default_flow_style=False)

    joblib.dump(model, iter_dir / "model.pkl")
    joblib.dump(scaler, iter_dir / "scaler.pkl")

    pd.DataFrame(
        list(coef_importance.items()),
        columns=["feature", "abs_coefficient"]
    ).sort_values("abs_coefficient", ascending=False).to_csv(
        iter_dir / "feature_importance.csv", index=False
    )

    print(f"\n  Artifacts saved to {iter_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()