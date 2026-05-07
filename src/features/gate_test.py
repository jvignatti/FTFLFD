"""
gate_test.py — Runs the 3-gate acceptance test for a candidate feature.

Gate 1: Availability — feature present in >= 95% of training rows
Gate 2: Signal — mutual info > 0.01 OR Spearman correlation > 0.05
Gate 3: Incremental gain — val recall improves by >= 0.005

Per CLAUDE.md: no feature enters the pipeline without passing all three gates.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def gate_1_availability(feature_values: pd.Series, threshold: float = 0.95) -> dict:
    """Gate 1: Feature must be present in >= 95% of training rows."""
    available = feature_values.notna().mean()
    passed = available >= threshold
    return {
        "gate": "Gate 1: Availability",
        "metric": "coverage",
        "value": round(available, 4),
        "threshold": threshold,
        "passed": passed,
    }


def gate_2_signal(
    feature_values: pd.Series,
    target: pd.Series,
    mi_threshold: float = 0.01,
    spearman_threshold: float = 0.05,
) -> dict:
    """Gate 2: Mutual info > 0.01 OR Spearman > 0.05."""
    clean = feature_values.fillna(0)
    mi = mutual_info_classif(
        clean.values.reshape(-1, 1), target, random_state=42, n_neighbors=5
    )[0]
    rho, _ = spearmanr(clean, target)
    rho = abs(rho)

    passed = (mi > mi_threshold) or (rho > spearman_threshold)
    return {
        "gate": "Gate 2: Signal",
        "mutual_info": round(float(mi), 6),
        "mi_threshold": mi_threshold,
        "spearman": round(float(rho), 6),
        "spearman_threshold": spearman_threshold,
        "passed": passed,
    }


def gate_3_incremental_gain(
    current_features: list,
    new_feature_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    min_gain: float = 0.005,
) -> dict:
    """Gate 3: Adding feature must improve val recall by >= 0.005."""
    scaler_base = StandardScaler()
    model_base = LogisticRegression(
        random_state=42, class_weight="balanced", max_iter=1000
    )

    # Baseline (current features only)
    X_tr_base = scaler_base.fit_transform(X_train[current_features])
    model_base.fit(X_tr_base, y_train)
    X_va_base = scaler_base.transform(X_val[current_features])
    recall_base = recall_score(y_val, model_base.predict(X_va_base), zero_division=0)

    # With new feature
    new_features = current_features + [new_feature_name]
    scaler_new = StandardScaler()
    model_new = LogisticRegression(
        random_state=42, class_weight="balanced", max_iter=1000
    )
    X_tr_new = scaler_new.fit_transform(X_train[new_features])
    model_new.fit(X_tr_new, y_train)
    X_va_new = scaler_new.transform(X_val[new_features])
    recall_new = recall_score(y_val, model_new.predict(X_va_new), zero_division=0)

    gain = recall_new - recall_base
    passed = gain >= min_gain

    return {
        "gate": "Gate 3: Incremental Gain",
        "recall_baseline": round(recall_base, 4),
        "recall_with_feature": round(recall_new, 4),
        "gain": round(gain, 4),
        "min_gain": min_gain,
        "passed": passed,
    }


def run_gate_test(
    feature_name: str,
    current_features: list,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> bool:
    """Run all 3 gates for a candidate feature. Returns True if all pass."""
    print(f"\n{'='*60}")
    print(f"3-GATE TEST: {feature_name}")
    print(f"{'='*60}")

    # Gate 1
    g1 = gate_1_availability(train_df[feature_name])
    status = "PASS" if g1["passed"] else "FAIL"
    print(f"\n  {g1['gate']}: {status}")
    print(f"    Coverage: {g1['value']} (threshold: {g1['threshold']})")

    if not g1["passed"]:
        print(f"\n  REJECTED at Gate 1. Feature not added.")
        return False

    # Gate 2 (sample for speed — mutual info on 6M rows is slow)
    sample_size = min(100_000, len(train_df))
    sample = train_df.sample(n=sample_size, random_state=42)
    g2 = gate_2_signal(sample[feature_name], sample["label"])
    status = "PASS" if g2["passed"] else "FAIL"
    print(f"\n  {g2['gate']}: {status}")
    print(f"    Mutual info: {g2['mutual_info']} (threshold: {g2['mi_threshold']})")
    print(f"    Spearman:    {g2['spearman']} (threshold: {g2['spearman_threshold']})")

    if not g2["passed"]:
        print(f"\n  REJECTED at Gate 2. Feature not added.")
        return False

    # Gate 3
    print(f"\n  Running Gate 3 (training two models — this may take a minute)...")
    g3 = gate_3_incremental_gain(
        current_features, feature_name,
        train_df, train_df["label"],
        val_df, val_df["label"],
    )
    status = "PASS" if g3["passed"] else "FAIL"
    print(f"\n  {g3['gate']}: {status}")
    print(f"    Recall without: {g3['recall_baseline']}")
    print(f"    Recall with:    {g3['recall_with_feature']}")
    print(f"    Gain:           {g3['gain']} (threshold: {g3['min_gain']})")

    if not g3["passed"]:
        print(f"\n  REJECTED at Gate 3. Feature not added.")
        return False

    print(f"\n  ALL GATES PASSED. Feature '{feature_name}' is accepted.")
    return True


if __name__ == "__main__":
    from src.features.baseline_features import FEATURE_COLUMNS

    # Load training crashes to compute the new feature
    print("Loading training crashes...")
    train_crashes = pd.read_parquet(SPLITS_DIR / "train.parquet")

    # Compute segment_fatal_ever from training data only
    fatal_segments = set(
        train_crashes[
            train_crashes["severity_class"] == "fatal"
        ]["segment_id"].dropna().unique()
    )
    print(f"  Segments with at least one fatal in training: {len(fatal_segments):,}")

    # Load featured splits
    print("\nLoading featured splits...")
    train_df = pd.read_parquet(SPLITS_DIR / "train_featured.parquet")
    val_df = pd.read_parquet(SPLITS_DIR / "val_featured.parquet")

    # Add candidate feature to both splits
    train_df["segment_fatal_ever"] = train_df["segment_id"].isin(fatal_segments).astype(int)
    val_df["segment_fatal_ever"] = val_df["segment_id"].isin(fatal_segments).astype(int)

    print(f"  Train: {train_df['segment_fatal_ever'].mean():.4f} of segments have fatal history")
    print(f"  Val:   {val_df['segment_fatal_ever'].mean():.4f} of segments have fatal history")

    # Run gate test
    passed = run_gate_test(
        feature_name="segment_fatal_ever",
        current_features=FEATURE_COLUMNS,
        train_df=train_df,
        val_df=val_df,
    )

    if passed:
        print("\nProceed to iteration 002.")
    else:
        print("\nDo NOT proceed. Choose a different feature candidate.")