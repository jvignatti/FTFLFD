"""
baseline_features.py — Computes features for the baseline model (Iteration 001).

Features are intentionally simple. The baseline validates the pipeline,
not the model. If this does not beat random chance, the problem is in
the data or framing, not in the model.

All features are computed from TRAINING DATA ONLY and applied forward
to val/benchmark sets. No leakage by construction.

Feature list (8 features):
  1. segment_crash_rate     — historical crashes per year in this segment
  2. segment_fatal_rate     — historical fatal crashes per year in this segment
  3. segment_injury_rate    — historical injury crashes per year in this segment
  4. segment_pdo_rate       — historical PDO crashes per year in this segment
  5. month                  — month of year from window_start (1-12)
  6. is_winter              — November through March (Vermont winter)
  7. is_weekend_heavy       — June through October (tourism/weekend traffic)
  8. segment_type_is_grid   — 1 if grid cell, 0 if LRS segment
"""

import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def compute_segment_rates(train_crashes: pd.DataFrame) -> pd.DataFrame:
    """Compute per-segment historical rates from training crash data.

    These rates are computed ONCE from training data and applied to all splits.
    This is the canonical method to prevent leakage — rates never see
    validation or benchmark data.

    Args:
        train_crashes: Crash-level training DataFrame with segment_id,
                       severity_class, event_date

    Returns:
        DataFrame with one row per segment_id and rate columns.
    """
    df = train_crashes[train_crashes["segment_id"].notna()].copy()

    if len(df) == 0:
        raise ValueError("No rows with valid segment_id in training data")

    # Compute years spanned by training data
    min_date = df["event_date"].min()
    max_date = df["event_date"].max()
    years_spanned = max((max_date - min_date).days / 365.25, 1.0)

    print(f"  Training period: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"  Years spanned: {years_spanned:.2f}")

    # Count by segment and severity
    counts = df.groupby("segment_id").agg(
        total_crashes=("OBJECTID", "count"),
        fatal_crashes=("severity_class", lambda x: (x == "fatal").sum()),
        injury_crashes=("severity_class", lambda x: (x == "injury").sum()),
        pdo_crashes=("severity_class", lambda x: (x == "pdo").sum()),
        segment_type=("segment_type", "first"),
    ).reset_index()

    # Compute annual rates
    counts["segment_crash_rate"] = counts["total_crashes"] / years_spanned
    counts["segment_fatal_rate"] = counts["fatal_crashes"] / years_spanned
    counts["segment_injury_rate"] = counts["injury_crashes"] / years_spanned
    counts["segment_pdo_rate"] = counts["pdo_crashes"] / years_spanned

    # Keep only rate columns and segment_type
    rates = counts[[
        "segment_id", "segment_type",
        "segment_crash_rate", "segment_fatal_rate",
        "segment_injury_rate", "segment_pdo_rate"
    ]]

    print(f"  Computed rates for {len(rates):,} segments")
    return rates


def add_calendar_features(windows: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived from window_start date."""
    df = windows.copy()
    df["month"] = df["window_start"].dt.month
    df["is_winter"] = df["month"].isin([11, 12, 1, 2, 3]).astype(int)
    df["is_weekend_heavy"] = df["month"].isin([6, 7, 8, 9, 10]).astype(int)
    return df


def build_baseline_features(
    windows: pd.DataFrame,
    segment_rates: pd.DataFrame
) -> pd.DataFrame:
    """Join segment rates to window observations and add calendar features.

    Args:
        windows: Window-level observations (from window_generator)
        segment_rates: Per-segment rates (from compute_segment_rates)

    Returns:
        DataFrame ready for model training with feature columns and label.
    """
    df = windows.copy()

    # Join segment rates
    df = df.merge(segment_rates, on="segment_id", how="left", suffixes=("", "_rate"))

    # Handle segments that exist in val/bench but not in training
    # (new segments — no historical rate available)
    rate_cols = [
        "segment_crash_rate", "segment_fatal_rate",
        "segment_injury_rate", "segment_pdo_rate"
    ]
    new_segments = df[rate_cols[0]].isna().sum()
    if new_segments > 0:
        print(f"  {new_segments:,} observations from segments not seen in training (rates set to 0)")
        for col in rate_cols:
            df[col] = df[col].fillna(0.0)

    # Add calendar features
    df = add_calendar_features(df)

    # Binary encode segment type
    # Use segment_type from rates if available, fall back to windows
    if "segment_type_rate" in df.columns:
        df["segment_type"] = df["segment_type_rate"].fillna(df["segment_type"])
        df = df.drop(columns=["segment_type_rate"])
    df["segment_type_is_grid"] = (df["segment_type"] == "grid").astype(int)

    return df


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


def prepare_xy(df: pd.DataFrame):
    """Extract feature matrix X and label vector y."""
    X = df[FEATURE_COLUMNS].copy()
    y = df["label"].copy()

    # Final safety check — no NaN in features
    nan_counts = X.isna().sum()
    if nan_counts.any():
        print(f"  WARNING: NaN in features:\n{nan_counts[nan_counts > 0]}")
        X = X.fillna(0.0)

    return X, y


if __name__ == "__main__":
    # Load training crashes for rate computation
    train_crashes = pd.read_parquet(SPLITS_DIR / "train.parquet")

    print("Computing segment rates from training data...")
    segment_rates = compute_segment_rates(train_crashes)

    # Load window observations
    for split_name in ["train", "val", "b1", "b2", "b3", "b4"]:
        windows_path = SPLITS_DIR / f"{split_name}_windows.parquet"
        if not windows_path.exists():
            print(f"  SKIP: {windows_path.name} not found")
            continue

        print(f"\nBuilding features for {split_name}...")
        windows = pd.read_parquet(windows_path)
        featured = build_baseline_features(windows, segment_rates)
        X, y = prepare_xy(featured)

        print(f"  Shape: X={X.shape}, y={y.shape}")
        print(f"  Positive rate: {y.mean():.4f}")
        print(f"  Features: {list(X.columns)}")

        # Save featured dataset
        output_path = SPLITS_DIR / f"{split_name}_featured.parquet"
        featured.to_parquet(output_path, index=False)
        print(f"  Saved: {output_path.name}")

    print("\nDone.")