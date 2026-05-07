"""
splitter.py — Enforces time-based data splits with gap windows.

Reads: config/splits.yaml for split boundaries and gap rules
Input: ingested DataFrame with event_date column
Output: dict of split DataFrames (train, val, b1, b2, b3, b4)

All splits are chronological. No random shuffling.
Gap windows between splits are enforced and rows in gaps are dropped.
See docs/technical/01_project_design.md and config/splits.yaml v1.1.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import timedelta
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "splits.yaml"
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def load_split_config() -> dict:
    """Load split configuration from config/splits.yaml."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Split config not found: {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    return config


def compute_boundaries(config: dict) -> Dict[str, dict]:
    """Compute effective split boundaries with gap enforcement.

    Returns dict of split_name -> {start, end, eval_start, eval_end}
    where eval_start/eval_end are the actual usable date range after
    gap windows are applied.
    """
    gap_days = config["gap_days"]
    gap = timedelta(days=gap_days)
    splits = config["splits"]
    split_names = list(splits.keys())

    boundaries = {}
    for i, name in enumerate(split_names):
        raw_start = pd.Timestamp(splits[name]["start"])
        raw_end_value = splits[name]["end"]

        # Handle "auto" end date (for b4 — uses current date)
        if raw_end_value == "auto":
            raw_end = pd.Timestamp.now().normalize()
        else:
            raw_end = pd.Timestamp(raw_end_value)

        # Apply gap: push start forward if this is not the first split
        if i == 0:
            effective_start = raw_start
        else:
            prev_end_value = splits[split_names[i - 1]]["end"]
            if prev_end_value == "auto":
                prev_end = pd.Timestamp.now().normalize()
            else:
                prev_end = pd.Timestamp(prev_end_value)
            effective_start = prev_end + gap

        effective_end = raw_end

        boundaries[name] = {
            "raw_start": raw_start,
            "raw_end": raw_end,
            "effective_start": effective_start,
            "effective_end": effective_end,
            "evaluation": splits[name].get("evaluation", ""),
        }

    return boundaries


def split_data(
    df: pd.DataFrame,
    date_col: str = "event_date"
) -> Dict[str, pd.DataFrame]:
    """Split DataFrame into train/val/b1/b2/b3/b4 based on config.

    Enforces:
    - Chronological ordering
    - Gap windows between consecutive splits
    - No row appears in more than one split
    - Rows in gap windows are dropped entirely

    Returns dict of split_name -> DataFrame.
    """
    config = load_split_config()
    boundaries = compute_boundaries(config)

    if date_col not in df.columns:
        raise ValueError(f"Date column '{date_col}' not found in DataFrame")

    # Ensure datetime
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    splits = {}
    total_assigned = 0
    total_in_gaps = 0

    print("=" * 60)
    print("SPLIT ASSIGNMENT")
    print(f"Gap window: {config['gap_days']} days between each split")
    print("=" * 60)

    for name, bounds in boundaries.items():
        mask = (
            (df[date_col] >= bounds["effective_start"]) &
            (df[date_col] <= bounds["effective_end"])
        )
        split_df = df[mask].copy()
        splits[name] = split_df
        total_assigned += len(split_df)

        # Count fatals and injuries for reporting
        fatal_count = 0
        injury_count = 0
        if "severity_class" in split_df.columns:
            fatal_count = (split_df["severity_class"] == "fatal").sum()
            injury_count = (split_df["severity_class"] == "injury").sum()

        print(
            f"  {name:6s}: {bounds['effective_start'].strftime('%Y-%m-%d')} "
            f"to {bounds['effective_end'].strftime('%Y-%m-%d')} | "
            f"{len(split_df):>7,} rows | "
            f"Fatal: {fatal_count:>4} | Injury: {injury_count:>5} | "
            f"Eval: {bounds['evaluation']}"
        )

    # Count rows that fell in gaps or outside all boundaries
    unassigned = len(df) - total_assigned
    print(f"\n  Total assigned: {total_assigned:,}")
    print(f"  In gaps / outside boundaries: {unassigned:,}")
    print("=" * 60)

    return splits


def validate_splits(splits: Dict[str, pd.DataFrame], date_col: str = "event_date") -> bool:
    """Run leakage and integrity assertions on splits.

    Checks:
    1. No date overlap between any consecutive splits
    2. No OBJECTID appears in more than one split
    3. Each split is internally sorted by date
    4. Gap windows are respected (min gap between consecutive splits)

    Returns True if all checks pass, raises AssertionError otherwise.
    """
    config = load_split_config()
    gap_days = config["gap_days"]
    split_names = list(splits.keys())

    print("\nRunning split validation...")

    # Check 1: No date overlap between consecutive splits
    for i in range(len(split_names) - 1):
        name_a = split_names[i]
        name_b = split_names[i + 1]
        if len(splits[name_a]) == 0 or len(splits[name_b]) == 0:
            print(f"  SKIP: {name_a} or {name_b} is empty")
            continue

        max_date_a = splits[name_a][date_col].max()
        min_date_b = splits[name_b][date_col].min()

        assert max_date_a < min_date_b, (
            f"LEAK: {name_a} max date ({max_date_a}) >= "
            f"{name_b} min date ({min_date_b})"
        )

        actual_gap = (min_date_b - max_date_a).days
        assert actual_gap >= gap_days, (
            f"GAP VIOLATION: {name_a} to {name_b} gap is {actual_gap} days, "
            f"required {gap_days}"
        )

        print(f"  {name_a} → {name_b}: gap = {actual_gap} days (required: {gap_days}) ✓")

    # Check 2: No OBJECTID overlap between any splits
    if "OBJECTID" in splits[split_names[0]].columns:
        all_pairs = []
        for i in range(len(split_names)):
            for j in range(i + 1, len(split_names)):
                name_a = split_names[i]
                name_b = split_names[j]
                ids_a = set(splits[name_a]["OBJECTID"])
                ids_b = set(splits[name_b]["OBJECTID"])
                overlap = ids_a & ids_b
                assert len(overlap) == 0, (
                    f"ID LEAK: {len(overlap)} OBJECTIDs shared between "
                    f"{name_a} and {name_b}"
                )
        print(f"  No OBJECTID overlap across any split pair ✓")

    # Check 3: Each split is internally sorted
    for name in split_names:
        if len(splits[name]) > 0:
            dates = splits[name][date_col].values
            assert np.all(dates[:-1] <= dates[1:]), (
                f"SORT ERROR: {name} is not chronologically sorted"
            )
    print(f"  All splits chronologically sorted ✓")

    print("  ALL VALIDATION CHECKS PASSED ✓")
    return True


def save_splits(splits: Dict[str, pd.DataFrame]) -> None:
    """Save each split as a parquet file in data/splits/."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        output_path = SPLITS_DIR / f"{name}.parquet"
        df.to_parquet(output_path, index=False)
        print(f"  Saved {name}: {len(df):,} rows → {output_path.name}")


if __name__ == "__main__":
    # Load ingested data
    ingested_path = REPO_ROOT / "data" / "processed" / "vt_crashes_ingested.parquet"
    if not ingested_path.exists():
        raise FileNotFoundError(
            f"Ingested data not found at {ingested_path}. "
            "Run src.ingestion.loader first."
        )

    print(f"Loading ingested data from {ingested_path}...")
    df = pd.read_parquet(ingested_path)
    print(f"  {len(df):,} rows loaded\n")

    # Split
    splits = split_data(df)

    # Validate
    validate_splits(splits)

    # Save
    print("\nSaving splits...")
    save_splits(splits)
    print("\nDone.")