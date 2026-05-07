"""
window_generator.py — Generates rolling 30-day prediction windows with 7-day step.

For each segment, creates one observation per window:
  - Label: did a fatal/injury crash occur in this segment during the 30-day window?
  - Window start dates are spaced 7 days apart
  - Windows that cross split boundaries are dropped
  - Features are NOT computed here (that is a separate step)

Leakage mitigations:
  1. Windows crossing split boundaries are dropped (hard enforcement)
  2. Labels use only events within [window_start, window_end)
  3. Feature computation (separate module) must use only data before window_start

See docs/technical/03_prediction_unit.md v1.1 for segment definitions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"

WINDOW_DAYS = 30
STEP_DAYS = 7


def generate_windows_for_split(
    df: pd.DataFrame,
    split_start: pd.Timestamp,
    split_end: pd.Timestamp,
    window_days: int = WINDOW_DAYS,
    step_days: int = STEP_DAYS,
) -> pd.DataFrame:
    """Generate labeled (segment, window) observations for one split.

    Args:
        df: Crash-level DataFrame for this split (must have event_date,
            segment_id, segment_type, target, severity_class)
        split_start: First allowable window start date
        split_end: Last date in this split (windows extending past this are dropped)
        window_days: Length of prediction window in days
        step_days: Days between consecutive window start dates

    Returns:
        DataFrame with columns:
            segment_id, segment_type, window_start, window_end,
            label, fatal_count, injury_count, pdo_count, total_crashes
    """
    # Filter to rows with valid segment assignment
    df = df[df["segment_id"].notna()].copy()

    if len(df) == 0:
        print("  WARNING: No rows with valid segment_id")
        return pd.DataFrame()

    # Generate all window start dates within the split
    window_starts = pd.date_range(
        start=split_start,
        end=split_end - pd.Timedelta(days=window_days),
        freq=f"{step_days}D"
    )

    if len(window_starts) == 0:
        print("  WARNING: No valid windows (split too short for window size)")
        return pd.DataFrame()

    # Get all unique segments
    segments = df[["segment_id", "segment_type"]].drop_duplicates()

    print(f"  Generating {len(window_starts)} windows x {len(segments)} segments...")

    # Pre-index crashes by segment for fast lookup
    crash_dates = df.groupby("segment_id").apply(
        lambda g: g[["event_date", "target", "severity_class"]].reset_index(drop=True),
        include_groups=False
    )

    observations = []

    for ws in window_starts:
        we = ws + pd.Timedelta(days=window_days)

        # LEAKAGE CHECK: window must not extend past split boundary
        if we > split_end + pd.Timedelta(days=1):
            continue

        for _, seg_row in segments.iterrows():
            sid = seg_row["segment_id"]
            stype = seg_row["segment_type"]

            # Get crashes for this segment in this window
            try:
                seg_crashes = crash_dates.loc[sid]
                in_window = seg_crashes[
                    (seg_crashes["event_date"] >= ws) &
                    (seg_crashes["event_date"] < we)
                ]
            except KeyError:
                in_window = pd.DataFrame()

            if len(in_window) == 0:
                fatal_count = 0
                injury_count = 0
                pdo_count = 0
                total = 0
                label = 0
            else:
                fatal_count = (in_window["severity_class"] == "fatal").sum()
                injury_count = (in_window["severity_class"] == "injury").sum()
                pdo_count = (in_window["severity_class"] == "pdo").sum()
                total = len(in_window)
                label = 1 if (fatal_count + injury_count) > 0 else 0

            observations.append({
                "segment_id": sid,
                "segment_type": stype,
                "window_start": ws,
                "window_end": we,
                "label": label,
                "has_fatal": 1 if fatal_count > 0 else 0,
                "has_injury": 1 if injury_count > 0 else 0,
                "fatal_count": fatal_count,
                "injury_count": injury_count,
                "pdo_count": pdo_count,
                "total_crashes": total,
            })

    result = pd.DataFrame(observations)

    # Summary
    if len(result) > 0:
        positive = result["label"].sum()
        fatal_windows = result["has_fatal"].sum()
        print(f"  Total observations: {len(result):,}")
        print(f"  Positive (fatal+injury): {positive:,} ({100*positive/len(result):.2f}%)")
        print(f"  Windows with fatal: {fatal_windows:,}")
        print(f"  Class ratio: 1:{int((len(result)-positive)/max(positive,1))}")

    return result


def generate_all_windows() -> Dict[str, pd.DataFrame]:
    """Generate windows for all splits. Returns dict of split_name -> observations."""
    import yaml

    config_path = REPO_ROOT / "config" / "splits.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    gap_days = config["gap_days"]
    split_names = list(config["splits"].keys())

    results = {}

    for i, name in enumerate(split_names):
        print(f"\n{'='*60}")
        print(f"Processing split: {name}")
        print(f"{'='*60}")

        # Load split parquet
        split_path = SPLITS_DIR / f"{name}.parquet"
        if not split_path.exists():
            print(f"  SKIP: {split_path} not found")
            continue

        df = pd.read_parquet(split_path)

        # Compute effective boundaries (same logic as splitter.py)
        raw_start = pd.Timestamp(config["splits"][name]["start"])
        raw_end_value = config["splits"][name]["end"]
        if raw_end_value == "auto":
            raw_end = pd.Timestamp.now().normalize()
        else:
            raw_end = pd.Timestamp(raw_end_value)

        if i == 0:
            effective_start = raw_start
        else:
            prev_end_value = config["splits"][split_names[i-1]]["end"]
            if prev_end_value == "auto":
                prev_end = pd.Timestamp.now().normalize()
            else:
                prev_end = pd.Timestamp(prev_end_value)
            effective_start = prev_end + pd.Timedelta(days=gap_days)

        effective_end = raw_end

        print(f"  Date range: {effective_start.strftime('%Y-%m-%d')} to {effective_end.strftime('%Y-%m-%d')}")
        print(f"  Crash rows: {len(df):,}")

        # Generate windows
        windows = generate_windows_for_split(
            df, effective_start, effective_end
        )

        if len(windows) > 0:
            results[name] = windows

            # Save
            output_path = SPLITS_DIR / f"{name}_windows.parquet"
            windows.to_parquet(output_path, index=False)
            print(f"  Saved: {output_path.name}")

    return results


if __name__ == "__main__":
    results = generate_all_windows()

    print(f"\n{'='*60}")
    print("WINDOW GENERATION SUMMARY")
    print(f"{'='*60}")
    for name, df in results.items():
        positive = df["label"].sum()
        fatal_w = df["has_fatal"].sum()
        print(
            f"  {name:6s}: {len(df):>10,} observations | "
            f"Positive: {positive:>6,} ({100*positive/len(df):.2f}%) | "
            f"Fatal windows: {fatal_w:>4}"
        )
    print(f"{'='*60}")