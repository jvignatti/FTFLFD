"""
checks.py — Leakage detection and prevention assertions.

Per CLAUDE.md Leakage Rules (Comprehensive):
  - Temporal: no feature uses future data relative to prediction date
  - Target encoding: encodings fit on training only
  - Spatial: neighbor outcomes not used without time lag
  - Global statistics: no full-dataset aggregates used as features
  - Split boundaries: no data crosses split dates, gaps enforced

This module provides:
  1. run_all_checks() — full leakage audit, returns structured report
  2. Individual check functions for targeted verification
  3. CLI entry point via: python run.py check-leakage

Returns a structured report with PASS/FAIL per check and details
for any violations found. NOT just pass/fail — violations are listed.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
CONFIG_DIR = REPO_ROOT / "config"


def load_split_config() -> dict:
    config_path = CONFIG_DIR / "splits.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def check_split_date_boundaries() -> dict:
    """Verify no crash date in any split falls outside its declared boundary.

    Loads each split parquet and confirms every event_date falls within
    the effective date range (including gap enforcement).
    """
    result = {
        "check": "Split Date Boundaries",
        "passed": True,
        "violations": [],
        "details": {},
    }

    config = load_split_config()
    gap_days = config["gap_days"]
    split_names = list(config["splits"].keys())

    for i, name in enumerate(split_names):
        split_path = SPLITS_DIR / f"{name}.parquet"
        if not split_path.exists():
            result["violations"].append(f"{name}.parquet not found — cannot verify")
            result["passed"] = False
            continue

        df = pd.read_parquet(split_path)
        if "event_date" not in df.columns:
            result["violations"].append(f"{name}: no event_date column")
            result["passed"] = False
            continue

        # Compute effective boundaries
        raw_start = pd.Timestamp(config["splits"][name]["start"])
        raw_end_value = config["splits"][name]["end"]
        if raw_end_value == "auto":
            raw_end = pd.Timestamp.now().normalize()
        else:
            raw_end = pd.Timestamp(raw_end_value)

        if i == 0:
            effective_start = raw_start
        else:
            prev_end_value = config["splits"][split_names[i - 1]]["end"]
            if prev_end_value == "auto":
                prev_end = pd.Timestamp.now().normalize()
            else:
                prev_end = pd.Timestamp(prev_end_value)
            effective_start = prev_end + timedelta(days=gap_days)

        effective_end = raw_end

        min_date = df["event_date"].min()
        max_date = df["event_date"].max()

        before_start = (df["event_date"] < effective_start).sum()
        after_end = (df["event_date"] > effective_end).sum()

        split_detail = {
            "effective_range": f"{effective_start.strftime('%Y-%m-%d')} to {effective_end.strftime('%Y-%m-%d')}",
            "actual_range": f"{min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}",
            "rows": len(df),
            "before_start": int(before_start),
            "after_end": int(after_end),
        }
        result["details"][name] = split_detail

        if before_start > 0:
            result["violations"].append(
                f"{name}: {before_start} rows before effective start {effective_start.strftime('%Y-%m-%d')}"
            )
            result["passed"] = False

        if after_end > 0:
            result["violations"].append(
                f"{name}: {after_end} rows after effective end {effective_end.strftime('%Y-%m-%d')}"
            )
            result["passed"] = False

    return result


def check_split_id_overlap() -> dict:
    """Verify no OBJECTID appears in more than one split."""
    result = {
        "check": "Split ID Overlap",
        "passed": True,
        "violations": [],
        "details": {},
    }

    config = load_split_config()
    split_names = list(config["splits"].keys())

    id_sets = {}
    for name in split_names:
        split_path = SPLITS_DIR / f"{name}.parquet"
        if not split_path.exists():
            continue
        df = pd.read_parquet(split_path, columns=["OBJECTID"])
        id_sets[name] = set(df["OBJECTID"].dropna())

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            name_a = split_names[i]
            name_b = split_names[j]
            if name_a not in id_sets or name_b not in id_sets:
                continue
            overlap = id_sets[name_a] & id_sets[name_b]
            if len(overlap) > 0:
                result["violations"].append(
                    f"{name_a} <-> {name_b}: {len(overlap)} shared OBJECTIDs"
                )
                result["passed"] = False
            result["details"][f"{name_a}_vs_{name_b}"] = {
                "overlap_count": len(overlap)
            }

    return result


def check_gap_enforcement() -> dict:
    """Verify minimum gap between consecutive splits is respected."""
    result = {
        "check": "Gap Enforcement",
        "passed": True,
        "violations": [],
        "details": {},
    }

    config = load_split_config()
    gap_days = config["gap_days"]
    split_names = list(config["splits"].keys())

    for i in range(len(split_names) - 1):
        name_a = split_names[i]
        name_b = split_names[i + 1]

        path_a = SPLITS_DIR / f"{name_a}.parquet"
        path_b = SPLITS_DIR / f"{name_b}.parquet"
        if not path_a.exists() or not path_b.exists():
            continue

        df_a = pd.read_parquet(path_a, columns=["event_date"])
        df_b = pd.read_parquet(path_b, columns=["event_date"])

        if len(df_a) == 0 or len(df_b) == 0:
            continue

        max_a = df_a["event_date"].max()
        min_b = df_b["event_date"].min()
        actual_gap = (min_b - max_a).days

        pair_detail = {
            "max_date_a": max_a.strftime("%Y-%m-%d"),
            "min_date_b": min_b.strftime("%Y-%m-%d"),
            "actual_gap_days": actual_gap,
            "required_gap_days": gap_days,
        }
        result["details"][f"{name_a}_to_{name_b}"] = pair_detail

        if actual_gap < gap_days:
            result["violations"].append(
                f"{name_a} -> {name_b}: gap is {actual_gap} days, required {gap_days}"
            )
            result["passed"] = False

    return result


def check_chronological_order() -> dict:
    """Verify each split is internally sorted by event_date."""
    result = {
        "check": "Chronological Order",
        "passed": True,
        "violations": [],
        "details": {},
    }

    config = load_split_config()
    split_names = list(config["splits"].keys())

    for name in split_names:
        split_path = SPLITS_DIR / f"{name}.parquet"
        if not split_path.exists():
            continue

        df = pd.read_parquet(split_path, columns=["event_date"])
        if len(df) == 0:
            continue

        dates = df["event_date"].values
        is_sorted = bool(np.all(dates[:-1] <= dates[1:]))
        result["details"][name] = {"sorted": is_sorted, "rows": len(df)}

        if not is_sorted:
            result["violations"].append(f"{name}: not chronologically sorted")
            result["passed"] = False

    return result


def check_window_label_integrity() -> dict:
    """Verify window labels match the actual crashes in each window.

    Loads window observations and the corresponding crash-level split.
    For a sample of windows, verifies that:
      - label=1 windows actually contain fatal/injury crashes
      - label=0 windows actually contain no fatal/injury crashes
    """
    result = {
        "check": "Window Label Integrity",
        "passed": True,
        "violations": [],
        "details": {},
    }

    config = load_split_config()

    for name in ["train", "val"]:
        windows_path = SPLITS_DIR / f"{name}_windows.parquet"
        crashes_path = SPLITS_DIR / f"{name}.parquet"

        if not windows_path.exists() or not crashes_path.exists():
            continue

        windows = pd.read_parquet(windows_path)
        crashes = pd.read_parquet(crashes_path)

        # Sample 1000 windows for verification (full check would be too slow)
        sample_size = min(1000, len(windows))
        sample = windows.sample(n=sample_size, random_state=42)

        mismatches = 0
        checked = 0

        for _, row in sample.iterrows():
            sid = row["segment_id"]
            ws = row["window_start"]
            we = row["window_end"]
            label = row["label"]

            seg_crashes = crashes[
                (crashes["segment_id"] == sid) &
                (crashes["event_date"] >= ws) &
                (crashes["event_date"] < we) &
                (crashes["target"] == 1)
            ]

            actual_label = 1 if len(seg_crashes) > 0 else 0
            if actual_label != label:
                mismatches += 1

            checked += 1

        result["details"][name] = {
            "checked": checked,
            "mismatches": mismatches,
        }

        if mismatches > 0:
            result["violations"].append(
                f"{name}: {mismatches}/{checked} sampled windows have incorrect labels"
            )
            result["passed"] = False

    return result


def check_feature_temporal_integrity() -> dict:
    """Verify segment rate features are computed from training data only.

    Checks that the segment rates in val/benchmark featured splits
    match the rates computed from training crashes. If they differ,
    it means rates were computed from non-training data (leakage).
    """
    result = {
        "check": "Feature Temporal Integrity",
        "passed": True,
        "violations": [],
        "details": {},
    }

    train_featured_path = SPLITS_DIR / "train_featured.parquet"
    val_featured_path = SPLITS_DIR / "val_featured.parquet"

    if not train_featured_path.exists() or not val_featured_path.exists():
        result["violations"].append("Featured splits not found — cannot verify")
        result["passed"] = False
        return result

    train_df = pd.read_parquet(train_featured_path)
    val_df = pd.read_parquet(val_featured_path)

    # Get segment rates from training data
    rate_cols = [
        "segment_crash_rate", "segment_fatal_rate",
        "segment_injury_rate", "segment_pdo_rate"
    ]

    train_rates = train_df.groupby("segment_id")[rate_cols].first()

    # For segments that appear in both train and val,
    # verify the rates are identical
    val_rates = val_df.groupby("segment_id")[rate_cols].first()

    shared_segments = train_rates.index.intersection(val_rates.index)

    if len(shared_segments) == 0:
        result["details"]["shared_segments"] = 0
        return result

    train_shared = train_rates.loc[shared_segments]
    val_shared = val_rates.loc[shared_segments]

    mismatches = 0
    for col in rate_cols:
        diff = (train_shared[col] - val_shared[col]).abs()
        col_mismatches = (diff > 1e-6).sum()
        if col_mismatches > 0:
            mismatches += col_mismatches
            result["violations"].append(
                f"{col}: {col_mismatches} segments have different rates in train vs val"
            )

    if mismatches > 0:
        result["passed"] = False

    result["details"] = {
        "shared_segments": len(shared_segments),
        "rate_columns_checked": rate_cols,
        "total_mismatches": mismatches,
    }

    return result


def check_benchmark_not_in_training() -> dict:
    """Verify no benchmark data was used in training features or labels.

    Checks that the date ranges of benchmark splits do not overlap
    with the training period, including a gap buffer.
    """
    result = {
        "check": "Benchmark Isolation",
        "passed": True,
        "violations": [],
        "details": {},
    }

    config = load_split_config()
    gap_days = config["gap_days"]

    train_path = SPLITS_DIR / "train.parquet"
    if not train_path.exists():
        result["violations"].append("train.parquet not found")
        result["passed"] = False
        return result

    train_df = pd.read_parquet(train_path, columns=["event_date"])
    train_max = train_df["event_date"].max()

    for bench_name in ["b1", "b2", "b3", "b4"]:
        bench_path = SPLITS_DIR / f"{bench_name}.parquet"
        if not bench_path.exists():
            continue

        bench_df = pd.read_parquet(bench_path, columns=["event_date"])
        if len(bench_df) == 0:
            continue

        bench_min = bench_df["event_date"].min()
        gap = (bench_min - train_max).days

        result["details"][bench_name] = {
            "train_max": train_max.strftime("%Y-%m-%d"),
            "bench_min": bench_min.strftime("%Y-%m-%d"),
            "gap_days": gap,
        }

        if gap < gap_days:
            result["violations"].append(
                f"{bench_name}: only {gap} days after training end "
                f"(required: {gap_days})"
            )
            result["passed"] = False

    return result


def run_all_checks() -> Dict[str, dict]:
    """Run all leakage checks and return structured report."""
    checks = [
        ("1. Split Date Boundaries", check_split_date_boundaries),
        ("2. Split ID Overlap", check_split_id_overlap),
        ("3. Gap Enforcement", check_gap_enforcement),
        ("4. Chronological Order", check_chronological_order),
        ("5. Window Label Integrity", check_window_label_integrity),
        ("6. Feature Temporal Integrity", check_feature_temporal_integrity),
        ("7. Benchmark Isolation", check_benchmark_not_in_training),
    ]

    print("=" * 60)
    print("LEAKAGE CHECK REPORT")
    print("=" * 60)

    all_passed = True
    results = {}

    for check_name, check_fn in checks:
        print(f"\n  Running {check_name}...")
        try:
            result = check_fn()
        except Exception as e:
            result = {
                "check": check_name,
                "passed": False,
                "violations": [f"ERROR: {str(e)}"],
                "details": {},
            }

        status = "PASS" if result["passed"] else "FAIL"
        print(f"    {status}")

        if result["violations"]:
            for v in result["violations"]:
                print(f"    ! {v}")

        if not result["passed"]:
            all_passed = False

        results[check_name] = result

    print(f"\n{'='*60}")
    if all_passed:
        print("  ALL LEAKAGE CHECKS PASSED")
    else:
        failed = [name for name, r in results.items() if not r["passed"]]
        print(f"  {len(failed)} CHECK(S) FAILED:")
        for f in failed:
            print(f"    - {f}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_all_checks()