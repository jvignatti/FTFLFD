"""
aadt_features.py — Integrates AADT data into the feature pipeline.

Joins Vermont AADT (Annual Average Daily Traffic) to crash segments
using direct LRSNUMBER == StandardRouteCode matching with milepoint overlap.

Source: VTrans ArcGIS REST FeatureServer (2024 AADT, single year)
Join method: route code + milepoint range overlap (no spatial join needed)

New features derivable:
  - segment_aadt: raw traffic volume
  - segment_crash_rate_per_vmt: crashes normalized by exposure
  - functional_class: FHWA road classification (1-6)
  - is_divided: whether road has divided lanes
"""

import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
SPLITS_DIR = REPO_ROOT / "data" / "splits"


def load_aadt() -> pd.DataFrame:
    """Load and combine both AADT layers."""
    limited = pd.read_csv(RAW_DIR / "aadt_limited.csv")
    other = pd.read_csv(RAW_DIR / "aadt_other.csv")
    aadt = pd.concat([limited, other], ignore_index=True)

    # Clean
    aadt["StandardRouteCode"] = aadt["StandardRouteCode"].astype(str).str.strip()
    aadt["BeginMM"] = pd.to_numeric(aadt["BeginMM"], errors="coerce")
    aadt["EndMM"] = pd.to_numeric(aadt["EndMM"], errors="coerce")
    aadt["AADT"] = pd.to_numeric(aadt["AADT"], errors="coerce")
    aadt["FunctionalClass"] = pd.to_numeric(aadt["FunctionalClass"], errors="coerce")

    # Drop rows without usable data
    before = len(aadt)
    aadt = aadt.dropna(subset=["StandardRouteCode", "BeginMM", "EndMM", "AADT"])
    dropped = before - len(aadt)
    if dropped > 0:
        print(f"  Dropped {dropped} AADT records with missing key fields")

    print(f"  Loaded {len(aadt):,} AADT records")
    print(f"  Routes: {aadt['StandardRouteCode'].nunique()}")
    print(f"  AADT range: {aadt['AADT'].min():,.0f} to {aadt['AADT'].max():,.0f}")

    return aadt


def build_segment_aadt_lookup(aadt: pd.DataFrame) -> pd.DataFrame:
    """Build a lookup table mapping (route, mile_floor) to AADT.

    For each integer milepoint on each route, finds the AADT segment
    that covers it (BeginMM <= mile_floor < EndMM).

    Returns DataFrame with columns: LRSNUMBER, mile_floor, aadt_value,
    functional_class, is_divided
    """
    lookups = []

    for _, row in aadt.iterrows():
        route = row["StandardRouteCode"]
        begin = row["BeginMM"]
        end = row["EndMM"]
        aadt_val = row["AADT"]
        func_class = row["FunctionalClass"]
        is_div = 1 if row.get("IsDivided") == "Y" else 0

        # Generate all integer milepoints this AADT segment covers
        start_mile = int(np.floor(begin))
        end_mile = int(np.floor(end))

        for mile in range(start_mile, end_mile + 1):
            lookups.append({
                "LRSNUMBER": route,
                "mile_floor": mile,
                "aadt_value": aadt_val,
                "functional_class": func_class,
                "is_divided": is_div,
            })

    lookup_df = pd.DataFrame(lookups)

    # Handle overlapping AADT segments (same route+mile covered by multiple records)
    # Take the one with highest AADT (conservative — assumes more traffic)
    if len(lookup_df) > 0:
        before = len(lookup_df)
        lookup_df = lookup_df.sort_values("aadt_value", ascending=False)
        lookup_df = lookup_df.drop_duplicates(subset=["LRSNUMBER", "mile_floor"], keep="first")
        deduped = before - len(lookup_df)
        if deduped > 0:
            print(f"  Resolved {deduped} overlapping AADT segments (kept highest AADT)")

    print(f"  Lookup table: {len(lookup_df):,} (route, milepoint) entries")
    print(f"  Covering {lookup_df['LRSNUMBER'].nunique()} routes")

    return lookup_df


def join_aadt_to_crashes(crash_df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
    """Join AADT data to crash-level DataFrame.

    Matches on LRSNUMBER == LRSNUMBER and floor(AOTACTUALMILEPOINT) == mile_floor.
    Crashes without AADT coverage get null values.
    """
    df = crash_df.copy()

    # Compute mile floor for joining (same as segment assignment)
    df["mile_floor"] = np.floor(df["AOTACTUALMILEPOINT"]).astype("Int64")

    # Join
    df = df.merge(
        lookup_df,
        on=["LRSNUMBER", "mile_floor"],
        how="left"
    )

    # Report coverage
    has_aadt = df["aadt_value"].notna().sum()
    total = len(df)
    coverage = has_aadt / total if total > 0 else 0

    print(f"  AADT coverage: {has_aadt:,} / {total:,} ({coverage:.1%})")
    print(f"  Missing AADT: {total - has_aadt:,}")

    # Clean up
    df = df.drop(columns=["mile_floor"], errors="ignore")

    return df


def compute_aadt_features(
    featured_df: pd.DataFrame,
    segment_aadt: pd.DataFrame,
) -> pd.DataFrame:
    """Add AADT-derived features to window-level featured DataFrame.

    Args:
        featured_df: Window-level observations with segment_id
        segment_aadt: Per-segment AADT lookup (segment_id -> aadt, func_class, etc.)

    Returns:
        featured_df with new columns added
    """
    df = featured_df.merge(segment_aadt, on="segment_id", how="left")

    # Fill missing AADT with 0 (grid segments and uncovered LRS segments)
    missing = df["segment_aadt"].isna().sum()
    if missing > 0:
        print(f"  {missing:,} observations without AADT (set to 0)")

    df["segment_aadt"] = df["segment_aadt"].fillna(0)
    df["segment_functional_class"] = df["segment_functional_class"].fillna(0)
    df["segment_is_divided"] = df["segment_is_divided"].fillna(0)

    # Compute crash rate per VMT (vehicle-miles-traveled)
    # VMT = AADT * segment_length(1 mile) * 365 days
    # Rate = crashes / VMT
    df["segment_crash_rate_per_vmt"] = np.where(
        df["segment_aadt"] > 0,
        df["segment_crash_rate"] / (df["segment_aadt"] * 365 / 1_000_000),
        0
    )

    df["segment_fatal_rate_per_vmt"] = np.where(
        df["segment_aadt"] > 0,
        df["segment_fatal_rate"] / (df["segment_aadt"] * 365 / 1_000_000),
        0
    )

    return df


def build_segment_aadt_table(crash_df: pd.DataFrame, lookup_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-segment AADT summary for joining to window observations.

    Groups crash data by segment_id and assigns AADT from the lookup table.
    Returns one row per segment with AADT and structural features.
    """
    # Get segments with LRS data
    lrs_segments = crash_df[
        (crash_df["segment_id"].notna()) &
        (crash_df["LRSNUMBER"].notna()) &
        (crash_df["AOTACTUALMILEPOINT"].notna())
    ].copy()

    lrs_segments["mile_floor"] = np.floor(lrs_segments["AOTACTUALMILEPOINT"]).astype("Int64")

    # Get unique segment -> (LRSNUMBER, mile_floor) mapping
    segment_lrs = lrs_segments.groupby("segment_id").agg(
        LRSNUMBER=("LRSNUMBER", "first"),
        mile_floor=("mile_floor", "first"),
    ).reset_index()

    # Join AADT
    segment_lrs = segment_lrs.merge(lookup_df, on=["LRSNUMBER", "mile_floor"], how="left")

    # Rename for clarity
    segment_aadt = segment_lrs[["segment_id", "aadt_value", "functional_class", "is_divided"]].copy()
    segment_aadt = segment_aadt.rename(columns={
        "aadt_value": "segment_aadt",
        "functional_class": "segment_functional_class",
        "is_divided": "segment_is_divided",
    })

    has_aadt = segment_aadt["segment_aadt"].notna().sum()
    total = len(segment_aadt)
    print(f"  Segments with AADT: {has_aadt:,} / {total:,} ({has_aadt/total:.1%})")

    return segment_aadt


if __name__ == "__main__":
    print("=" * 60)
    print("AADT DATA EXPLORATION")
    print("=" * 60)

    print("\nLoading AADT...")
    aadt = load_aadt()

    print("\nBuilding lookup table...")
    lookup = build_segment_aadt_lookup(aadt)

    print("\nLoading training crashes...")
    train = pd.read_parquet(SPLITS_DIR / "train.parquet")

    print("\nBuilding segment AADT table...")
    segment_aadt = build_segment_aadt_table(train, lookup)

    print(f"\n  AADT distribution for matched segments:")
    matched = segment_aadt[segment_aadt["segment_aadt"].notna()]
    if len(matched) > 0:
        print(f"    Min:    {matched['segment_aadt'].min():,.0f}")
        print(f"    Median: {matched['segment_aadt'].median():,.0f}")
        print(f"    Mean:   {matched['segment_aadt'].mean():,.0f}")
        print(f"    Max:    {matched['segment_aadt'].max():,.0f}")

    print(f"\n  Functional class distribution:")
    print(matched["segment_functional_class"].value_counts().sort_index().to_string())

    print(f"\n  Is divided distribution:")
    print(matched["segment_is_divided"].value_counts().to_string())

    print("\n" + "=" * 60)
    print("AADT exploration complete.")
    print("Ready for 3-gate feature testing after window generation finishes.")
    print("=" * 60)