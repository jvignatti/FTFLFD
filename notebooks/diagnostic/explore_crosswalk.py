"""
explore_crosswalk.py
--------------------
Diagnostic: explore LRSNUMBER to AADT StandardRouteCode join feasibility.

Purpose
-------
Investigates route identifier overlap between crash LRSNUMBER field and
AADT StandardRouteCode field. Tests AOTROUTE and AOTROUTEid as bridge fields.
Establishes milepoint range comparison between crash locations and AADT sections.

Key findings
------------
- Direct LRSNUMBER == StandardRouteCode join is the primary viable path
- AOTROUTE and AOTROUTEid do not provide meaningful additional coverage
- StandardRouteCode covers state highway system at ~65% route-level overlap
- Milepoint interval join required for record-level matching

Date: 2026-05-16
Status: Diagnostic — informs AADT join strategy in aadt_coverage_analysis.py
LRS methodology credit: Owen Mosley
See: docs/technical/07_aadt_coverage_diagnosis.md
"""

import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
RAW_DIR = REPO_ROOT / "data" / "raw"

# Load crash training data
print("Loading crash data...")
crashes = pd.read_parquet(SPLITS_DIR / "train.parquet")
print(f"  {len(crashes):,} crash records")

# Load AADT data
print("\nLoading AADT data...")
aadt_limited = pd.read_csv(RAW_DIR / "aadt_limited.csv")
aadt_other = pd.read_csv(RAW_DIR / "aadt_other.csv")
aadt = pd.concat([aadt_limited, aadt_other], ignore_index=True)
print(f"  {len(aadt):,} AADT records")

# What route identifiers exist in crash data?
print("\n=== CRASH DATA ROUTE FIELDS ===")
print(f"  LRSNUMBER unique values: {crashes['LRSNUMBER'].nunique()}")
print(f"  LRSNUMBER sample: {sorted(crashes['LRSNUMBER'].dropna().unique()[:20])}")

# Check if AOTROUTE exists and what it looks like
if "AOTROUTE" in crashes.columns:
    print(f"\n  AOTROUTE unique values: {crashes['AOTROUTE'].nunique()}")
    print(f"  AOTROUTE sample: {sorted(crashes['AOTROUTE'].dropna().unique()[:20])}")

if "AOTROUTEid" in crashes.columns:
    print(f"\n  AOTROUTEid unique values: {crashes['AOTROUTEid'].nunique()}")
    print(f"  AOTROUTEid sample: {sorted(crashes['AOTROUTEid'].dropna().unique()[:20])}")

# What route identifiers exist in AADT data?
print("\n=== AADT ROUTE FIELDS ===")
print(f"  StandardRouteCode unique: {aadt['StandardRouteCode'].nunique()}")
print(f"  StandardRouteCode sample: {sorted(aadt['StandardRouteCode'].dropna().unique()[:20])}")

print(f"\n  RouteNum unique: {aadt['RouteNum'].nunique()}")
print(f"  RouteNum sample: {sorted(aadt['RouteNum'].dropna().unique()[:20])}")

print(f"\n  RouteType unique: {aadt['RouteType'].nunique()}")
print(f"  RouteType values: {sorted(aadt['RouteType'].dropna().unique())}")

# Try to find overlapping identifiers
print("\n=== LOOKING FOR BRIDGES ===")

# Check if any LRSNUMBER values appear in StandardRouteCode
crash_lrs = set(crashes["LRSNUMBER"].dropna().unique())
aadt_codes = set(aadt["StandardRouteCode"].dropna().unique())

# Check if LRSNUMBER is embedded in StandardRouteCode
print("\n  Checking numeric extraction from StandardRouteCode...")
aadt_nums = {}
for code in sorted(aadt["StandardRouteCode"].dropna().unique()):
    parts = code.split("-")
    if len(parts) >= 2:
        route_prefix = parts[0]  # e.g., V104, I091, U002
        aadt_nums[code] = route_prefix

print(f"  Route prefixes in AADT: {sorted(set(aadt_nums.values()))[:30]}")

# Check AOTROUTEid in crashes vs RouteNum in AADT
if "AOTROUTEid" in crashes.columns:
    crash_route_ids = set(str(x) for x in crashes["AOTROUTEid"].dropna().unique())
    aadt_route_nums = set(str(x) for x in aadt["RouteNum"].dropna().unique())
    overlap = crash_route_ids & aadt_route_nums
    print(f"\n  AOTROUTEid vs AADT RouteNum overlap: {len(overlap)} matches")
    if overlap:
        print(f"  Sample overlaps: {sorted(list(overlap))[:20]}")

# Check AOTROUTE text vs AADT RouteName
if "AOTROUTE" in crashes.columns:
    crash_routes = set(crashes["AOTROUTE"].dropna().unique())
    aadt_routes = set(aadt["RouteName"].dropna().unique())
    overlap = crash_routes & aadt_routes
    print(f"\n  AOTROUTE vs AADT RouteName overlap: {len(overlap)} matches")
    if overlap:
        print(f"  Sample overlaps: {sorted(list(overlap))[:20]}")

# Most promising: check if crash LRSNUMBER maps to AADT by shared milepoint ranges
print("\n=== CRASH LRS MILEPOINT RANGES (top 20 by crash count) ===")
lrs_stats = crashes[crashes["LRSNUMBER"].notna()].groupby("LRSNUMBER").agg(
    crash_count=("OBJECTID", "count"),
    min_mp=("AOTACTUALMILEPOINT", "min"),
    max_mp=("AOTACTUALMILEPOINT", "max"),
    sample_route=("AOTROUTE", "first"),
).sort_values("crash_count", ascending=False).head(20)
print(lrs_stats.to_string())

# Show AADT segments for comparison
print("\n=== AADT MILEPOINT RANGES (sample by route) ===")
aadt_stats = aadt.groupby("StandardRouteCode").agg(
    segment_count=("OBJECTID", "count"),
    min_begin=("BeginMM", "min"),
    max_end=("EndMM", "max"),
    mean_aadt=("AADT", "mean"),
).sort_values("segment_count", ascending=False).head(20)
print(aadt_stats.to_string())
