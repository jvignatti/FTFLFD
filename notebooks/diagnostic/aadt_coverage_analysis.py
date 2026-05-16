"""
aadt_coverage_analysis.py
--------------------------
Diagnostic: compute AADT join coverage at record, segment, and severity levels.

Purpose
-------
Quantifies how many crash records (overall and by severity class) successfully
join to a matching AADT interval on the same route. Results from this script
informed the Gate 1 availability decision documented in
docs/technical/07_aadt_coverage_diagnosis.md.

Key findings from initial run (Phase 2 train.parquet, 2010-2022):
    All records:  58.06%
    Fatal only:   70.85%
    Injury:       67.53%
    PDO:          55.58%

NOTE: These numbers were computed on the Phase 2 training set (2010-2022).
They must be recomputed after reverting to Phase 1 splits (2010-2019).
See docs/technical/07_aadt_coverage_diagnosis.md, Section 4.

Run from project root:
    python notebooks/diagnostic/aadt_coverage_analysis.py

Outputs: printed to stdout.
    python notebooks/diagnostic/aadt_coverage_analysis.py > notebooks/diagnostic/output/aadt_coverage_output.txt
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths — relative to project root (FTFLFD/)
# ---------------------------------------------------------------------------
TRAIN_PARQUET    = "data/splits/train.parquet"
AADT_LIMITED_CSV = "data/raw/aadt_limited.csv"
AADT_OTHER_CSV   = "data/raw/aadt_other.csv"

# ---------------------------------------------------------------------------
# Load and combine AADT files
# ---------------------------------------------------------------------------
print("Loading data...")
crashes = pd.read_parquet(TRAIN_PARQUET)
crashes["crash_id"] = np.arange(len(crashes))

aadt = pd.concat([
    pd.read_csv(AADT_LIMITED_CSV),
    pd.read_csv(AADT_OTHER_CSV)
]).drop_duplicates(subset=["StandardRouteCode", "BeginMM", "EndMM"])

print(f"Crashes loaded:        {len(crashes):,} records")
print(f"AADT sections loaded:  {len(aadt):,} sections")

# ---------------------------------------------------------------------------
# Join: crash LRSNUMBER -> AADT StandardRouteCode
# Filter to rows where crash milepoint falls within AADT section interval
# ---------------------------------------------------------------------------
joined = crashes.merge(
    aadt[["StandardRouteCode", "BeginMM", "EndMM",
          "AADT", "FunctionalClass", "IsDivided"]],
    left_on="LRSNUMBER",
    right_on="StandardRouteCode",
    how="left"
)

joined["aadt_match"] = (
    joined["AADT"].notna()
    & (joined["AOTACTUALMILEPOINT"] >= joined["BeginMM"])
    & (joined["AOTACTUALMILEPOINT"] <= joined["EndMM"])
)

# Collapse back to one row per crash (max match across all joined AADT rows)
record_match = joined.groupby("crash_id")["aadt_match"].max()

# ---------------------------------------------------------------------------
# Coverage: overall
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("RECORD-LEVEL AADT COVERAGE")
print("="*60)
print(f"All records:  {record_match.mean():.2%}  ({record_match.sum():,} / {len(record_match):,})")

# ---------------------------------------------------------------------------
# Coverage: by severity class
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("COVERAGE BY SEVERITY CLASS")
print("="*60)
for sev in ["fatal", "injury", "pdo"]:
    ids = crashes[crashes["severity_class"] == sev]["crash_id"]
    cov = record_match[record_match.index.isin(ids)]
    print(f"  {sev:<8}: {cov.mean():.2%}  ({cov.sum():,} / {len(cov):,})")

# ---------------------------------------------------------------------------
# Coverage: segment-level (LRSNUMBER + mile_bin + year)
# ---------------------------------------------------------------------------
crashes["mile_bin"] = np.floor(crashes["AOTACTUALMILEPOINT"]).astype("Int64")
crashes["year"]     = pd.to_datetime(crashes["event_date"]).dt.year

seg_match = (
    joined.assign(
        mile_bin=np.floor(joined["AOTACTUALMILEPOINT"]).astype("Int64"),
        year=pd.to_datetime(joined["event_date"]).dt.year
    )
    .groupby(["LRSNUMBER", "mile_bin", "year"])["aadt_match"]
    .max()
)
print("\n" + "="*60)
print("SEGMENT-LEVEL AADT COVERAGE  (LRSNUMBER x mile_bin x year)")
print("="*60)
print(f"  {seg_match.mean():.2%}  ({seg_match.sum():,} / {len(seg_match):,} segments)")

# ---------------------------------------------------------------------------
# Coverage: by route type (structural missingness analysis)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("COVERAGE BY ROUTE GROUP (RoadGroup field)")
print("="*60)
if "RoadGroup" in crashes.columns:
    rg = (
        crashes[["crash_id", "RoadGroup"]]
        .join(record_match, on="crash_id")
    )
    summary = (
        rg.groupby("RoadGroup")["aadt_match"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "coverage", "count": "n_records"})
        .sort_values("coverage")
    )
    summary["coverage"] = summary["coverage"].map("{:.2%}".format)
    print(summary.to_string())
else:
    print("  RoadGroup column not found — skipping.")

# ---------------------------------------------------------------------------
# Fan-out check: how many AADT rows does a single crash join to?
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("JOIN FAN-OUT CHECK (records per crash after merge)")
print("="*60)
fanout = joined.groupby("crash_id").size()
print(f"  Mean rows per crash:   {fanout.mean():.2f}")
print(f"  Max rows per crash:    {fanout.max()}")
print(f"  Crashes with >1 row:   {(fanout > 1).sum():,} ({(fanout > 1).mean():.2%})")

# ---------------------------------------------------------------------------
# Gap vs no-coverage breakdown
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("MISSINGNESS MECHANISM")
print("="*60)
route_matched = joined["AADT"].notna()
interval_hit  = joined["aadt_match"]
no_route      = ~route_matched
in_gap        = route_matched & ~interval_hit

n_crashes    = len(crashes)
no_route_n   = joined[joined["crash_id"].isin(
                   joined[no_route]["crash_id"])]["crash_id"].nunique()
in_gap_n     = joined[joined["crash_id"].isin(
                   joined[in_gap]["crash_id"])]["crash_id"].nunique()
matched_n    = record_match.sum()

print(f"  Matched (in interval):        {matched_n:>6,}  ({matched_n/n_crashes:.2%})")
print(f"  Route not in AADT at all:     {no_route_n:>6,}  ({no_route_n/n_crashes:.2%})")
print(f"  Route in AADT but gap in MP:  {in_gap_n:>6,}  ({in_gap_n/n_crashes:.2%})")

print("\nDone.")