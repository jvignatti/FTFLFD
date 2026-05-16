"""
functional_class_coverage.py
-----------------------------
Diagnostic: compute FunctionalClass (FUNCL) join coverage at record and severity levels.

Purpose
-------
Assesses Gate 1 availability for the functional_class feature candidate using the
VTrans Road Centerline layer (public, open source) joined via TWN_LR -> LRSNUMBER.

This is the first feature in the AADT gate testing sequence. FunctionalClass is
tested before segment_aadt because:
  1. It has no imputation dependency
  2. The road centerline layer provides 100% FUNCL coverage across all 78,876 segments
  3. If it passes Gate 1, it becomes the imputation key for AADT in a subsequent iteration

Data sources:
  Crashes:     data/splits/train.parquet (Phase 1, 2010-2019, 109,695 records)
  Centerline:  data/raw/road_centerline.csv (VTrans, public)
               https://geodata.vermont.gov/datasets/VTrans::vt-road-centerline/about

LRS integration methodology: Owen Mosley.
See docs/technical/07_aadt_coverage_diagnosis.md for full context.

Run from project root:
    python notebooks/diagnostic/functional_class_coverage.py

Output:
    python notebooks/diagnostic/functional_class_coverage.py > notebooks/diagnostic/output/functional_class_coverage_output.txt
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Paths — relative to project root (FTFLFD/)
# ---------------------------------------------------------------------------
TRAIN_PARQUET  = "data/splits/train.parquet"
CENTERLINE_CSV = "data/raw/road_centerline.csv"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading data...")
crashes    = pd.read_parquet(TRAIN_PARQUET)
centerline = pd.read_csv(CENTERLINE_CSV)

print(f"Crashes loaded:        {len(crashes):,} records")
print(f"Centerline loaded:     {len(centerline):,} segments")

# ---------------------------------------------------------------------------
# Route-level overlap
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("ROUTE-LEVEL OVERLAP")
print("="*60)

crash_routes      = set(crashes["LRSNUMBER"].dropna().unique())
centerline_routes = set(centerline["TWN_LR"].dropna().unique())
ete_routes        = set(centerline["ETE_LR"].dropna().unique())

direct   = crash_routes & centerline_routes
ete      = crash_routes & ete_routes
combined = crash_routes & (centerline_routes | ete_routes)

print(f"Crash unique LRSNUMBERs:       {len(crash_routes):,}")
print(f"Centerline unique TWN_LRs:     {len(centerline_routes):,}")
print(f"Direct overlap (TWN_LR):       {len(direct):,} ({len(direct)/len(crash_routes):.2%})")
print(f"ETE_LR fallback overlap:       {len(ete):,} ({len(ete)/len(crash_routes):.2%})")
print(f"Combined (TWN_LR + ETE_LR):    {len(combined):,} ({len(combined)/len(crash_routes):.2%})")

# ---------------------------------------------------------------------------
# Record-level coverage — TWN_LR join
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("RECORD-LEVEL FUNCL COVERAGE (TWN_LR join)")
print("="*60)

joined = crashes.merge(
    centerline[["TWN_LR", "FUNCL", "Facility_Type"]],
    left_on="LRSNUMBER",
    right_on="TWN_LR",
    how="left"
)

record_cov = joined["FUNCL"].notna().mean()
print(f"All records:  {record_cov:.2%}  ({joined['FUNCL'].notna().sum():,} / {len(crashes):,})")

print("\nBy severity:")
for sev in ["fatal", "injury", "pdo"]:
    sub = joined[joined["severity_class"] == sev]
    cov = sub["FUNCL"].notna().mean()
    n   = sub["FUNCL"].notna().sum()
    tot = len(sub)
    print(f"  {sev:<8}: {cov:.2%}  ({n:,} / {tot:,})")

# ---------------------------------------------------------------------------
# Fan-out check
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("JOIN FAN-OUT CHECK")
print("="*60)
fanout = joined.groupby(joined.index).size()
print(f"  Mean rows per crash:   {fanout.mean():.2f}")
print(f"  Max rows per crash:    {fanout.max()}")
print(f"  Crashes with >1 row:   {(fanout > 1).sum():,} ({(fanout > 1).mean():.2%})")

# ---------------------------------------------------------------------------
# FUNCL distribution on matched crashes
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("FUNCL DISTRIBUTION ON MATCHED CRASH RECORDS")
print("="*60)
funcl_map = {
    0: "Unclassified/Local",
    1: "Interstate",
    2: "Other Freeway/Expressway",
    3: "Other Principal Arterial",
    4: "Minor Arterial",
    5: "Major Collector",
    6: "Minor Collector",
    7: "Local",
}
funcl_counts = joined["FUNCL"].value_counts().sort_index()
for code, count in funcl_counts.items():
    label = funcl_map.get(int(code), "Unknown")
    pct   = count / len(joined)
    print(f"  FUNCL {int(code)} ({label:<30}): {count:>7,}  ({pct:.2%})")

# ---------------------------------------------------------------------------
# FUNCL distribution for fatal crashes only
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("FUNCL DISTRIBUTION — FATAL CRASHES ONLY")
print("="*60)
fatal_funcl = joined[joined["severity_class"] == "fatal"]["FUNCL"].value_counts().sort_index()
for code, count in fatal_funcl.items():
    label = funcl_map.get(int(code), "Unknown")
    pct   = count / fatal_funcl.sum()
    print(f"  FUNCL {int(code)} ({label:<30}): {count:>5,}  ({pct:.2%})")

# ---------------------------------------------------------------------------
# Facility_Type encoding check
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("FACILITY_TYPE VALUE COUNTS (divided/undivided candidate)")
print("="*60)
print(joined["Facility_Type"].value_counts().sort_index().to_string())

# ---------------------------------------------------------------------------
# Gate 1 summary
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("GATE 1 SUMMARY")
print("="*60)
print(f"  Threshold:            95.00%")
print(f"  All records:          {record_cov:.2%}  {'PASS' if record_cov >= 0.95 else 'FAIL'}")
fatal_cov = joined[joined["severity_class"] == "fatal"]["FUNCL"].notna().mean()
print(f"  Fatal records:        {fatal_cov:.2%}  {'PASS' if fatal_cov >= 0.95 else 'FAIL'}")

print("\nDone.")