"""
schema_check.py
---------------
Diagnostic: inspect crash and AADT file schemas, field names, and route ID overlap.

Purpose
-------
Establishes ground truth on available fields, data types, and join key compatibility
between crash records and AADT survey data. Output from this script directly informed
the feature engineering strategy documented in docs/technical/07_aadt_coverage_diagnosis.md.

Run from project root:
    python notebooks/diagnostic/schema_check.py

Outputs: printed to stdout. Redirect to file if permanent record is needed:
    python notebooks/diagnostic/schema_check.py > notebooks/diagnostic/output/schema_check_output.txt
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths — relative to project root (FTFLFD/)
# ---------------------------------------------------------------------------
TRAIN_PARQUET     = "data/splits/train.parquet"
AADT_LIMITED_CSV  = "data/raw/aadt_limited.csv"
AADT_OTHER_CSV    = "data/raw/aadt_other.csv"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading data...")
crashes      = pd.read_parquet(TRAIN_PARQUET)
aadt_limited = pd.read_csv(AADT_LIMITED_CSV)
aadt_other   = pd.read_csv(AADT_OTHER_CSV)

# ---------------------------------------------------------------------------
# Crash schema
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("CRASHES SCHEMA")
print("="*60)
print(crashes.dtypes.to_string())
print(f"\nShape: {crashes.shape}")

print("\n" + "="*60)
print("CRASHES SAMPLE (3 rows)")
print("="*60)
print(crashes.head(3).to_string())

# ---------------------------------------------------------------------------
# AADT schemas
# ---------------------------------------------------------------------------
for name, df in [("AADT LIMITED", aadt_limited), ("AADT OTHER", aadt_other)]:
    print("\n" + "="*60)
    print(f"{name} SCHEMA")
    print("="*60)
    print(df.dtypes.to_string())
    print(f"\nShape: {df.shape}")
    print(f"\n{name} SAMPLE (3 rows)")
    print(df.head(3).to_string())

# ---------------------------------------------------------------------------
# Temporal coverage of AADT
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("AADT YEAR COVERAGE")
print("="*60)
for name, df in [("aadt_limited", aadt_limited), ("aadt_other", aadt_other)]:
    if "Year" in df.columns:
        print(f"{name}: {sorted(df['Year'].unique())}")
    else:
        print(f"{name}: No 'Year' column found")

# ---------------------------------------------------------------------------
# Severity field identification
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("CRASH SEVERITY CANDIDATES")
print("="*60)
sev_cols = [c for c in crashes.columns
            if any(x in c.upper() for x in ["SEVER", "FATAL", "INJUR", "CRASH_TYPE"])]
print(f"Candidate columns: {sev_cols}")
for c in sev_cols:
    print(f"\n{c}:")
    print(crashes[c].value_counts().to_string())

# ---------------------------------------------------------------------------
# Route ID overlap — all candidate columns
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("ROUTE ID OVERLAP CHECK (crash LRSNUMBER vs AADT route columns)")
print("="*60)
crash_routes = set(crashes["LRSNUMBER"].dropna().unique())
print(f"Crash unique LRSNUMBERs: {len(crash_routes)}")

for name, df in [("aadt_limited", aadt_limited), ("aadt_other", aadt_other)]:
    route_cols = [c for c in df.columns
                  if any(x in c.upper() for x in ["ROUTE", "LRS", "STANDARD"])]
    for col in route_cols:
        aadt_routes = set(df[col].dropna().unique())
        overlap = crash_routes & aadt_routes
        pct = len(overlap) / len(crash_routes) if crash_routes else 0
        print(f"  {name}.{col}: {len(aadt_routes)} unique | "
              f"{len(overlap)} overlap with crashes ({pct:.2%})")

print("\nDone.")