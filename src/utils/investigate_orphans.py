"""
Investigates rows in vt_crashes_all.csv whose ACCIDENTDATE cannot be parsed
as Unix milliseconds. These rows are excluded from all yearly splits.

Writes the full orphaned set to data/raw/orphaned_rows.csv for inspection.

Usage:
    python -m src.utils.investigate_orphans
"""

import pandas as pd
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
INPUT_FILE = RAW_DIR / "vt_crashes_all.csv"
OUTPUT_FILE = RAW_DIR / "orphaned_rows.csv"

SAMPLE_COLS = [
    "OBJECTID",
    "REPORTNUMBER",
    "ACCIDENTDATE",
    "CITYORTOWN",
    "InjuryType",
    "LATITUDE",
    "LONGITUDE",
]


def main() -> None:
    if not INPUT_FILE.exists():
        print(f"ERROR: input file not found: {INPUT_FILE}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {INPUT_FILE.name} ...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    dates = pd.to_datetime(df["ACCIDENTDATE"], unit="ms", errors="coerce")
    orphans = df[dates.isna()].copy()

    print(f"\n=== Total orphaned rows: {len(orphans):,} of {len(df):,} ===\n")

    # Distribution by ReportingAgency
    print("=== By ReportingAgency (top 10) ===")
    agency_counts = orphans["ReportingAgency"].value_counts(dropna=False).head(10)
    print(agency_counts.to_string())
    print()

    # Distribution by InjuryType
    print("=== By InjuryType ===")
    injury_counts = orphans["InjuryType"].value_counts(dropna=False)
    print(injury_counts.to_string())
    print()

    # Raw ACCIDENTDATE value analysis
    print("=== Raw ACCIDENTDATE values ===")
    null_count = orphans["ACCIDENTDATE"].isna().sum()
    non_null_count = orphans["ACCIDENTDATE"].notna().sum()
    print(f"  Null / NaN      : {null_count:,}")
    print(f"  Non-null (other): {non_null_count:,}")
    if non_null_count > 0:
        print("  Sample non-null raw values:")
        for val in orphans["ACCIDENTDATE"].dropna().head(20):
            print(f"    {repr(val)}")
    print()

    # Sample rows
    print("=== Sample of 10 orphaned rows ===")
    available = [c for c in SAMPLE_COLS if c in orphans.columns]
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(orphans[available].head(10).to_string(index=False))
    print()

    # Write full orphaned set
    orphans.to_csv(OUTPUT_FILE, index=False)
    print(f"Full orphaned rows written to: {OUTPUT_FILE}")
    print(f"  ({len(orphans):,} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    main()
