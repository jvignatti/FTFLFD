"""
loader.py — Loads and validates Vermont crash data from yearly CSV files.

Reads from: data/raw/yearly/vt_crashes_YYYY.csv
Outputs: cleaned DataFrame with validated schema, parsed dates,
         target labels, data_era flag, and hybrid segment assignment.

Hybrid segmentation:
  - Primary: LRS segments (LRSNUMBER, floor(milepoint)) for State System roads
  - Secondary: 1km grid cells (lat/lon) for crashes without LRS data
  - See docs/technical/03_prediction_unit.md v1.1

This module performs Layer 1 (ingestion + cleaning) of the pipeline.
No feature engineering happens here.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
YEARLY_DIR = RAW_DIR / "yearly"

REQUIRED_COLUMNS = [
    "OBJECTID", "ACCIDENTDATE", "InjuryType",
    "LRSNUMBER", "AOTACTUALMILEPOINT",
    "CITYORTOWN", "latitude", "longitude",
    "Weather", "SurfaceCondition", "RoadCondition",
    "DayNight", "DirOfCollision", "Animal",
    "Impairment", "Involving", "RoadCharacteristics",
    "RoadGroup", "ReportingAgency"
]

MILEPOINT_SENTINEL = 999.99

# Grid parameters (Vermont bounding box)
GRID_ORIGIN_LAT = 42.7
GRID_ORIGIN_LON = -73.5
GRID_CELL_LAT = 0.009   # ~1km at Vermont latitude
GRID_CELL_LON = 0.012   # ~1km at Vermont latitude


def load_year(year: int) -> pd.DataFrame:
    """Load a single yearly CSV file and apply base cleaning."""
    filepath = YEARLY_DIR / f"vt_crashes_{year}.csv"
    if not filepath.exists():
        raise FileNotFoundError(f"Yearly file not found: {filepath}")

    df = pd.read_csv(filepath, low_memory=False)

    # Validate required columns exist
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {filepath.name}: {missing}")

    # Parse ACCIDENTDATE from Unix milliseconds
    df["event_date"] = pd.to_datetime(
        df["ACCIDENTDATE"], unit="ms", errors="coerce"
    )

    # Drop rows with unparseable dates (ghost rows)
    before = len(df)
    df = df.dropna(subset=["event_date"])
    dropped = before - len(df)
    if dropped > 0:
        print(f"  [{year}] Dropped {dropped} rows with null dates")

    return df


def assign_target(df: pd.DataFrame) -> pd.DataFrame:
    """Derive binary target from InjuryType. Blanks are excluded."""
    df = df.copy()

    # Standardize whitespace
    df["InjuryType"] = df["InjuryType"].astype(str).str.strip()

    # Map to target
    target_map = {
        "Fatal": 1,
        "Injury": 1,
        "Property Damage Only": 0
    }
    df["target"] = df["InjuryType"].map(target_map)

    # Preserve severity detail for separate recall reporting
    severity_map = {
        "Fatal": "fatal",
        "Injury": "injury",
        "Property Damage Only": "pdo"
    }
    df["severity_class"] = df["InjuryType"].map(severity_map)

    # Drop rows with unmapped InjuryType (blank/unknown/nan)
    before = len(df)
    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)
    dropped = before - len(df)
    if dropped > 0:
        print(f"  Dropped {dropped} rows with blank/unknown InjuryType")

    return df


def assign_data_era(df: pd.DataFrame) -> pd.DataFrame:
    """Assign data_era flag based on year."""
    df = df.copy()
    year = df["event_date"].dt.year
    df["data_era"] = np.select(
        [year <= 2014, year <= 2019, year >= 2020],
        ["early", "historical", "modern"],
        default="unknown"
    )
    return df


def assign_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Assign hybrid segment ID: LRS primary, grid secondary."""
    df = df.copy()

    # --- LRS SEGMENTATION (PRIMARY) ---

    # Null out sentinel milepoints
    df.loc[
        df["AOTACTUALMILEPOINT"] == MILEPOINT_SENTINEL,
        "AOTACTUALMILEPOINT"
    ] = np.nan

    # Check LRS availability
    has_lrs = (
        df["LRSNUMBER"].notna() &
        df["AOTACTUALMILEPOINT"].notna()
    )

    # Compute LRS segment
    df["segment_milepoint"] = np.nan
    df.loc[has_lrs, "segment_milepoint"] = np.floor(
        df.loc[has_lrs, "AOTACTUALMILEPOINT"]
    ).astype(int)

    # --- GRID SEGMENTATION (SECONDARY) ---

    # Check coordinate availability (Vermont bounding box)
    has_coords = (
        df["latitude"].notna() &
        df["longitude"].notna() &
        df["latitude"].between(42.7, 45.1) &
        df["longitude"].between(-73.5, -71.4)
    )

    # Compute grid cell for all rows with valid coords
    df["grid_row"] = np.nan
    df["grid_col"] = np.nan
    df.loc[has_coords, "grid_row"] = np.floor(
        (df.loc[has_coords, "latitude"] - GRID_ORIGIN_LAT) / GRID_CELL_LAT
    ).astype(int)
    df.loc[has_coords, "grid_col"] = np.floor(
        (df.loc[has_coords, "longitude"] - GRID_ORIGIN_LON) / GRID_CELL_LON
    ).astype(int)

    # --- HYBRID ASSIGNMENT ---

    # Priority: LRS first, then grid, then none
    conditions = [
        has_lrs,
        ~has_lrs & has_coords,
    ]
    segment_ids = [
        "LRS_" + df["LRSNUMBER"].astype(str) + "_" + df["segment_milepoint"].astype(str),
        "GRID_" + df["grid_row"].astype(str) + "_" + df["grid_col"].astype(str),
    ]
    df["segment_id"] = np.select(conditions, segment_ids, default=None)

    # Segment type column
    segment_types = ["lrs", "grid"]
    df["segment_type"] = np.select(conditions, segment_types, default="none")

    # Replace string "None" with actual None
    df.loc[df["segment_type"] == "none", "segment_id"] = None

    # Mark coordinate validity
    df["has_valid_coords"] = has_coords

    # --- REPORTING ---

    lrs_count = has_lrs.sum()
    grid_only = (~has_lrs & has_coords).sum()
    no_spatial = (~has_lrs & ~has_coords).sum()

    print(f"  LRS segments assigned:  {lrs_count:,}")
    print(f"  Grid cells assigned:    {grid_only:,}")
    print(f"  No spatial data:        {no_spatial:,}")

    # Fatal coverage report
    fatal = df["severity_class"] == "fatal"
    fatal_lrs = (fatal & has_lrs).sum()
    fatal_grid = (fatal & ~has_lrs & has_coords).sum()
    fatal_none = (fatal & ~has_lrs & ~has_coords).sum()
    print(f"  Fatal coverage: LRS={fatal_lrs}, Grid={fatal_grid}, Lost={fatal_none} of {fatal.sum()}")

    return df


def assign_town(df: pd.DataFrame) -> pd.DataFrame:
    """Map CITYORTOWN. Mark missing as unknown."""
    df = df.copy()
    df["town"] = df["CITYORTOWN"].fillna("unknown").str.strip()
    unknown_count = (df["town"] == "unknown").sum()
    if unknown_count > 0:
        print(f"  {unknown_count} rows with unknown town")
    return df


def load_all_years(
    start_year: int = 2010,
    end_year: int = 2026
) -> pd.DataFrame:
    """Load all yearly files and apply full ingestion pipeline."""
    frames = []
    for year in range(start_year, end_year + 1):
        print(f"Loading {year}...")
        try:
            df = load_year(year)
            frames.append(df)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")

    if not frames:
        raise RuntimeError("No yearly files loaded")

    print(f"\nConcatenating {len(frames)} years...")
    df = pd.concat(frames, ignore_index=True)
    print(f"  Total rows after concat: {len(df):,}")
    df = df.sort_values("event_date").reset_index(drop=True)

    # Apply pipeline steps in order
    print("\nAssigning targets...")
    df = assign_target(df)
    print(f"  Rows after target assignment: {len(df):,}")

    print("\nAssigning data era...")
    df = assign_data_era(df)

    print("\nAssigning hybrid segments...")
    df = assign_segment(df)

    print("\nAssigning towns...")
    df = assign_town(df)

    # Summary
    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Total rows loaded:       {len(df):,}")
    print(f"Fatal:                   {(df['severity_class'] == 'fatal').sum():,}")
    print(f"Injury:                  {(df['severity_class'] == 'injury').sum():,}")
    print(f"PDO:                     {(df['severity_class'] == 'pdo').sum():,}")
    print(f"With LRS segment:        {(df['segment_type'] == 'lrs').sum():,}")
    print(f"With grid cell:          {(df['segment_type'] == 'grid').sum():,}")
    print(f"No spatial assignment:   {(df['segment_type'] == 'none').sum():,}")
    print(f"With valid coords:       {df['has_valid_coords'].sum():,}")
    print(f"Data era distribution:")
    print(df["data_era"].value_counts().to_string(header=False))
    print("=" * 60)

    return df


if __name__ == "__main__":
    df = load_all_years()
    output_path = REPO_ROOT / "data" / "processed" / "vt_crashes_ingested.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")