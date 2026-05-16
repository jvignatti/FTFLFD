"""
pull_road_centerline.py — Pulls Vermont Road Centerline data from VTrans ArcGIS REST API.

Source: https://maps.vtrans.vermont.gov/arcgis/rest/services/Master/General/FeatureServer/39
Layer: VT Road Centerline (layer 39)

Purpose: Provides FunctionalClass (FUNCL) and road geometry attributes for all
public roads in Vermont. Used for Gate 1 feature availability assessment and
as imputation key for AADT coverage gaps.

LRS join key: TWN_LR -> crash LRSNUMBER
Alternate:    ETE_LR -> crash LRSNUMBER (fallback)

Fields pulled (subset — excludes geometry, audit, and redundant fields):
    OBJECTID, SEGMENTID, TWN_LR, ETE_LR, CTCODE,
    FUNCL, RPCCLASS, AOTCLASS, NHS, UA,
    ARCMILES, AOTMILES, Facility_Type, ONEWAY,
    SURFACETYPE, RDFLNAME, SPEEDLIMIT

Credit: LRS integration methodology — Owen Mosley.
Source data: Vermont Open Geodata Portal (public, no authentication required).
    https://geodata.vermont.gov/datasets/VTrans::vt-road-centerline/about

Output: data/raw/road_centerline.csv

Run from project root:
    python data/raw/pull_road_centerline.py
"""

import requests
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

BASE_URL = (
    "https://maps.vtrans.vermont.gov/arcgis/rest/services/"
    "Master/General/FeatureServer/39"
)

MAX_RECORDS = 1000  # conservative — large geometry layer, avoid timeouts

OUT_FIELDS = ",".join([
    "OBJECTID", "SEGMENTID",
    "TWN_LR", "ETE_LR", "CTCODE",
    "FUNCL", "RPCCLASS", "AOTCLASS",
    "NHS", "UA",
    "ARCMILES", "AOTMILES",
    "Facility_Type", "ONEWAY",
    "SURFACETYPE", "RDFLNAME",
    "SPEEDLIMIT",
])


def pull_centerline():
    """Pull all road centerline records using pagination."""
    all_features = []
    offset = 0

    print("Pulling VT Road Centerline (layer 39)...")
    print(f"Endpoint: {BASE_URL}")
    print(f"Page size: {MAX_RECORDS} records")
    print("-" * 60)

    while True:
        url = f"{BASE_URL}/query"
        params = {
            "where": "1=1",
            "outFields": OUT_FIELDS,
            "returnGeometry": "false",
            "outSR": "4326",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": MAX_RECORDS,
        }

        try:
            response = requests.get(url, params=params, timeout=60)
            response.raise_for_status()
        except requests.exceptions.Timeout:
            print(f"  TIMEOUT at offset {offset} — retrying with smaller page...")
            MAX_RECORDS_RETRY = 500
            params["resultRecordCount"] = MAX_RECORDS_RETRY
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()

        data = response.json()

        if "error" in data:
            print(f"  API ERROR: {data['error']}")
            break

        features = data.get("features", [])
        if not features:
            break

        for f in features:
            all_features.append(f["attributes"])

        print(f"  Offset {offset:>7,}: fetched {len(features):>5} | "
              f"total so far: {len(all_features):>7,}")

        if len(features) < MAX_RECORDS:
            break

        offset += MAX_RECORDS

    return all_features


def main():
    print("=" * 60)
    print("VT ROAD CENTERLINE PULL — Vermont VTrans")
    print("Source: ArcGIS REST FeatureServer (public, no auth)")
    print("Credit: LRS methodology — Owen Mosley")
    print("=" * 60)

    features = pull_centerline()

    if not features:
        print("ERROR: No features returned. Check endpoint and network.")
        return

    df = pd.DataFrame(features)

    print("\n" + "=" * 60)
    print(f"Total records pulled: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    # Quick diagnostics
    print(f"\nTWN_LR non-null:     {df['TWN_LR'].notna().sum():,} "
          f"({df['TWN_LR'].notna().mean():.2%})")
    print(f"ETE_LR non-null:     {df['ETE_LR'].notna().sum():,} "
          f"({df['ETE_LR'].notna().mean():.2%})")
    print(f"FUNCL non-null:      {df['FUNCL'].notna().sum():,} "
          f"({df['FUNCL'].notna().mean():.2%})")
    print(f"\nFUNCL value counts:")
    print(df["FUNCL"].value_counts().sort_index().to_string())
    print(f"\nFacility_Type value counts:")
    print(df["Facility_Type"].value_counts().sort_index().to_string())

    # Save
    output_path = SCRIPT_DIR / "road_centerline.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path.name} ({len(df):,} rows, {len(df.columns)} columns)")
    print("\nRun hash_data.py to register this file before use.")
    print("=" * 60)


if __name__ == "__main__":
    main()