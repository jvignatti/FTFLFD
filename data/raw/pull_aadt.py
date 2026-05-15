"""
pull_aadt.py — Pulls Vermont AADT data from VTrans ArcGIS REST API.

Source: https://maps.vtrans.vermont.gov/arcgis/rest/services/Layers/AADT/FeatureServer
Two layers:
  Layer 0: AADT Limited (state highways)
  Layer 1: AADT Other (other roads)

MaxRecordCount is 5000, so we paginate with resultOffset.
Output: data/raw/aadt_limited.csv and data/raw/aadt_other.csv
"""

import requests
import pandas as pd
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_URL = "https://maps.vtrans.vermont.gov/arcgis/rest/services/Layers/AADT/FeatureServer"
MAX_RECORDS = 5000

LAYERS = {
    "aadt_limited": 0,
    "aadt_other": 1,
}


def pull_layer(layer_id, layer_name):
    """Pull all records from a layer using pagination."""
    all_features = []
    offset = 0

    print(f"\nPulling {layer_name} (layer {layer_id})...")

    while True:
        url = f"{BASE_URL}/{layer_id}/query"
        params = {
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "false",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": MAX_RECORDS,
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        features = data.get("features", [])
        if not features:
            break

        for f in features:
            all_features.append(f["attributes"])

        print(f"  Fetched {len(features)} records (total: {len(all_features)})")

        if len(features) < MAX_RECORDS:
            break

        offset += MAX_RECORDS

    print(f"  Total records: {len(all_features)}")
    return all_features


def main():
    print("=" * 60)
    print("AADT DATA PULL — Vermont VTrans")
    print("Source: ArcGIS REST FeatureServer (public, no auth)")
    print("=" * 60)

    for name, layer_id in LAYERS.items():
        features = pull_layer(layer_id, name)

        if not features:
            print(f"  WARNING: No features returned for {name}")
            continue

        df = pd.DataFrame(features)
        output_path = SCRIPT_DIR / f"{name}.csv"
        df.to_csv(output_path, index=False)
        print(f"  Saved: {output_path.name} ({len(df):,} rows, {len(df.columns)} columns)")
        print(f"  Columns: {list(df.columns)}")

    print("\n" + "=" * 60)
    print("AADT pull complete.")
    print("Run hash_data.py to register these files before use.")
    print("=" * 60)


if __name__ == "__main__":
    main()