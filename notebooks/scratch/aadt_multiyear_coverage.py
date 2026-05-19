"""
aadt_multiyear_coverage.py
--------------------------
Diagnostic: test whether pulling AADT from multiple years increases crash route coverage.

Key finding: union of 2020–2024 AADT layers adds only 9 routes over the 2024 snapshot
alone. Using the single 2024 snapshot is sufficient; multi-year pull provides negligible
coverage gain.

Date: 2026-05-16
Status: diagnostic only
"""

import requests
import pandas as pd

crashes = pd.read_parquet("data/splits/train.parquet")
crash_routes = set(crashes["LRSNUMBER"].dropna().unique())

layers = {2020: 71, 2021: 73, 2022: 74, 2023: 82, 2024: 84}
year_routes = {}

for year, layer_id in layers.items():
    all_codes = []
    offset = 0
    while True:
        url = f"https://maps.vtrans.vermont.gov/arcgis/rest/services/Master/General/FeatureServer/{layer_id}/query"
        params = {
            "where": "1=1",
            "outFields": "StandardRouteCode",
            "returnGeometry": "false",
            "f": "json",
            "resultOffset": offset,
            "resultRecordCount": 4000,
        }
        r = requests.get(url, params=params, timeout=60)
        features = r.json().get("features", [])
        if not features:
            break
        all_codes.extend([f["attributes"]["StandardRouteCode"] for f in features])
        if len(features) < 4000:
            break
        offset += 4000
    year_routes[year] = set(all_codes)
    overlap = crash_routes & year_routes[year]
    print(f"{year} (layer {layer_id}): {len(year_routes[year]):,} routes | "
          f"crash overlap: {len(overlap):,} ({len(overlap)/len(crash_routes):.2%})")

# Union across all years
union_routes = set()
for routes in year_routes.values():
    union_routes |= routes
union_overlap = crash_routes & union_routes
print(f"\nUnion all years: {len(union_routes):,} routes | "
      f"crash overlap: {len(union_overlap):,} ({len(union_overlap)/len(crash_routes):.2%})")

# What does union add over 2024 alone?
new_routes = union_overlap - (crash_routes & year_routes[2024])
print(f"Routes gained by union over 2024 alone: {len(new_routes):,}")