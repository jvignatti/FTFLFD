"""
aran_overlap.py
---------------
Diagnostic: test whether ARAN roadway data (ETE_LR, TOWN_LR) provides a viable join
to crash records.

Key finding: ARAN ETE_LR covers only 34.87% of crash routes, limited to the
interstate/highway system. ARAN is not a viable source for broad crash-to-road feature
joins.

Date: 2026-05-16
Status: diagnostic only
"""

import requests
import pandas as pd

crashes = pd.read_parquet("data/splits/train.parquet")
crash_routes = set(crashes["LRSNUMBER"].dropna().unique())

all_features = []
offset = 0
while True:
    url = "https://maps.vtrans.vermont.gov/arcgis/rest/services/AMP/ARAN_Roadway_Data/FeatureServer/0/query"
    params = {
        "where": "1=1",
        "outFields": "ETE_LR,TOWN_LR",
        "returnGeometry": "false",
        "f": "json",
        "resultOffset": offset,
        "resultRecordCount": 1000,
    }
    r = requests.get(url, params=params, timeout=60)
    features = r.json().get("features", [])
    if not features:
        break
    all_features.extend([f["attributes"] for f in features])
    if len(features) < 1000:
        break
    offset += 1000
    if offset % 10000 == 0:
        print(f"  Fetched {offset:,}...")

df = pd.DataFrame(all_features)
print(f"Total ARAN records: {len(df):,}")

ete_routes  = set(df["ETE_LR"].dropna().unique())
town_routes = set(df["TOWN_LR"].dropna().unique())

ete_overlap  = crash_routes & ete_routes
town_overlap = crash_routes & town_routes
combined     = crash_routes & (ete_routes | town_routes)

print(f"Crash unique LRSNUMBERs:     {len(crash_routes):,}")
print(f"ARAN unique ETE_LR:          {len(ete_routes):,}")
print(f"ARAN unique TOWN_LR:         {len(town_routes):,}")
print(f"ETE_LR overlap:              {len(ete_overlap):,} ({len(ete_overlap)/len(crash_routes):.2%})")
print(f"TOWN_LR overlap:             {len(town_overlap):,} ({len(town_overlap)/len(crash_routes):.2%})")
print(f"Combined overlap:            {len(combined):,} ({len(combined)/len(crash_routes):.2%})")

print("\nSample ETE_LR values:")
print(sorted(list(ete_routes))[:15])
print("\nSample TOWN_LR values:")
print(sorted(list(town_routes))[:15])