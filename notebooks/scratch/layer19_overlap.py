"""
layer19_overlap.py
------------------
Diagnostic: test overlap between VTrans General FeatureServer Layer 19 (ETE_LR, TWN_LR,
FUNCL) and crash route identifiers.

Key finding: TWN_LR from Layer 19 achieves 68.95% route overlap with crash LRSNUMBER.
Neither ETE_LR nor TWN_LR can be directly joined without a crosswalk due to format
differences.

Date: 2026-05-16
Status: diagnostic only — see crosswalkFUNCL.py for format classification
"""

import requests, pandas as pd

crashes = pd.read_parquet("data/splits/train.parquet")
crash_routes = set(crashes["LRSNUMBER"].dropna().unique())

# Pull all ETE_LR and TWN_LR from Layer 19
all_features = []
offset = 0
while True:
    url = "https://maps.vtrans.vermont.gov/arcgis/rest/services/Master/General/FeatureServer/19/query"
    params = {
        "where": "1=1",
        "outFields": "ETE_LR,TWN_LR,ETE_FMM,ETE_TMM,TWN_FMM,TWN_TMM,FUNCL,Facility_Type",
        "returnGeometry": "false",
        "f": "json",
        "resultOffset": offset,
        "resultRecordCount": 4000,
    }
    r = requests.get(url, params=params, timeout=60)
    features = r.json().get("features", [])
    if not features:
        break
    all_features.extend([f["attributes"] for f in features])
    print(f"  Fetched {len(features)} (total: {len(all_features)})")
    if len(features) < 4000:
        break
    offset += 4000

df = pd.DataFrame(all_features)
print(f"\nTotal records: {len(df):,}")

# Check overlaps
ete_overlap = crash_routes & set(df["ETE_LR"].dropna().unique())
twn_overlap = crash_routes & set(df["TWN_LR"].dropna().unique())
print(f"ETE_LR overlap: {len(ete_overlap):,} ({len(ete_overlap)/len(crash_routes):.2%})")
print(f"TWN_LR overlap: {len(twn_overlap):,} ({len(twn_overlap)/len(crash_routes):.2%})")

print("\nSample TWN_LR values:")
print(df["TWN_LR"].dropna().head(20).values)