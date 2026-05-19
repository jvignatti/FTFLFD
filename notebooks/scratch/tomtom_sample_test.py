"""
tomtom_sample_test.py
---------------------
Diagnostic: test TomTom Search API coverage on Vermont crash records.

Endpoint: /search/2/reverseGeocode
Returns: speedLimit, roadUse per coordinate

Date: 2026-05-18
Status: diagnostic only
API key: set via environment variable TOMTOM_API_KEY
"""

import requests
import pandas as pd
import os
import time

TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY")
if not TOMTOM_API_KEY:
    raise ValueError("TOMTOM_API_KEY environment variable not set")

df = pd.read_parquet(
    r"C:\Users\JC\Desktop\Personal\Models\FTFLFD\data\splits\train.parquet"
)

# Sample 100 records stratified by road group
sample = pd.concat([
    df[df["severity_class"] == "fatal"].head(30),
    df[df["severity_class"] == "injury"].head(30),
    df[df["severity_class"] == "pdo"].head(40),
]).drop_duplicates().head(100)

sample = sample[sample["has_valid_coords"] == True].copy()
print(f"Sample size: {len(sample)} records")

results = []

for i, (_, row) in enumerate(sample.iterrows()):
    lat = row["LATITUDE"]
    lon = row["LONGITUDE"]

    url = (
        f"https://api.tomtom.com/search/2/reverseGeocode"
        f"/{lat},{lon}.json"
        f"?key={TOMTOM_API_KEY}"
        f"&returnSpeedLimit=true"
        f"&returnRoadUse=true"
    )

    try:
        r = requests.get(url, timeout=10)
        data = r.json()

        addr = data.get("addresses", [{}])[0]
        address = addr.get("address", {})
        road_use = addr.get("roadUse", [None])

        speed_raw = address.get("speedLimit", None)
        speed_mph = None
        if speed_raw:
            try:
                speed_mph = float(speed_raw.replace("MPH", "").strip())
            except:
                speed_mph = None

        results.append({
            "objectid": row.get("OBJECTID"),
            "severity": row["severity_class"],
            "road_group": row["RoadGroup"],
            "lat": lat,
            "lon": lon,
            "speed_limit_mph": speed_mph,
            "road_use": road_use[0] if road_use else None,
            "status": r.status_code,
        })

    except Exception as e:
        results.append({
            "objectid": row.get("OBJECTID"),
            "severity": row["severity_class"],
            "road_group": row["RoadGroup"],
            "lat": lat,
            "lon": lon,
            "speed_limit_mph": None,
            "road_use": None,
            "status": str(e),
        })

    time.sleep(0.05)

    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/100...")

results_df = pd.DataFrame(results)

print("\n=== COVERAGE SUMMARY ===")
print(f"Speed limit non-null: {results_df['speed_limit_mph'].notna().sum()}/100 "
      f"({results_df['speed_limit_mph'].notna().mean():.2%})")
print(f"Road use non-null:    {results_df['road_use'].notna().sum()}/100 "
      f"({results_df['road_use'].notna().mean():.2%})")

print("\n=== COVERAGE BY ROAD GROUP ===")
for rg, grp in results_df.groupby("road_group"):
    cov = grp["speed_limit_mph"].notna().mean()
    print(f"  {rg[:60]}: {cov:.0%} ({grp['speed_limit_mph'].notna().sum()}/{len(grp)})")

print("\n=== SPEED LIMIT DISTRIBUTION ===")
print(results_df["speed_limit_mph"].value_counts().sort_index())

print("\n=== ROAD USE VALUES ===")
print(results_df["road_use"].value_counts())

print("\n=== COVERAGE BY SEVERITY ===")
for sev, grp in results_df.groupby("severity"):
    cov = grp["speed_limit_mph"].notna().mean()
    print(f"  {sev}: {cov:.2%}")

print("\n=== HTTP STATUS CODES ===")
print(results_df["status"].value_counts())