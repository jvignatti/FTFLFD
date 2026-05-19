import requests
import os
import pandas as pd

TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY")

df = pd.read_parquet(
    r"C:\Users\JC\Desktop\Personal\Models\FTFLFD\data\splits\train.parquet"
)

row = df[(df["severity_class"] == "fatal") & (df["has_valid_coords"] == True)].iloc[0]
lat = row["LATITUDE"]
lon = row["LONGITUDE"]
print(f"Coords: {lat}, {lon}")

# Correct TomTom reverse geocode endpoint
url = (
    f"https://api.tomtom.com/search/2/reverseGeocode"
    f"/{lat},{lon}.json"
    f"?key={TOMTOM_API_KEY}"
    f"&returnSpeedLimit=true"
    f"&returnRoadUse=true"
)

print(f"Testing reverse geocode...")
r = requests.get(url, timeout=10)
print(f"Status: {r.status_code}")
print(f"Response: {r.text[:1000]}")