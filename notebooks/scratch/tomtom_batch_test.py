"""
tomtom_batch_test.py
--------------------
Test whether TomTom batch sync endpoint is available on current API key.
Date: 2026-05-18
Status: diagnostic only
"""

import requests
import os

TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY")
if not TOMTOM_API_KEY:
    raise ValueError("TOMTOM_API_KEY not set")

url = f"https://api.tomtom.com/search/2/batch/sync.json?key={TOMTOM_API_KEY}"

payload = {
    "batchItems": [
        {"query": "/reverseGeocode/44.33401623,-73.21860647.json?returnSpeedLimit=true&returnRoadUse=true"},
        {"query": "/reverseGeocode/44.48676300,-73.18799300.json?returnSpeedLimit=true&returnRoadUse=true"},
        {"query": "/reverseGeocode/43.11759500,-73.11036800.json?returnSpeedLimit=true&returnRoadUse=true"},
    ]
}

r = requests.post(url, json=payload, timeout=30)
print(f"Status: {r.status_code}")
print(f"Response: {r.text[:1000]}")