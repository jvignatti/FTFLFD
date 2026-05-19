"""
ete_lr_format_test.py
---------------------
Diagnostic: test overlap between crash LRSNUMBER values and ETE_LR from road centerline.

Key finding: ETE_LR values from road centerline do not match crash LRSNUMBER encoding.
Confirmed that the road centerline ETE_LR field cannot serve as a join key.

Date: 2026-05-16
Status: diagnostic only — see OD-002 in open_decisions.md
"""

import pandas as pd

centerline = pd.read_csv("data/raw/road_centerline.csv")

# Sample ETE_LR values
print("Sample ETE_LR values:")
print(centerline["ETE_LR"].head(20).values)

# Check overlap with crash LRSNUMBERs
crashes = pd.read_parquet("data/splits/train.parquet")
crash_routes = set(crashes["LRSNUMBER"].dropna().unique())
ete_lr_set   = set(centerline["ETE_LR"].dropna().unique())
twn_lr_set   = set(centerline["TWN_LR"].dropna().unique())

ete_overlap = crash_routes & ete_lr_set
twn_overlap = crash_routes & twn_lr_set

print(f"\nCrash LRSNUMBERs:        {len(crash_routes):,}")
print(f"ETE_LR overlap:          {len(ete_overlap):,} ({len(ete_overlap)/len(crash_routes):.2%})")
print(f"TWN_LR overlap:          {len(twn_overlap):,} ({len(twn_overlap)/len(crash_routes):.2%})")

print("\nSample ETE_LR not matching crash routes:")
print(list(ete_lr_set - crash_routes)[:10])
print("\nSample crash routes not in ETE_LR:")
print(list(crash_routes - ete_lr_set)[:10])
