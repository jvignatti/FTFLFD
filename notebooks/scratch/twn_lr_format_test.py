"""
twn_lr_format_test.py
---------------------
Diagnostic: test overlap between segment_id-extracted LRSNUMBERs and TWN_LR from road
centerline using Phase 1 train_featured windows.

Key finding: extracted LRSNUMBER values from windowed segment_ids show no overlap with
TWN_LR values. Mismatch confirmed — road centerline TWN_LR join is invalid.

Date: 2026-05-16
Status: diagnostic only — see OD-002 in open_decisions.md
"""

import pandas as pd
centerline = pd.read_csv("data/raw/road_centerline.csv")
train = pd.read_parquet("data/splits/train_featured.parquet")

# Sample LRS segment_ids from windows
lrs_windows = train[train["segment_id"].str.startswith("LRS_")]["segment_id"].head(20)
print("Sample segment_ids:")
print(lrs_windows.values)

# Extract LRSNUMBERs
def extract_lrsnumber(segment_id):
    if segment_id.startswith("LRS_"):
        parts = segment_id[4:].rsplit("_", 1)
        return parts[0] if parts else None
    return None

extracted = lrs_windows.apply(extract_lrsnumber)
print("\nExtracted LRSNUMBERs:")
print(extracted.values)

# Sample TWN_LR values
print("\nSample TWN_LR values from centerline:")
print(centerline["TWN_LR"].head(20).values)

# Check overlap
extracted_set = set(extracted.dropna())
twn_lr_set = set(centerline["TWN_LR"].dropna())
overlap = extracted_set & twn_lr_set
print(f"\nExtracted sample: {len(extracted_set)} unique")
print(f"TWN_LR total: {len(twn_lr_set)} unique")
print(f"Overlap: {len(overlap)}")
print(f"\nSample extracted not in TWN_LR:")
print(list(extracted_set - twn_lr_set)[:10])
