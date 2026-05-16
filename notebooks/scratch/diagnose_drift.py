"""
Diagnose temporal drift — understand what changed between eras.

Questions:
  1. Are the same segments producing fatals in 2023 as in 2010-2019?
  2. Did the geographic distribution of crashes shift?
  3. Did the relationship between features and outcomes change?
  4. Is the drift in the segments, the timing, or both?
"""

import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"

# Load all crash-level data by era
print("Loading crash data by era...")

# We need the full ingested dataset
processed = REPO_ROOT / "data" / "processed" / "vt_crashes_ingested.parquet"
df = pd.read_parquet(processed)

# Define eras
df["year"] = df["event_date"].dt.year
era_1 = df[(df["year"] >= 2010) & (df["year"] <= 2014)]
era_2 = df[(df["year"] >= 2015) & (df["year"] <= 2019)]
era_3 = df[(df["year"] >= 2020) & (df["year"] <= 2021)]
era_4 = df[(df["year"] >= 2022) & (df["year"] <= 2023)]

print(f"  Era 1 (2010-2014): {len(era_1):,} crashes")
print(f"  Era 2 (2015-2019): {len(era_2):,} crashes")
print(f"  Era 3 (2020-2021 COVID): {len(era_3):,} crashes")
print(f"  Era 4 (2022-2023 recent): {len(era_4):,} crashes")

# Q1: Are the same segments producing fatals across eras?
print("\n" + "=" * 60)
print("Q1: FATAL SEGMENT PERSISTENCE ACROSS ERAS")
print("=" * 60)

for era_name, era_df in [("2010-2014", era_1), ("2015-2019", era_2), ("2020-2021", era_3), ("2022-2023", era_4)]:
    fatal_segs = set(era_df[era_df["severity_class"] == "fatal"]["segment_id"].dropna().unique())
    all_segs = set(era_df[era_df["segment_id"].notna()]["segment_id"].unique())
    print(f"\n  {era_name}:")
    print(f"    Total segments with crashes: {len(all_segs):,}")
    print(f"    Segments with fatals:        {len(fatal_segs):,}")
    print(f"    Fatal concentration:         {len(fatal_segs)/max(len(all_segs),1):.4f}")

# Cross-era fatal segment overlap
fatal_segs_1 = set(era_1[era_1["severity_class"] == "fatal"]["segment_id"].dropna())
fatal_segs_2 = set(era_2[era_2["severity_class"] == "fatal"]["segment_id"].dropna())
fatal_segs_3 = set(era_3[era_3["severity_class"] == "fatal"]["segment_id"].dropna())
fatal_segs_4 = set(era_4[era_4["severity_class"] == "fatal"]["segment_id"].dropna())

train_fatal = fatal_segs_1 | fatal_segs_2
recent_fatal = fatal_segs_4

overlap = train_fatal & recent_fatal
only_train = train_fatal - recent_fatal
only_recent = recent_fatal - train_fatal

print(f"\n  Fatal segment overlap (training 2010-2019 vs recent 2022-2023):")
print(f"    Segments fatal in BOTH eras:        {len(overlap)}")
print(f"    Segments fatal in training ONLY:     {len(only_train)}")
print(f"    Segments fatal in recent ONLY:       {len(only_recent)}")
print(f"    Overlap rate:                        {len(overlap)/max(len(recent_fatal),1):.2%}")

# Q2: Did crash volume per segment change?
print("\n" + "=" * 60)
print("Q2: CRASH VOLUME SHIFT BY ROAD GROUP")
print("=" * 60)

for era_name, era_df in [("2010-2019", pd.concat([era_1, era_2])), ("2020-2021", era_3), ("2022-2023", era_4)]:
    years = era_df["year"].nunique()
    print(f"\n  {era_name} (annual rate):")
    rg = era_df.groupby("RoadGroup").size() / years
    print(rg.sort_values(ascending=False).head(8).to_string())

# Q3: Did the feature-outcome relationship change?
print("\n" + "=" * 60)
print("Q3: FATAL RATE BY ROAD GROUP ACROSS ERAS")
print("=" * 60)

for era_name, era_df in [("2010-2019", pd.concat([era_1, era_2])), ("2020-2021", era_3), ("2022-2023", era_4)]:
    print(f"\n  {era_name}:")
    rg_stats = era_df.groupby("RoadGroup").agg(
        total=("OBJECTID", "count"),
        fatal=("severity_class", lambda x: (x == "fatal").sum()),
    )
    rg_stats["fatal_rate"] = (rg_stats["fatal"] / rg_stats["total"]).round(4)
    rg_stats = rg_stats.sort_values("total", ascending=False)
    print(rg_stats.head(8).to_string())

# Q4: New segments in recent years
print("\n" + "=" * 60)
print("Q4: NEW SEGMENTS APPEARING IN RECENT YEARS")
print("=" * 60)

train_all_segs = set(pd.concat([era_1, era_2])["segment_id"].dropna().unique())
recent_all_segs = set(era_4["segment_id"].dropna().unique())
new_segs = recent_all_segs - train_all_segs

print(f"  Segments in training (2010-2019):  {len(train_all_segs):,}")
print(f"  Segments in recent (2022-2023):    {len(recent_all_segs):,}")
print(f"  NEW segments (not in training):    {len(new_segs):,}")
print(f"  New segment rate:                  {len(new_segs)/max(len(recent_all_segs),1):.2%}")

# How many fatals happen in new segments?
recent_fatal_df = era_4[era_4["severity_class"] == "fatal"]
fatals_in_new = recent_fatal_df[recent_fatal_df["segment_id"].isin(new_segs)]
fatals_in_old = recent_fatal_df[recent_fatal_df["segment_id"].isin(train_all_segs)]

print(f"\n  Recent fatals in KNOWN segments:   {len(fatals_in_old)}")
print(f"  Recent fatals in NEW segments:     {len(fatals_in_new)}")
print(f"  % fatals in new segments:          {len(fatals_in_new)/max(len(recent_fatal_df),1):.2%}")

# Q5: Temporal patterns
print("\n" + "=" * 60)
print("Q5: MONTHLY FATAL RATE ACROSS ERAS")
print("=" * 60)

for era_name, era_df in [("2010-2019", pd.concat([era_1, era_2])), ("2020-2021", era_3), ("2022-2023", era_4)]:
    era_df = era_df.copy()
    era_df["month"] = era_df["event_date"].dt.month
    fatal = era_df[era_df["severity_class"] == "fatal"]
    monthly = fatal.groupby("month").size()
    years = era_df["year"].nunique()
    rate = (monthly / years).round(2)
    print(f"\n  {era_name} (fatals per month per year):")
    print(rate.to_string())

# Q6: Impairment pattern shift
print("\n" + "=" * 60)
print("Q6: IMPAIRMENT IN FATALS ACROSS ERAS")
print("=" * 60)

for era_name, era_df in [("2010-2019", pd.concat([era_1, era_2])), ("2020-2021", era_3), ("2022-2023", era_4)]:
    fatal = era_df[era_df["severity_class"] == "fatal"]
    total_fatal = len(fatal)
    impaired = fatal["Impairment"].isin(["Alcohol", "Drugs", "Alcohol and Drugs"]).sum()
    print(f"  {era_name}: {impaired}/{total_fatal} fatals impaired ({impaired/max(total_fatal,1):.1%})")

print("\n" + "=" * 60)
print("DRIFT DIAGNOSIS COMPLETE")
print("=" * 60)