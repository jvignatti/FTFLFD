"""
check_holidays.py
-----------------
Diagnostic: explore whether specific holiday windows show elevated fatal crash rates,
Phase 1 training set (2010–2019).

Key finding: holiday windows (New Year's, July 4th, Halloween) do not show a consistent
spike in daily fatal rate large enough to justify a dedicated holiday feature.

Date: 2026-05-15
Status: diagnostic only
"""

import pandas as pd

df = pd.read_parquet("data/splits/train.parquet")
df["month"] = df["event_date"].dt.month
df["day"] = df["event_date"].dt.day
df["dow"] = df["event_date"].dt.day_name()

fatal = df[df["severity_class"] == "fatal"]

print("=== FATAL CRASHES BY MONTH ===")
print(fatal["month"].value_counts().sort_index())

print("\n=== FATAL CRASHES BY DAY OF WEEK ===")
print(fatal["dow"].value_counts())

print("\n=== FATAL CRASHES DEC 31 - JAN 2 (NEW YEARS) ===")
ny = fatal[((fatal["month"] == 12) & (fatal["day"] >= 31)) | ((fatal["month"] == 1) & (fatal["day"] <= 2))]
print(f"Count: {len(ny)}")

print("\n=== FATAL CRASHES JUL 3-5 (JULY 4TH) ===")
j4 = fatal[(fatal["month"] == 7) & (fatal["day"].between(3, 5))]
print(f"Count: {len(j4)}")

print("\n=== FATAL CRASHES OCT 31 (HALLOWEEN) ===")
hw = fatal[(fatal["month"] == 10) & (fatal["day"] == 31)]
print(f"Count: {len(hw)}")

print("\n=== DAILY FATAL RATE (avg per day by month) ===")
days_per_month = df.groupby("month")["event_date"].apply(lambda x: x.dt.date.nunique())
fatals_per_month = fatal.groupby("month").size()
rate = (fatals_per_month / days_per_month).round(4)
print(rate.sort_index())