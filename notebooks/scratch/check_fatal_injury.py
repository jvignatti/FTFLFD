"""
check_fatal_injury.py
---------------------
Diagnostic: temporal and impairment patterns in fatal and injury crashes, Phase 1
training set (2010–2019).

Key finding: fatal crashes peak in summer months and on weekends; impaired driving is
elevated in fatals relative to injuries. No single pattern is strong enough to justify
a new standalone feature.

Date: 2026-05-15
Status: diagnostic only
"""

import pandas as pd

df = pd.read_parquet("data/splits/train.parquet")
df["month"] = df["event_date"].dt.month
df["day"] = df["event_date"].dt.day
df["dow"] = df["event_date"].dt.day_name()
df["year"] = df["event_date"].dt.year

fatal = df[df["severity_class"] == "fatal"]
injury = df[df["severity_class"] == "injury"]
fsi = df[df["severity_class"].isin(["fatal", "injury"])]

print("=== FATAL + INJURY BY MONTH ===")
print(fsi["month"].value_counts().sort_index())

print("\n=== FATAL + INJURY BY DAY OF WEEK ===")
print(fsi["dow"].value_counts())

print("\n=== FATAL ONLY BY DAY OF WEEK ===")
print(fatal["dow"].value_counts())

print("\n=== INJURY ONLY BY DAY OF WEEK ===")
print(injury["dow"].value_counts())

print("\n=== DAILY FSI RATE (avg per day by month) ===")
days_per_month = df.groupby("month")["event_date"].apply(lambda x: x.dt.date.nunique())
fsi_per_month = fsi.groupby("month").size()
rate = (fsi_per_month / days_per_month).round(4)
print(rate.sort_index())

print("\n=== FATAL RATE VS INJURY RATE BY MONTH ===")
fatal_per_month = fatal.groupby("month").size()
injury_per_month = injury.groupby("month").size()
fatal_rate = (fatal_per_month / days_per_month).round(4)
injury_rate = (injury_per_month / days_per_month).round(4)
comparison = pd.DataFrame({
    "fatal_rate": fatal_rate,
    "injury_rate": injury_rate,
    "fsi_rate": rate,
    "fatal_pct_of_fsi": (fatal_per_month / fsi_per_month * 100).round(1)
})
print(comparison)

print("\n=== WEEKEND VS WEEKDAY RATES ===")
is_weekend = fsi["dow"].isin(["Friday", "Saturday", "Sunday"])
weekend_days = df[df["dow"].isin(["Friday", "Saturday", "Sunday"])]["event_date"].dt.date.nunique()
weekday_days = df[~df["dow"].isin(["Friday", "Saturday", "Sunday"])]["event_date"].dt.date.nunique()
print("Weekend FSI per day: {:.4f}".format(is_weekend.sum() / weekend_days))
print("Weekday FSI per day: {:.4f}".format((~is_weekend).sum() / weekday_days))
fatal_weekend = fatal[fatal["dow"].isin(["Friday", "Saturday", "Sunday"])]
fatal_weekday = fatal[~fatal["dow"].isin(["Friday", "Saturday", "Sunday"])]
print("Weekend fatal per day: {:.4f}".format(len(fatal_weekend) / weekend_days))
print("Weekday fatal per day: {:.4f}".format(len(fatal_weekday) / weekday_days))

print("\n=== IMPAIRMENT IN FATAL VS INJURY ===")
print("Fatal impairment:")
print(fatal["Impairment"].value_counts(dropna=False))
print("\nInjury impairment:")
print(injury["Impairment"].value_counts(dropna=False).head(10))

print("\n=== DAYNIGHT IN FATAL VS INJURY ===")
print("Fatal DayNight:")
print(fatal["DayNight"].value_counts(dropna=False))
print("\nInjury DayNight:")
print(injury["DayNight"].value_counts(dropna=False))

print("\n=== FATAL + INJURY BY YEAR ===")
yearly = pd.DataFrame({
    "fatal": fatal.groupby("year").size(),
    "injury": injury.groupby("year").size(),
    "fsi_total": fsi.groupby("year").size(),
    "all_crashes": df.groupby("year").size(),
})
yearly["fatal_pct"] = (yearly["fatal"] / yearly["all_crashes"] * 100).round(2)
yearly["fsi_pct"] = (yearly["fsi_total"] / yearly["all_crashes"] * 100).round(2)
print(yearly)