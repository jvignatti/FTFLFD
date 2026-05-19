"""
crosswalkFUNCL.py
-----------------
Diagnostic: classify LRSNUMBER and TWN_LR format patterns to identify join feasibility.

Key finding: LRSNUMBER and TWN_LR use structurally different encoding schemes across all
route classes (Interstate, US, VT, local). A direct string join is not valid and explains
the road centerline join failure documented in OD-002.

Date: 2026-05-16
Status: diagnostic only
"""

import pandas as pd
import re

crashes    = pd.read_parquet("data/splits/train.parquet")
centerline = pd.read_csv("data/raw/road_centerline.csv")

crash_routes = crashes["LRSNUMBER"].dropna().unique()
twn_lr       = centerline["TWN_LR"].dropna().unique()

# Classify crash LRSNUMBER formats
def classify_lrs(lrs):
    if lrs.startswith("I"):
        return "Interstate"
    elif lrs.startswith("U"):
        return "US Route"
    elif lrs.startswith("V"):
        return "VT Route"
    elif lrs.startswith("S"):
        return "State"
    elif lrs.isdigit() or re.match(r"^\d", lrs):
        return "Numeric/Local"
    else:
        return "Other"

crash_df = pd.DataFrame({"LRSNUMBER": crash_routes})
crash_df["format"] = crash_df["LRSNUMBER"].apply(classify_lrs)
print("=== CRASH LRSNUMBER FORMATS ===")
print(crash_df["format"].value_counts())

# Classify centerline TWN_LR formats
def classify_twn(twn):
    if twn == "-":
        return "Null/Dash"
    elif twn.startswith("I"):
        return "Interstate"
    elif twn.startswith("U"):
        return "US Route"
    elif twn.startswith("V"):
        return "VT Route"
    elif twn.startswith("S"):
        return "State"
    elif twn.startswith("L"):
        return "Local"
    elif twn.startswith("Z"):
        return "Other/Z"
    else:
        return "Other"

twn_df = pd.DataFrame({"TWN_LR": twn_lr})
twn_df["format"] = twn_df["TWN_LR"].apply(classify_twn)
print("\n=== CENTERLINE TWN_LR FORMATS ===")
print(twn_df["format"].value_counts())

# Show samples of matching formats
print("\n=== CRASH US ROUTES (sample) ===")
print(crash_df[crash_df["format"]=="US Route"]["LRSNUMBER"].head(10).values)
print("\n=== CENTERLINE US ROUTES (sample) ===")
print(twn_df[twn_df["format"]=="US Route"]["TWN_LR"].head(10).values)

print("\n=== CRASH INTERSTATES (sample) ===")
print(crash_df[crash_df["format"]=="Interstate"]["LRSNUMBER"].head(10).values)
print("\n=== CENTERLINE INTERSTATES (sample) ===")
print(twn_df[twn_df["format"]=="Interstate"]["TWN_LR"].head(10).values)