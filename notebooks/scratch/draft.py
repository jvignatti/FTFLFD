import pandas as pd
import numpy as np

crashes = pd.read_parquet("data/splits/train.parquet")
crashes["crash_id"] = np.arange(len(crashes))

aadt = pd.concat([
    pd.read_csv("data/raw/aadt_limited.csv"),
    pd.read_csv("data/raw/aadt_other.csv")
]).drop_duplicates(subset=["StandardRouteCode", "BeginMM", "EndMM"])

joined = crashes.merge(
    aadt[["StandardRouteCode", "BeginMM", "EndMM", "AADT"]],
    left_on="LRSNUMBER",
    right_on="StandardRouteCode",
    how="left"
)

joined["aadt_match"] = (
    joined["AADT"].notna()
    & (joined["AOTACTUALMILEPOINT"] >= joined["BeginMM"])
    & (joined["AOTACTUALMILEPOINT"] <= joined["EndMM"])
)

# Classify each crash into exactly one population
route_in_aadt = joined.groupby("crash_id")["AADT"].any()
interval_match = joined.groupby("crash_id")["aadt_match"].max()

result = crashes[["crash_id", "severity_class"]].copy()
result["route_in_aadt"] = result["crash_id"].map(route_in_aadt)
result["interval_match"] = result["crash_id"].map(interval_match)

result["population"] = "A_no_route"
result.loc[result["route_in_aadt"] & ~result["interval_match"], "population"] = "B_gap"
result.loc[result["interval_match"], "population"] = "C_matched"

print("=== POPULATION BREAKDOWN BY SEVERITY ===")
print(result.groupby(["severity_class", "population"]).size().unstack(fill_value=0))

print("\n=== FATAL BREAKDOWN ===")
fatal = result[result["severity_class"] == "fatal"]
print(fatal["population"].value_counts())
print(f"\nFatal total: {len(fatal)}")