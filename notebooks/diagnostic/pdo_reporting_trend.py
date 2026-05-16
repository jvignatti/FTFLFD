"""
pdo_reporting_trend.py
-----------------------
Diagnostic: quantify PDO reporting collapse and validate fatal reporting stability.

Purpose
-------
Vermont's crash reporting system experienced a documented breakdown in PDO
(Property Damage Only) reporting starting around 2020, acknowledged by the state.
This script quantifies the temporal trend by severity class and agency to:
  1. Confirm that PDO contamination is concentrated in the Phase 2 window (2020-2022)
  2. Confirm that fatal reporting is unaffected
  3. Identify which agencies drove the reporting drop
  4. Support the decision to revert to Phase 1 splits (2010-2019)

Key findings from initial run (Phase 2 train.parquet, 2010-2022):
    PDO 2010-2019:  7,771 - 10,231 per year (~11% gradual decline)
    PDO 2020-2022:  5,285 - 5,552 per year (~35% cliff from 2019)
    Fatal 2010-2022: 42 - 73 per year, stable, no directional trend

See docs/technical/07_aadt_coverage_diagnosis.md, Section 3.

Run from project root:
    python notebooks/diagnostic/pdo_reporting_trend.py

Outputs: printed to stdout.
    python notebooks/diagnostic/pdo_reporting_trend.py > notebooks/diagnostic/output/pdo_trend_output.txt
"""

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Paths — relative to project root (FTFLFD/)
# ---------------------------------------------------------------------------
TRAIN_PARQUET = "data/splits/train.parquet"

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
print("Loading data...")
crashes = pd.read_parquet(TRAIN_PARQUET)
crashes["year"] = pd.to_datetime(crashes["event_date"]).dt.year

# ---------------------------------------------------------------------------
# Training set composition
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("TRAINING SET COMPOSITION")
print("="*60)
comp = crashes["severity_class"].value_counts()
print(comp.to_string())
print(f"\nTotal records: {len(crashes):,}")
for sev in ["fatal", "injury", "pdo"]:
    n = comp.get(sev, 0)
    print(f"  {sev:<8}: {n:>7,}  ({n/len(crashes):.4%})")

# ---------------------------------------------------------------------------
# Records by year and severity
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("RECORDS BY YEAR AND SEVERITY CLASS")
print("="*60)
pivot = (
    crashes.groupby(["year", "severity_class"])
    .size()
    .unstack(fill_value=0)
    [["fatal", "injury", "pdo"]]
)
print(pivot.to_string())

# ---------------------------------------------------------------------------
# PDO year-over-year change
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PDO YEAR-OVER-YEAR CHANGE")
print("="*60)
pdo_by_year = crashes[crashes["severity_class"] == "pdo"].groupby("year").size()
pdo_pct_change = pdo_by_year.pct_change() * 100
for yr in pdo_by_year.index:
    chg = pdo_pct_change.get(yr, float("nan"))
    chg_str = f"{chg:+.1f}%" if not np.isnan(chg) else "baseline"
    print(f"  {yr}: {pdo_by_year[yr]:>6,}  ({chg_str})")

phase1_avg = pdo_by_year[pdo_by_year.index <= 2019].mean()
phase2_avg = pdo_by_year[pdo_by_year.index >= 2020].mean()
print(f"\n  Phase 1 avg (2010-2019): {phase1_avg:,.0f}")
print(f"  Phase 2 avg (2020-2022): {phase2_avg:,.0f}")
print(f"  Phase 2 drop:            {(phase2_avg - phase1_avg) / phase1_avg:.1%}")

# ---------------------------------------------------------------------------
# Fatal trend — confirm stability
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("FATAL RECORDS BY YEAR (stability check)")
print("="*60)
fatal_by_year = crashes[crashes["severity_class"] == "fatal"].groupby("year").size()
for yr in fatal_by_year.index:
    print(f"  {yr}: {fatal_by_year[yr]:>4}")
print(f"\n  Mean:   {fatal_by_year.mean():.1f}")
print(f"  StdDev: {fatal_by_year.std():.1f}")
print(f"  Min:    {fatal_by_year.min()} ({fatal_by_year.idxmin()})")
print(f"  Max:    {fatal_by_year.max()} ({fatal_by_year.idxmax()})")

# ---------------------------------------------------------------------------
# Agency-level PDO breakdown (top contributors to drop)
# ---------------------------------------------------------------------------
print("\n" + "="*60)
print("PDO REPORTING BY AGENCY — TOP 20 OVERALL")
print("="*60)
pdo = crashes[crashes["severity_class"] == "pdo"]
agency_pdo = pdo.groupby("ReportingAgency").size().sort_values(ascending=False)
print(agency_pdo.head(20).to_string())

print("\n" + "="*60)
print("PDO BY AGENCY x ERA (Phase 1 vs Phase 2, top 15 agencies)")
print("="*60)
pdo = pdo.copy()
pdo["era"] = pdo["year"].apply(lambda y: "phase1" if y <= 2019 else "phase2")
top_agencies = agency_pdo.head(15).index
pdo_top = pdo[pdo["ReportingAgency"].isin(top_agencies)]
era_pivot = (
    pdo_top.groupby(["ReportingAgency", "era"])
    .size()
    .unstack(fill_value=0)
)
if "phase1" in era_pivot.columns and "phase2" in era_pivot.columns:
    era_pivot["phase1_annual_avg"] = era_pivot["phase1"] / 10
    era_pivot["phase2_annual_avg"] = era_pivot["phase2"] / 3
    era_pivot["annual_pct_change"] = (
        (era_pivot["phase2_annual_avg"] - era_pivot["phase1_annual_avg"])
        / era_pivot["phase1_annual_avg"] * 100
    ).map("{:+.1f}%".format)
    print(era_pivot[["phase1", "phase2",
                      "phase1_annual_avg", "phase2_annual_avg",
                      "annual_pct_change"]].to_string())

print("\nDone.")