# 07 — AADT Coverage Diagnosis

**Date:** 2026-05-16  
**Status:** Complete — Phase 1 revert complete, coverage recomputed 2026-05-16  
**Author:** JC  
**Diagnostic scripts:** `notebooks/diagnostic/schema_check.py`, `aadt_coverage_analysis.py`, `pdo_reporting_trend.py`

---

## 1. Context

Phase 1 (iterations 001–008) established LogReg iter_001 as the only model satisfying
all constraints (fatal recall 0.508, flag rate 0.143). Phase 2 retraining on an expanded
training window (2010–2022) reduced fatal recall to 0.427, violating the 0.50 floor.

Drift diagnosis (`06_drift_diagnosis.md`) identified the root cause: 90% of fatal
locations do not persist across eras, and all 8 existing features are crash-history
derived. They cannot predict fatals at new locations.

The proposed fix is injection of location-intrinsic features (AADT, FunctionalClass,
IsDivided) that describe the road itself rather than its crash history. This document
records the diagnostic work performed to evaluate data availability and feasibility
of that approach before any modeling changes are made.

---

## 2. Data Sources Identified

### 2.1 Crash Data

| File | Records | Years | Notes |
|---|---|---|---|
| `data/splits/train.parquet` | 109,695 | 2010–2019 | Phase 1 split — current active config |
| `data/splits/val.parquet` | 12,964 | 2020–2021 | Phase 1 validation |
| `data/processed/vt_crashes_ingested.parquet` | 179K+ | 2010–2026 | Full ingested dataset |

**Note:** Phase 1 revert is complete (commit 4048a05). Coverage numbers in Section 4 were
recomputed against the Phase 1 training set on 2026-05-16. Prior Phase 2 numbers
(train 130,946 rows, 58.06% overall coverage) are superseded by the values in Section 4.2.

### 2.2 AADT Data

| File | Records | Schema |
|---|---|---|
| `data/raw/aadt_limited.csv` | 765 | 29 columns |
| `data/raw/aadt_other.csv` | 2,572 | 29 columns (identical schema) |

Both files share the same schema. Key columns:

| Column | Type | Notes |
|---|---|---|
| `StandardRouteCode` | object | Join key — matches crash `LRSNUMBER` |
| `BeginMM` | float64 | Section begin milepoint |
| `EndMM` | float64 | Section end milepoint |
| `AADT` | int64 | Annual Average Daily Traffic |
| `FunctionalClass` | int64 | FHWA functional class code |
| `IsDivided` | object | Y/N — divided roadway flag |
| `Year` | int64 | Survey year |

**Critical finding:** Both AADT files contain only `Year = 2024`. Year-matched joining
to crash records (2010–2022) is not possible. AADT, FunctionalClass, and IsDivided
must be treated as **time-invariant road characteristics**.

This assumption is documented and locked: Vermont's road network changes slowly enough
that a 2024 AADT snapshot is considered a valid proxy for functional class and
divided/undivided status across the 2010–2019 training window. AADT volume values
are used as relative exposure indicators, not absolute traffic counts.

---

## 3. PDO Reporting Collapse

### 3.1 Findings

PDO (Property Damage Only) records show a sharp structural break starting in 2020,
acknowledged by the Vermont Agency of Transportation as a reporting breakdown across
multiple agencies.

**PDO records by year:**

| Year | PDO Records | YoY Change |
|---|---|---|
| 2010 | 8,760 | baseline |
| 2011 | 10,231 | +16.8% |
| 2012 | 9,302 | −9.1% |
| 2013 | 9,468 | +1.8% |
| 2014 | 9,144 | −3.4% |
| 2015 | 8,673 | −5.2% |
| 2016 | 8,270 | −4.6% |
| 2017 | 8,210 | −0.7% |
| 2018 | 7,994 | −2.6% |
| 2019 | 7,771 | −2.8% |
| 2020 | 5,285 | **−32.0%** |
| 2021 | 5,401 | +2.2% |
| 2022 | 5,552 | +2.8% |

**Phase 1 annual average (2010–2019):** ~8,782 records/year  
**Phase 2 annual average (2020–2022):** ~5,413 records/year  
**Phase 2 drop:** −38.3% from Phase 1 average

**Fatal records by year:** Stable throughout. Range: 42–73 per year, mean ~60,
no directional trend. Fatal reporting is mandatory and unaffected by the PDO
reporting breakdown.

### 3.2 Implication

The PDO contamination is almost entirely a Phase 2 phenomenon. Reverting to Phase 1
splits (2010–2019) largely resolves it. This is an additional, independent justification
for the Phase 1 revert beyond the modeling performance reasons established in
`06_drift_diagnosis.md`.

PDO records in the Phase 1 window show a gradual 11% decline over 9 years — manageable
and unlikely to materially bias crash-history features over that period.

---

## 4. AADT Join Coverage (Phase 1 Training Set)

### 4.1 Join Logic

- **Join key:** `crash.LRSNUMBER == aadt.StandardRouteCode`
- **Interval filter:** `crash.AOTACTUALMILEPOINT >= aadt.BeginMM AND <= aadt.EndMM`
- **Method:** left merge, crash as primary, deduplicated on `(StandardRouteCode, BeginMM, EndMM)`
- **Year matching:** not applied — single 2024 snapshot used for all crash years

### 4.2 Coverage Results

**Recomputed on Phase 1 training set (2010–2019) on 2026-05-16.**

| Severity | Total Records | Matched | Coverage |
|---|---|---|---|
| All records | 109,695 | 62,296 | **56.79%** |
| Fatal | 566 | 396 | **69.96%** |
| Injury | 21,325 | 14,205 | **66.61%** |
| PDO | 87,804 | 47,695 | **54.32%** |

### 4.3 Gate 1 Assessment

Gate 1 threshold: feature availability ≥ 95% of training records.

**Gate 1 fails at all severity levels on the Phase 1 training set.**
Overall coverage of 56.79% is 38 percentage points below threshold.

However, the coverage pattern is informative:

1. **Missingness is structural, not random.** Unmatched crashes concentrate on
   local and town roads not included in the VTrans AADT survey. This is the
   best-case pattern for this use case — fatals concentrate on state-maintained
   roads which have the highest AADT coverage.

2. **Fatal coverage (69.96%) is the most relevant number.** The Gate 1 threshold
   was designed to prevent sparse features from distorting model coefficients.
   A feature that covers 70% of fatals — the signal the model is trained to predict —
   is more informative than the same coverage rate over all records.

3. **Phase 1 vs Phase 2 comparison.** Phase 1 overall coverage (56.79%) is slightly
   lower than Phase 2 (58.06%) despite having fewer PDO records. This is because PDO
   crashes that do match AADT were removed when the training window shrank. Fatal
   coverage is stable: 69.96% (Phase 1) vs 70.85% (Phase 2) — a 0.89 point difference
   that does not change the Gate 1 outcome.

### 4.4 Missingness Mechanism

Two distinct failure modes exist in the join:

- **Route not in AADT at all:** crash LRSNUMBER has no matching StandardRouteCode
  in either AADT file. These are primarily local roads, town highways, and
  non-state-system routes not included in the VTrans traffic count program.

- **Route in AADT but milepoint falls in a gap:** StandardRouteCode matches but
  the crash milepoint does not fall within any surveyed section's BeginMM–EndMM
  interval. These are spatial gaps between adjacent AADT sections on the same route.

These are distinct categories with different implications for imputation strategy
and must be reported separately. See `notebooks/diagnostic/aadt_coverage_analysis.py`
for the breakdown.

---

## 5. Feature Dependency Structure

All three candidate AADT features derive from the same join:

| Feature | Source Column | Notes |
|---|---|---|
| `segment_aadt` | `AADT` | Raw volume count |
| `functional_class` | `FunctionalClass` | FHWA integer code |
| `is_divided` | `IsDivided` | Binary Y/N |

This has a direct implication for testing order: these features are not independent
joins. A crash record that fails the AADT join will have null values for all three
features simultaneously. Testing them in separate iterations remains correct per
CLAUDE.md one-change-per-iteration rules, but the imputation/exclusion strategy
decided for one feature applies structurally to all three.

---

## 6. Open Decisions

The following decisions are deferred until Phase 1 coverage numbers are available.
None of these decisions may be made or implied by modeling code until they are
explicitly documented and locked.

| # | Decision | Blocker |
|---|---|---|
| 1 | Impute missing AADT vs exclude unmatched records vs reject feature | Coverage now available (56.79% overall, 69.96% fatal). **Decision still pending.** See `open_decisions.md`. |
| 2 | Whether to source FunctionalClass independently from road network layer | **REOPENED** — road centerline join diagnosed as invalid (TWN_LR/LRSNUMBER format mismatch). FunctionalClass to be sourced from AADT files via the validated StandardRouteCode join. Coverage via AADT: 56.79% overall, 69.96% fatal — same as segment_aadt. Gate 1 outcome depends on OD-004 resolution. See `open_decisions.md` OD-002 and OD-004. |
| 3 | Final feature testing order for AADT-derived features | Decision 1 (imputation strategy) |
| 4 | Whether Gate 1 threshold applies to all records or fatal records only | Methodological discussion. See `open_decisions.md`. |

---

## 7. Immediate Next Steps

1. ~~Locate and read split generation logic~~ — **Done.** splits.yaml v1.2 active, Phase 1 boundaries confirmed.
2. ~~Regenerate train.parquet for Phase 1 (2010–2019)~~ — **Done.** 109,695 records, 2010-01-01 to 2019-12-30. Commit 4048a05.
3. ~~Rerun `aadt_coverage_analysis.py` against Phase 1 training set~~ — **Done 2026-05-16.** Results in Section 4.2.
4. **Resolve open decisions (Section 6)** — Coverage numbers now available. Decision 1 (imputation strategy) and Decision 4 (Gate 1 threshold scope) remain open. See `docs/technical/open_decisions.md`.
5. **Do not touch any modeling code** until open decisions are documented and locked.

---

## 8. Diagnostic Scripts

All scripts are in `notebooks/diagnostic/` and are self-contained, runnable from
the project root. They reproduce the exact outputs that informed this document.

| Script | Purpose |
|---|---|
| `schema_check.py` | Field names, data types, route ID overlap |
| `aadt_coverage_analysis.py` | Record-level and severity-stratified coverage |
| `pdo_reporting_trend.py` | PDO collapse quantification, fatal stability |

To reproduce all outputs:
```
python notebooks/diagnostic/schema_check.py
python notebooks/diagnostic/aadt_coverage_analysis.py
python notebooks/diagnostic/pdo_reporting_trend.py
```