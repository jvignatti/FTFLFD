# Phase 2 Plan — Retrain and Data Expansion

**Document:** `docs/technical/05_phase2_plan.md`
**Version:** 1.0
**Created:** 2026-05-03
**Status:** ABANDONED — 2026-05-16

> **STATUS: ABANDONED** — Phase 2 training (2010–2022) produced fatal recall 0.427,
> below the 0.50 floor. PDO reporting collapse confirmed as a 2020–2022 phenomenon.
> Splits reverted to Phase 1 (v1.2, train 2010–2019). See `docs/technical/07_aadt_coverage_diagnosis.md`.
> B3/B4 parquet files that predate this plan remain valid Phase 1 artifacts (see Section 2 note).

---

## 1. Why Phase 2

Phase 1 identified temporal drift as the primary limitation. The model trained on 2010–2019 data degrades by year 4:

| Benchmark | Years from Training | Fatal Recall |
|---|---|---|
| Val (2020–2021) | 1–2 | 0.508 |
| B1 (2022) | 3 | 0.523 |
| B2 (2023) | 4 | 0.463 (below 0.50 floor) |

Retraining with more recent data is the direct fix.

## 2. Phase 2 Split Strategy (v2.0)

| Set | Phase 1 (v1.1) | Phase 2 (v2.0) | Rationale |
|---|---|---|---|
| Train | 2010–2019 | 2010–2022 | Add 3 years including COVID recovery |
| Val | 2020–2021 | 2023 | Most recent stable year |
| B1 | 2022 | 2024 | First blind test |
| B2 | 2023 | 2025 | Second blind test |
| B3 | 2024–2025 | — | Absorbed into B1/B2 |
| B4 | 2026 | 2026 | Final test — UNCHANGED |

- 30-day gaps enforced between all splits (unchanged)
- data_era flag updated: early (2010–2014), historical (2015–2019), modern (2020+)
- B4 (2026) remains untouched — final test preserved across phases

> **Note on Phase 1 B3 artifacts:** The files `data/splits/b3.parquet`, `b3_featured.parquet`, and `b3_windows.parquet` still exist on disk. These are Phase 1 artifacts covering 2024–2025 under the old split schema. They are not referenced by any Phase 2 pipeline code and must not be used for tuning or evaluation. B3 no longer exists as a split in `config/splits.yaml` v2.0 — that date range is now covered by B1 (2024) and B2 (2025). Do not delete these files; they are retained as a Phase 1 audit trail.

## 3. AADT Data Integration

### 3.1 Source

Vermont Agency of Transportation — Annual Average Daily Traffic (AADT)

- ArcGIS REST Service: https://maps.vtrans.vermont.gov/arcgis/rest/services/Layers/AADT/FeatureServer
- Public access, no authentication required
- Two layers: AADT Limited (Layer 0), AADT Other (Layer 1)

### 3.2 Data Structure

The AADT data contains the following key fields:

| Field | Description | Relevance |
|---|---|---|
| AADT | Annual Average Daily Traffic count | THE exposure denominator — transforms crash counts into crash rates per vehicle-miles-traveled |
| BeginMM / EndMM | Begin and end milepoints along each route | Maps directly to LRS segments via milepoint matching |
| RouteName / RouteNum | Route identification | Cross-reference with crash data LRSNUMBER |
| FunctionalClass | FHWA standard road classification | Cleaner version of RoadGroup — standard functional class categories |
| TownName | Municipality | Cross-reference for town aggregation |
| Year | Year the AADT was calculated | Enables historical AADT — traffic volume can vary over time in the model |
| IsDivided | Whether road is divided | Structural road characteristic relevant to crash severity |
| StandardRouteCode | Standardized route identifier | Alternative join key |
| Geometry | Polyline (road segments) | Spatial reference for verification |

Spatial Reference: EPSG 32145 (Vermont State Plane, meters)

### 3.3 Why AADT Changes the Model

Phase 1 features use crash COUNTS per segment. This creates a fundamental blind spot:

- A segment with 50,000 vehicles/day and 5 crashes = low risk (1 crash per 10,000 vehicles)
- A segment with 500 vehicles/day and 5 crashes = high risk (1 crash per 100 vehicles)

Without AADT, the model treats both segments identically. With AADT, it can distinguish between high-volume/low-rate segments (safe despite many crashes) and low-volume/high-rate segments (dangerous despite few crashes).

### 3.4 Features Derivable from AADT

| Feature | Definition | What It Captures |
|---|---|---|
| segment_aadt | Raw traffic volume for the segment | Exposure level |
| segment_crash_rate_per_vmt | Crashes / (AADT × segment_length × 365) | True crash risk rate normalized by exposure |
| segment_fatal_rate_per_vmt | Fatal crashes / (AADT × segment_length × 365) | True fatality risk rate normalized by exposure |
| functional_class | FHWA functional classification | Road type (interstate, arterial, collector, local) |
| is_divided | Whether road has divided lanes | Structural characteristic affecting head-on risk |

### 3.5 Integration Method

AADT data uses milepoints (BeginMM, EndMM) on routes. Our crash segments use floor(AOTACTUALMILEPOINT) on LRSNUMBER. The join logic:

    For each crash segment (LRSNUMBER, floor_milepoint):
      Find the AADT record where:
        - Route matches (StandardRouteCode ↔ LRSNUMBER crosswalk)
        - BeginMM <= floor_milepoint < EndMM
      Assign that record's AADT value to the segment

Segments without AADT coverage (local roads, grid-only segments) get AADT = null and are handled with a fallback strategy (median AADT for road group, or explicit "no AADT" flag).

### 3.6 Integration Sequence

1. Pull AADT data from ArcGIS REST API (same method as crash data pull)
2. Document schema and compute SHA256 hash
3. Build route crosswalk (LRSNUMBER ↔ StandardRouteCode)
4. Join AADT to segments by milepoint overlap
5. Compute AADT-derived features
6. Run 3-gate test for each new feature (one at a time per CLAUDE.md rules)
7. Retrain if any feature passes all gates

### 3.7 Open Questions

- Does the AADT service provide historical years (multiple Year values) or only current year?
- What is the coverage on non-State-System roads?
- How does the route coding in AADT (StandardRouteCode) map to crash data (LRSNUMBER)?
- Are there AADT segments that span multiple of our 1-mile crash segments, or vice versa?

These will be answered during data exploration after the pull.

## 4. Phase 2 Iteration Plan

| Step | Action | Dependency |
|---|---|---|
| 1 | Update splits.yaml to v2.0 | None |
| 2 | Re-run ingestion with new split boundaries | Step 1 |
| 3 | Re-run window generation | Step 2 |
| 4 | Re-run baseline features | Step 3 |
| 5 | Retrain LogReg iter_001 equivalent on new splits | Step 4 |
| 6 | Validate on 2023 data | Step 5 |
| 7 | Run B1 (2024) benchmark | Step 6 |
| 8 | Pull and explore AADT data | Parallel with steps 1-7 |
| 9 | Build route crosswalk | Step 8 |
| 10 | Integrate AADT features (one at a time) | Steps 7 + 9 |
| 11 | Run B2 (2025) benchmark on best model | Step 10 |

## 5. Success Criteria for Phase 2

| Metric | Phase 1 Result | Phase 2 Target |
|---|---|---|
| Fatal recall (val) | 0.508 | > 0.55 |
| Fatal recall (B1) | 0.523 | > 0.50 |
| Combined recall (val) | 0.569 | > 0.60 |
| Flag rate | 0.143 | ≤ 0.20 |
| Precision | 0.052 | ≥ 0.05 |
| Generalization gap | 0.064 | ≤ 0.10 |

B4 (2026) is NOT evaluated in Phase 2. It remains the final test for the completed system.