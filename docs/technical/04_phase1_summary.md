# Phase 1 Summary — Complete

**Document:** `docs/technical/04_phase1_summary.md`
**Date:** 2026-05-03
**Status:** Phase 1 closed

---

## Objective

Build a functional proactive traffic fatality prediction pipeline using Vermont crash data. Validate that ML adds value beyond naive heuristics. Identify limitations honestly.

## What Was Built

An end-to-end prediction pipeline:

    Raw CSV → Ingestion → Time-Based Splitting → Rolling Window Generation
    → Feature Engineering → Model Training → Evaluation → Benchmark Audit

### Data
- 179,282 Vermont crash records (2010–2026) from VTrans Public Crash Query Tool
- 154,838 labeled records after cleaning (958 fatal, 31,754 injury, 122,126 PDO)
- Hybrid spatial segmentation: LRS segments (primary) + 1km grid cells (secondary)
- 99.5% fatal coverage (953 of 958 fatalities assigned to prediction units)
- 5 fatalities excluded due to missing spatial data (documented)
- 123 ghost rows identified and excluded (documented)

### Splits
- Train: 2010–2019 (109,695 crashes, 566 fatal)
- Val: 2020–2021 (12,964 crashes, 124 fatal)
- B1: 2022 (6,711 crashes, 69 fatal)
- B2: 2023 (6,426 crashes, 63 fatal)
- B3: 2024–2025 (reserved)
- B4: 2026 (final test, untouched)
- 30-day gaps enforced between all splits

### Features (8 baseline)
1. segment_crash_rate (historical crashes per year)
2. segment_fatal_rate (historical fatals per year)
3. segment_injury_rate (historical injuries per year)
4. segment_pdo_rate (historical PDO per year)
5. month (1-12)
6. is_winter (Nov-Mar)
7. is_weekend_heavy (Jun-Oct)
8. segment_type_is_grid (LRS vs grid segment)

### Infrastructure
- 7 automated leakage checks (all passing)
- Naive baseline comparisons (Top-K and random)
- Experiment registry with 15-column audit trail
- Narrative iteration log with hypothesis/result/decision
- Benchmark audit log (append-only JSONL)
- SHA256 hashes for all data files

## Iterations Completed

| Iter | Model | Change | Result |
|---|---|---|---|
| 001 | LogReg | Baseline 8 features | ACCEPTED — fatal recall 0.508, combined 0.569 |
| 002 | LogReg | Threshold 0.50→0.52 | REJECTED — fatal recall dropped to 0.456 |
| 003a | LogReg | +segment_impairment_ratio | REJECTED Gate 3 — collinear with rates |
| 003b | LogReg | +segment_crash_trend | REJECTED Gate 2 — too sparse |
| 003c | LogReg | +road_group_risk | REJECTED Gate 3 — gain 0.0015 |
| 004 | RF | Model advancement depth=10 | CONDITIONAL — recall up but over-flags |
| 005 | RF | max_depth 10→6 | REJECTED — depth not the cause |
| 006 | RF | class_weight 71x→10x | REJECTED — too conservative |
| 007 | RF | Weight sweep 20x-50x | REJECTED — no feasible weight |
| 008 | LightGBM | Model advancement | REJECTED — same over-flagging pattern |

## Key Findings

### 1. LogReg is the best model for 8 features
Three model classes tested. Only LogReg satisfies all constraints simultaneously. Tree-based models (RF, LightGBM) cannot improve over the linear baseline because the feature set is too small and correlated for non-linear methods to exploit.

### 2. The model adds real predictive value
At 14.3% flag rate, LogReg catches 66% of fatal windows — 4x better than Top-K naive baseline (15.8% fatal recall at 15.3% flag rate). This is not a frequency model. It is genuinely predictive.

### 3. Temporal drift is the primary limitation
Performance degrades with distance from training data:
- Val (1-2 years out): fatal recall 0.508
- B1 (3 years out): fatal recall 0.523
- B2 (4 years out): fatal recall 0.463 (below 0.50 floor)
Retraining with newer data is required for production viability.

### 4. Feature engineering has diminishing returns with LogReg
Four feature candidates tested, all rejected. Segment-level aggregates of crash attributes are collinear with existing rate features. New signal requires either orthogonal data sources (traffic volume, road geometry, weather) or non-linear models with more features.

### 5. Impairment is the strongest crash-level fatal predictor
47.5% of fatals involve impairment vs 10.4% of injuries (4.5x overrepresentation). However, this crash-level signal does not translate to segment-level predictive value with current features.

## Benchmark Results (Audit Only)

| Benchmark | Combined Recall | Fatal Recall | Gap from Val |
|---|---|---|---|
| B1 (2022) | 0.676 | 0.523 | +0.015 combined, -0.053 fatal |
| B2 (2023) | 0.481 | 0.463 | -0.088 combined, -0.045 fatal |

B2 confirmed temporal drift. The model's shelf life is approximately 3 years without retraining.

## What Phase 1 Did NOT Do

- Did not retrain with expanded data (Phase 2)
- Did not incorporate traffic volume/flow data (Phase 2)
- Did not distinguish serious injury from minor injury (pending DOH data)
- Did not implement real-time prediction serving
- Did not build stakeholder-facing dashboards or maps
- Did not resolve 23 U.S.C. § 409 legal questions

## Phase 2 Direction

1. Retrain with expanded training window to address temporal drift
2. Incorporate traffic flow data from VTrans TCDS
3. Re-evaluate features that failed Gate 3 with LogReg — they may pass with fresher training data
4. Validate on B3 (2024-2025) and preserve B4 (2026) as final test