> **Note (2026-05-02):** This document reflects the initial project design. The authoritative source for current rules is CLAUDE.md at the project root. Key updates since this document was written: Event Contract v1.1 (Fatal as primary target), adaptive kill switch, prediction unit locked (1-mile LRS segments). A full revision of this document is pending.

# FTFLFD — Project Design Document

**Version:** 1.0  
**Status:** LOCKED  
**Date:** 2026-05-02  

---

## 1. Event Contract (v1.0 — LOCKED)

### 1.1 Target Events

A qualifying event is any crash recorded in Vermont crash data with:
- **Crash Type = "Fatal"**, OR
- **Crash Type = "Injury"**

Excluded from the positive class:
- Crash Type = "Property Damage Only" → labeled 0 (negative)
- Crash Type = blank/unknown → **dropped from training entirely, not labeled as 0**

### 1.2 Prediction Window

**30 days.** The model predicts whether a qualifying event will occur within a 30-day forward window from the observation date.

### 1.3 Prediction Unit

*To be locked during the data ingestion phase before iteration 1 begins.* The prediction unit (e.g., road segment, geographic zone, statewide window) must be defined and appended to this document as an amendment before any model training occurs.

### 1.4 Label Definition

| Label | Condition |
|-------|-----------|
| 1 (positive) | A qualifying event occurs within the 30-day prediction window for the given prediction unit |
| 0 (negative) | No qualifying event occurs within the 30-day prediction window for the given prediction unit |

### 1.5 Contract Lock

This event contract is locked at v1.0. Any change to the target event definition, prediction window, or label logic constitutes a **breaking change** and requires a new contract version. A new contract version resets the iteration counter and invalidates all prior experiment results.

---

## 2. Split Strategy (config/splits.yaml v1.1)

All splits are **time-based**. No random shuffling. Data must not cross split boundaries in any direction. A mandatory gap of **30 days** is enforced between every consecutive split — rows falling within a gap window are dropped.

| Set | Period | Role | Evaluation Frequency |
|-----|--------|------|----------------------|
| Train | 2010–2019 | Model fitting | Used freely |
| Val | 2020–2021 | Hyperparameter tuning and model selection | Used freely |
| B1 | 2022 | Blind evaluation — batch 1 | Every 5 iterations |
| B2 | 2023 | Blind evaluation — batch 2 | Every 10 iterations |
| B3 | 2024–2025 | Blind evaluation — batch 3 | Once per phase |
| B4 | 2026 | Final holdout | Once. Ever. Final test. |

### 2.1 Data Era Flag

Training data spans 2010–2019. Ingestion must add a `data_era` column to every row:

| Value | Years |
|-------|-------|
| `early` | 2010–2014 |
| `historical` | 2015–2019 |
| `modern` | 2020+ |

This flag is a locked contextual feature and is gate-exempt (see `config/features.yaml`).

### 2.2 Rules

- No feature engineering, preprocessing, or hyperparameter decision may reference B1, B2, B3, or B4 data.
- B4 is evaluated exactly once, after all iterations are exhausted or the kill switch fires.
- Leakage checks run before every split is used for the first time.
- Split boundaries and gap rules are defined in `config/splits.yaml` and must not be edited without a version bump.

---

## 3. Kill Switch

### 3.1 Rules

| Parameter | Value |
|-----------|-------|
| Maximum iterations | 20 |
| Minimum recall improvement per iteration | 3 percentage points absolute on val recall |

- An iteration that does not improve val recall by ≥ 3pp **triggers the kill switch**.
- The kill switch halts further iteration. The best-performing model advances to B1 evaluation.
- "Best-performing" is determined by val recall, with precision as a tiebreaker at equal recall.
- All kill switch verdicts are recorded in `experiments/registry.csv`.

### 3.2 Generalization and Benchmark Warnings

| Warning | Condition |
|---------|-----------|
| Overfitting warning | `recall_train − recall_val > 0.10` |
| Benchmark gap warning | `recall_val − benchmark_recall > 0.05` |

Warnings do not halt iteration but must be logged and addressed.

---

## 4. Principles

### 4.1 Boring Models First

Model complexity escalates only when simpler models are exhausted and the exhaustion is documented:

1. Logistic regression (mandatory baseline — `config/models/logistic_baseline.yaml`)
2. Random Forest
3. Gradient boosting (LightGBM or XGBoost)
4. Ensembles or deep learning — only if all above are exhausted and the justification is written in `experiments/iteration_log.md`

### 4.2 Recall-Prioritized

Primary metric: **recall**. A missed fatal crash is categorically worse than a false alarm. Precision is secondary. Threshold selection must be explicitly justified in each iteration's `notes.md`.

### 4.3 One Change Per Iteration

Each iteration changes **exactly one variable**: a feature set, a model hyperparameter, a preprocessing step, or a decision threshold. Multi-variable changes are not permitted.

### 4.4 Feature Acceptance: 3-Gate Test

Every new feature must pass all three gates before entering the pipeline (see `config/features.yaml`):

1. **Availability:** present in ≥ 95% of training rows
2. **Signal:** mutual info > 0.01 OR Spearman correlation > 0.05 with target
3. **Incremental gain:** adding the feature improves val recall by ≥ 0.005

### 4.5 Discipline Over Creativity

No novel architectures or unsanctioned techniques without prior exhaustion of simpler alternatives. Complexity must be justified in writing before implementation.

---

## 5. Audit Trail Design

### 5.1 experiments/registry.csv

Machine-readable index of all iterations. One row per iteration. Columns:

`iter_id, date, model, change_description, hypothesis, recall_train, recall_val, precision_val, recall_delta, generalization_gap, benchmark_set_used, benchmark_recall, accepted, kill_switch_verdict, notes`

### 5.2 experiments/iteration_log.md

Human-readable master narrative. For each iteration: what changed, why, hypothesis, qualitative result, decision. Metrics are not repeated here.

### 5.3 experiments/benchmark_audit.jsonl

Append-only audit trail for benchmark evaluations. Each line is a JSON object:
`{"date": "...", "model": "...", "split": "...", "metrics": {...}, "purpose": "..."}`

Written exclusively by `src/tracking/tracker.py`. Never edited manually.

**Note:** tracker.py is not yet implemented. This section describes the contract it must fulfill once written.

### 5.4 experiments/iter_NNN/

One directory per iteration containing:

| File | Contents |
|------|----------|
| `config_snapshot.yaml` | Full reproducibility record — configs, data files, hashes, git SHA |
| `metrics.json` | Recall, precision, F1, support, generalization gap, benchmark gap |
| `notes.md` | Qualitative observations, decisions, anomalies |

### 5.5 Metrics Schema

See `experiments/iter_001/metrics.json` for the canonical schema. All iteration metric files must conform to `_schema_version: "1.0"`.

### 5.6 Notebook Rule

Notebooks in `notebooks/` are for exploration only. Any finding must be re-implemented in `src/` before it influences the pipeline. Notebooks are never imported by `src/` code.
