# CLAUDE.md — FTFLFD Project Instructions

## What This Project Is

FTFLFD is a proactive traffic fatality and serious injury prediction system for Vermont. It uses 16 years of crash data (2010–2026, 179,282 records) to predict which road locations have high probability of fatal or serious injury crashes within a 30-day window.

This is a methodology-driven ML system. Discipline over creativity. Boring models first. A system that learns to improve predictions, not just predict events.

## Core Principles (Non-Negotiable)

1. **One change per iteration.** Never change two things at once. Every iteration modifies exactly one variable — one feature, one hyperparameter, one preprocessing step, OR one threshold. Never a combination.
2. **Recall-prioritized with precision floor.** False negatives (missed fatalities) are unacceptable. False positives (over-flagging) are acceptable within operational limits. Primary metric is recall. However, precision must not fall below 0.05 (5%) — a model that flags everything is useless. Maximum flag rate must remain operationally viable (no more than 30% of prediction units flagged as high-risk per cycle).
3. **Boring models first.** Progression: Logistic Regression → Random Forest → Gradient Boosting → Ensembles/Deep Learning. Never skip levels. Never advance without documented exhaustion of simpler models (see Model Advancement Criteria below).
4. **Zero data leakage.** No feature, encoding, or aggregate may use information from the future. Rolling windows cannot cross split boundaries. Encodings are fit on training only. See Leakage Rules for full specification.
5. **Frozen benchmarks.** Benchmark sets are never used for tuning. B1 every 5 iterations (log only). B2 every 10 iterations (log only). B3 once per phase. B4 once ever (final test). Benchmark results must NEVER influence any modeling, feature, or threshold decision. They exist solely as an audit trail.
6. **Log like a scientist.** Every iteration logs: what changed, why, hypothesis, result, random seed, dataset hash, feature set version. Failures are entries too.
7. **Full reproducibility.** Every experiment must be fully reproducible given: the data (identified by SHA256 hash), the config files (identified by version), the feature set (identified by version), and the random seed. If any of these are missing from an experiment log, the experiment is invalid.

## Hard Blockers (Cannot Proceed Without These)

The following must be completed and locked before Iteration 001 can begin:

1. **Prediction unit must be defined and locked.** The event contract (Section 1.3) requires an explicit spatial unit (town, road segment, grid cell, corridor) before any model training. This is appended to docs/technical/01_project_design.md as a formal amendment. No code in src/training/ may execute until this is locked.
2. **SHA256 hashes must be registered for all datasets.** Every data file used in training, validation, or benchmarking must have its SHA256 hash recorded in data/raw/file_hashes.json before it is used. Unregistered data cannot enter the pipeline.
3. **Split logic must be programmatically enforced and tested.** The split boundaries in config/splits.yaml must be enforced by code in src/training/splitter.py with automated assertions (not manual verification). tests/test_splitter.py must pass before any training run.
4. **Virtual environment must be active with pinned dependencies.** All dependencies must be installed from requirements.txt in an isolated Python 3.11 environment.

## Event Contract (v1.0 — LOCKED)

- **Target column:** `InjuryType` (this is the canonical column name across the entire project — do not use `Crash Type`, `CrashType`, `severity`, or any other variant)
- **Positive class (1):** InjuryType = "Fatal" OR InjuryType = "Injury"
- **Negative class (0):** InjuryType = "Property Damage Only"
- **Excluded from training:** InjuryType = blank, null, or unknown (dropped entirely, never labeled as 0)
- **Prediction window:** 30 days
- **Prediction unit:** BLOCKER — must be locked before Iteration 001 (see docs/technical/01_project_design.md Section 1.3)
- **Priority:** Maximize recall with precision floor of 0.05 and maximum flag rate of 30%

## Split Strategy (v1.1)

| Set | Period | Use |
|---|---|---|
| Train | 2010–2019 | Model fitting (used freely) |
| Val | 2020–2021 | Tuning (used freely, includes COVID disruption) |
| B1 | 2022 | Blind eval every 5 iterations |
| B2 | 2023 | Blind eval every 10 iterations |
| B3 | 2024–2025 | Blind eval once per phase |
| B4 | 2026 | Final test. Once. Ever. |

- 30-day gap between every split boundary (rows in gaps are dropped)
- data_era flag required: early (2010–2014), historical (2015–2019), modern (2020+)
- See config/splits.yaml for exact dates
- Split boundaries must be enforced programmatically in src/training/splitter.py
- tests/test_splitter.py must assert: no date overlap, no ID overlap, gap windows correct, row counts reconcile

## Kill Switch (Adaptive)

- Maximum 20 iterations per phase
- Improvement is measured as absolute recall gain on validation set
- Adaptive rule:
  - Iterations 1–5: minimum 3% recall improvement per iteration (pipeline validation phase)
  - Iterations 6–15: minimum 1% recall improvement per iteration (refinement phase)
  - Iterations 16–20: any measurable improvement (>0.1%) is acceptable (diminishing returns phase)
- If threshold not met for 3 consecutive iterations → full stop, root cause analysis
- Generalization gap warning: train_recall - val_recall > 0.10
- Benchmark gap warning: val_recall - benchmark_recall > 0.05
- All kill switch verdicts recorded in experiments/registry.csv

## Model Advancement Criteria

A model level is considered exhausted when ALL of the following are true:

| Criterion | Requirement |
|---|---|
| Feature exhaustion | All features passing the 3-gate test have been tried |
| Hyperparameter search | At least 3 meaningful hyperparameter variations tested |
| Plateau evidence | Val recall has not improved by > 0.5% in last 3 accepted iterations |
| Documentation | Exhaustion rationale written in experiments/iteration_log.md |

Only after documenting exhaustion may the next model level be attempted:
1. Logistic Regression (mandatory baseline)
2. Random Forest
3. Gradient Boosting (LightGBM or XGBoost)
4. Ensembles or deep learning (requires written justification beyond exhaustion)

## Feature Rules

### 3-Gate Acceptance Test

Every new feature must pass all three gates before entering the pipeline:

| Gate | Test | Metric | Threshold |
|---|---|---|---|
| Gate 1: Availability | Feature present in training rows | Coverage % | ≥ 95% |
| Gate 2: Signal | Measured on training set only | Mutual information score with target | > 0.01 |
| Gate 3: Incremental Gain | Measured on validation set only | Recall improvement when feature is added | ≥ 0.005 |

- Gate 2 signal metric: mutual_info_classif from sklearn. Spearman correlation > 0.05 accepted as alternative for continuous features.
- Gate 3 evaluation: must be performed ONLY on the validation set. Never on training data (prevents overfitting to training distribution).
- See config/features.yaml for full rules and current active features
- Feature count limits: LogReg ≤ 20, RF ≤ 50, GBM ≤ 100

### Feature Set Versioning

Every change to the active feature set bumps the version in config/features.yaml. The version is logged in every experiment's config_snapshot.yaml.

## Leakage Rules (Comprehensive)

### Temporal Leakage
- No feature may use information from the future relative to the prediction date
- Rolling windows cannot cross split boundaries
- Lag features must not reach past the split boundary

### Target Leakage via Encoding
- Target encoding (mean target per category) must be fit on training folds only, never on the full training set
- Leave-one-out encoding or k-fold encoding within training is required if target encoding is used
- Frequency encoding is preferred over target encoding where possible

### Spatial Leakage
- If predicting at the road segment level, neighboring segments' outcomes cannot be used as features unless explicitly lagged in time
- Spatial aggregates (e.g., "incidents within 500m") must use only historical data, never concurrent or future events

### Global Statistics Leakage
- No global mean, median, or distribution computed on the full dataset may be used as a feature
- All statistics must be computed within the training set only and applied forward
- StandardScaler, MinMaxScaler, and similar transforms must be fit on training data only

### Enforcement
- src/leakage/checks.py must assert all rules above before every training run
- tests/test_leakage.py must cover each leakage type with explicit test cases

## Reproducibility Requirements

### Random Seeds
- Global random seed: 42 (defined in config/thresholds.yaml)
- Every model, every split, every sampling operation must use this seed or a deterministic derivative
- The seed used must be logged in every experiment's config_snapshot.yaml and metrics.json
- If a different seed is used for any reason, it must be documented with justification

### Dataset Lineage
- Every derived dataset (cleaned, feature-engineered, split) must document:
  - Which raw files it was derived from (by SHA256 hash)
  - What transformations were applied (by code version / git SHA)
  - When it was generated (timestamp)
- This lineage is recorded in each experiment's config_snapshot.yaml

### Environment
- Python 3.11 pinned in .python-version
- All dependencies pinned with version ranges in requirements.txt
- pip freeze output should be captured in each experiment directory as requirements_frozen.txt
- If the project moves to containerization, a Dockerfile will be added at the root

## Threshold Selection Strategy

Threshold selection (the probability cutoff for classifying a prediction unit as high-risk) is a modeling decision that must follow these rules:

1. **Default threshold:** 0.5 for baseline. Adjusted only after baseline is validated.
2. **Selection method:** Choose threshold that maximizes recall subject to the precision floor (≥ 0.05) and maximum flag rate (≤ 30%).
3. **Threshold is a single variable.** Changing the threshold counts as one iteration change. It cannot be combined with feature or model changes.
4. **Threshold must be logged** in each experiment's config_snapshot.yaml and metrics.json.
5. **Precision-recall curve** must be generated and stored in each experiment directory when threshold is adjusted.

## Class Imbalance Strategy

Fatal and serious injury crashes are rare events relative to the full dataset. The following strategies are required:

1. **Mandatory for all models:** Use `class_weight="balanced"` (or equivalent) as the default. This is non-negotiable for the baseline.
2. **Permitted alternatives (one at a time, as iteration changes):**
   - SMOTE oversampling (on training data only, never on validation)
   - Undersampling of majority class
   - Custom class weights derived from domain knowledge
3. **Prohibited:** Oversampling or any resampling applied to validation or benchmark sets.
4. **Monitoring:** Class distribution must be logged for every split in each experiment's config_snapshot.yaml.

## Calibration Requirement

Predicted probabilities must be meaningful, not just ordinal rankings:

1. **After baseline is validated:** Run calibration analysis (reliability diagram / Brier score) on validation set.
2. **If calibration is poor:** Apply Platt scaling or isotonic regression as a post-processing step (fit on validation set, evaluate on benchmark).
3. **Calibration is logged** in metrics.json under a dedicated "calibration" section.
4. **Why this matters:** Operationally, a "0.7 probability of fatal crash" must mean something real. Uncalibrated probabilities mislead decision-makers.

## Model Registry

Every trained model is registered with the following fields in experiments/iter_NNN/config_snapshot.yaml:

| Field | Description |
|---|---|
| iter_id | Iteration identifier (iter_001, iter_002, ...) |
| model_class | Model class name (e.g., LogisticRegression) |
| hyperparameters | Full parameter dict as passed to the model |
| random_seed | Seed used for this run |
| feature_set_version | Version from config/features.yaml |
| active_features | List of features used |
| splits_config_version | Version from config/splits.yaml |
| thresholds_config_version | Version from config/thresholds.yaml |
| dataset_hashes | SHA256 of every data file used |
| git_commit_sha | Exact code state |
| threshold | Classification threshold used |
| class_weight | Class weight strategy used |
| metrics | Full metrics dict (train, val, benchmark if run) |
| decision | Accepted / Rejected |
| predecessor | iter_id of the previous accepted iteration |

## Data Rules

- Raw data lives in data/raw/ (gitignored, local only)
- Yearly splits in data/raw/yearly/ (gitignored)
- SHA256 hashes of all data files in data/raw/file_hashes.json (tracked)
- NEVER commit raw crash records to GitHub
- data/public/ is for aggregated, non-sensitive outputs only
- 3,030 records lack coordinates → excluded from spatial models (see INGEST_NOTES.md Section 5.2)
- 123 ghost rows with null dates → dropped at ingestion (see INGEST_NOTES.md Section 5.5)
- Target column is always `InjuryType` — no synonyms, no aliases
- Every derived dataset must have documented lineage (source hashes, transformations, timestamp)

## Experiment Logging Requirements

Every iteration MUST log the following in experiments/iter_NNN/:

### config_snapshot.yaml (machine-readable)
- iter_id, date, model_class, hyperparameters
- random_seed
- dataset_hashes (SHA256 of every data file used)
- feature_set_version and active feature list
- splits_config_version, thresholds_config_version
- git commit SHA
- threshold used
- class_weight strategy
- class distribution per split (positive count, negative count, ratio)

### metrics.json (machine-readable)
- Train: recall, precision, f1, support_positive, support_total
- Val: recall, precision, f1, support_positive, support_total
- Benchmark (if run): set used, recall, precision, f1, audit_only: true
- Diagnostics: generalization_gap, benchmark_gap, overfitting_warning
- Calibration: brier_score (after baseline is validated)
- Threshold: value used, flag_rate, precision_at_threshold

### notes.md (human-readable)
- Change description
- Hypothesis
- Qualitative result
- Decision (Accepted / Rejected)
- Observations

### requirements_frozen.txt
- Output of pip freeze at time of run

## Project Structure
