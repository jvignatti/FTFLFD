# Open Decisions

**Document:** `docs/technical/open_decisions.md`
**Created:** 2026-05-16
**Purpose:** Living tracker for unresolved methodological decisions. Each entry
must be resolved before dependent modeling work can proceed. Entries are closed
(not deleted) when resolved.

---

## OD-001 — AADT Imputation Strategy

**Status:** OPEN
**Blocker for:** AADT feature gate testing (iter_009 or equivalent)
**Created:** 2026-05-16 (from 07_aadt_coverage_diagnosis.md Section 6, Decision 1)

**Decision required:**
AADT coverage on the Phase 1 training set is 56.79% overall, 69.96% fatal. Records
without AADT fall into two categories:
- Route not in AADT at all (39.52% of training records) — primarily local/town roads
- Route in AADT but milepoint falls in a surveyed gap (53.72% of training records)

Three options:
1. **Impute** — assign median AADT for the segment's road group to unmatched records
2. **Exclude** — drop unmatched records from training (reduce training set to ~56K rows)
3. **Reject** — treat AADT as failing Gate 1 and abandon segment_aadt as a feature

Note: FunctionalClass is not affected by this decision — it is sourced from the road
centerline layer (98.72% coverage, Gate 1 PASS). See OD-002.

**Evidence available:**
- Phase 1 coverage numbers: `docs/technical/07_aadt_coverage_diagnosis.md` Section 4.2
- Missingness mechanism breakdown: `notebooks/diagnostic/aadt_coverage_analysis.py`
- Fatal coverage on state-maintained roads is highest (85–91%) — where fatals concentrate

**Dependencies:** OD-003 (feature testing order) blocked on this decision.

---

## OD-002 — FunctionalClass Source

**Status:** EFFECTIVELY RESOLVED — use road centerline
**Blocker for:** functional_class gate testing
**Created:** 2026-05-16 (from 07_aadt_coverage_diagnosis.md Section 6, Decision 2)

**Decision:** Whether to source FunctionalClass from the VTrans road centerline
layer (independent source) or from the AADT join (dependent on AADT coverage).

**Resolution:** Road centerline (`data/raw/road_centerline.csv`, 78,876 segments)
provides 98.72% overall coverage and 99.43% fatal coverage — Gate 1 PASS via
independent source. Gate 2 and Gate 3 gate testing is underway
(see `notebooks/diagnostic/gate_test_functional_class.py`).

**Action required:** Commit Gate 2/Gate 3 results to a formal experiment artifact
when testing is complete.

---

## OD-003 — Feature Testing Order for AADT-Derived Features

**Status:** PARTIALLY BLOCKED
**Blocker for:** Iterations after functional_class gate test
**Created:** 2026-05-16

**Decision required:**
After functional_class gate test completes, the testing order for remaining features:
- `segment_aadt` (raw traffic volume)
- `segment_crash_rate_per_vmt` (crash rate normalized by AADT)
- `is_divided` (divided roadway flag)

This order is blocked on OD-001 (AADT imputation strategy), since segment_aadt and
rate-per-vmt features require resolving how to handle the 43% of records with no AADT.

**Current state:** functional_class is being tested first (independent of AADT join).
AADT-dependent features cannot be ordered until OD-001 is resolved.

---

## OD-004 — Gate 1 Threshold Scope

**Status:** OPEN
**Blocker for:** Final gate decision on AADT-derived features
**Created:** 2026-05-16 (from 07_aadt_coverage_diagnosis.md Section 6, Decision 4)

**Decision required:**
Gate 1 as written in CLAUDE.md: "Feature must be available for at least 95% of
training rows."

For AADT-derived features:
- Overall coverage: 56.79% (fails Gate 1 by 38 pp)
- Fatal record coverage: 69.96% (fails Gate 1 by 25 pp)

If Gate 1 is evaluated on **fatal records only**, the interpretation changes:
the model's signal comes disproportionately from fatal windows, and 70% fatal
coverage may be acceptable if imputation handles the gap.

**Arguments for all-records threshold:**
- Consistency — same rule applies to every feature
- PDO windows are real training signal (negative class definition)
- Relaxing the threshold sets a precedent

**Arguments for fatal-records threshold:**
- The model's purpose is predicting fatal locations
- PDO missingness is heavily concentrated on local roads (low fatal risk anyway)
- 70% fatal coverage with imputation may outperform lower-coverage features

**Resolution path:** Requires explicit discussion and documented decision before
any AADT feature gate result is accepted or rejected.

---

## OD-005 — Create experiments/iter_007/ Directory

**Status:** OPEN
**Blocker for:** Nothing (audit/completeness question only)
**Created:** 2026-05-16

**Decision required:**
iter_007 was a class weight sweep (20x, 25x, 30x, 35x, 40x, 50x) for RandomForest.
No model was accepted. The registry records the outcome but no experiment directory
exists. Per-weight sweep metrics (individual recall/flag_rate for each of the 6 weights)
are not preserved anywhere on disk.

**Options:**
1. Create `experiments/iter_007/` with a metrics.json capturing the per-weight sweep
   results, for completeness and future reference
2. Leave absent — treat the sweep as a single rejected iteration fully documented in
   registry.csv and iteration_log.md

---

## OD-006 — functional_class Gate 2 and Gate 3 Results

**Status:** OPEN — results not yet committed
**Blocker for:** iter_009 (adding functional_class to feature set, if accepted)
**Created:** 2026-05-16

**Decision required:**
`notebooks/diagnostic/gate_test_functional_class.py` exists and is ready to run.
Gate 1 already passed (98.72% all, 99.43% fatal, from functional_class_coverage.py).
Gate 2 and Gate 3 results must be:
1. Run against Phase 1 train/val featured parquets
2. Committed to a formal artifact (experiment directory or iteration_log.md entry)
3. Recorded in features.yaml if Gate 3 passes

The script uses the correct Phase 1 baseline (BASELINE_FATAL_RECALL=0.508) and
reads Phase 1 parquets. It is ready to run.

---

## OD-007 — B2 gap_warning Disposition

**Status:** ACKNOWLEDGED — see protocol_deviations.md PD-003
**Blocker for:** Nothing (audit-only result)
**Created:** 2026-05-16

iter_001 on B2 (2023 data): fatal_recall=0.4625, benchmark_gap=0.0881.
Gap warning fires. Cannot influence modeling decisions per protocol.
Full disposition documented in `docs/technical/protocol_deviations.md` PD-003.
Root cause: temporal drift documented in `docs/technical/06_drift_diagnosis.md`.
