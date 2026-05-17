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

**Status:** REOPENED — road centerline join invalid
**Blocker for:** functional_class gate testing
**Created:** 2026-05-16 (from 07_aadt_coverage_diagnosis.md Section 6, Decision 2)
**Reopened:** 2026-05-16

**Decision:** Whether to source FunctionalClass from the VTrans road centerline
layer (independent source) or from the AADT join (dependent on AADT coverage).

**Prior resolution (invalidated):** Road centerline was selected as the source
(98.72% overall coverage, 99.43% fatal). Gate 2 and Gate 3 testing was pending.

**Invalidation:** The road centerline join was diagnosed as invalid. The centerline
uses TWN_LR encoding; the crash data uses LRSNUMBER encoding. These are not
equivalent formats. The join produces coverage figures that cannot be trusted.
All prior Gate 1 figures derived from the road centerline source are void.

**New decision:** FunctionalClass will be sourced from the AADT files
(`data/raw/aadt_limited.csv` + `data/raw/aadt_other.csv`) via the validated
StandardRouteCode interval join. This is the same join used for segment_aadt.

**Coverage via AADT source:**
- Overall: 56.79% (same as segment_aadt — FunctionalClass resides in the same records)
- Fatal records: 69.96%

**Gate 1 status:** Depends on OD-004 resolution. Under the current 95% all-records
threshold, FunctionalClass fails Gate 1 by 38 pp.

**Dependencies:** OD-004 must be resolved before Gate 1 verdict is issued.
Gate test script has been rewritten to use AADT join (see OD-006).

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

**Status:** ACTIVE BLOCKER — explicit sign-off required before any AADT feature gate result is accepted
**Blocker for:** Gate 1 verdict on functional_class, segment_aadt, segment_crash_rate_per_vmt, is_divided
**Created:** 2026-05-16 (from 07_aadt_coverage_diagnosis.md Section 6, Decision 4)

**Decision required:**
Gate 1 as written in CLAUDE.md: "Feature must be available for at least 95% of
training rows."

For AADT-derived features:
- Overall coverage: 56.79% (fails Gate 1 by 38 pp)
- Fatal record coverage: 69.96% (fails Gate 1 by 25 pp)
- State-system fatal coverage: 85–91% (best case — still below 95%)

**Three options:**

Option A — Keep Gate 1 as written (95% of all training rows):
- FunctionalClass / segment_aadt fail at 56.79% (38 pp short)
- Consistent with CLAUDE.md as written; no precedent set
- Requires either imputation (OD-001) to raise coverage, or rejection of AADT features

Option B — Evaluate Gate 1 on fatal records only (95% of fatal training rows):
- FunctionalClass / segment_aadt fail at 69.96% (25 pp short)
- More aligned with the model's purpose (predicting fatal locations)
- Still requires imputation or rejection — PDO missingness alone does not close the gap
- Sets a precedent for class-conditional coverage thresholds; must be documented as such

Option C — Restrict Gate 1 scope to state-system roads (~85–91% coverage):
- State-system roads have highest AADT coverage and highest fatal concentration
- Coverage within state-system scope may approach 85–91% — still potentially below 95%
- Would require the model's operational scope to be explicitly locked to state-system roads
- Most complex option; requires a separate scope decision before feature work can proceed

**Arguments for all-records threshold (Option A):**
- Consistency — same rule applies to every feature
- PDO windows are real training signal (negative class definition)
- Relaxing the threshold sets a precedent

**Arguments for fatal-records threshold (Option B):**
- The model's purpose is predicting fatal locations
- PDO missingness is heavily concentrated on local roads (low fatal risk anyway)
- 70% fatal coverage with imputation may outperform lower-coverage features

**Resolution path:** Requires explicit sign-off — not inferred from context. This
decision must be recorded here before gate_test_functional_class.py is run and
before any Gate 1 verdict is issued for any AADT-derived feature.

**Interaction:** This decision also interacts with OD-001 (AADT imputation strategy).
If Option A or B is chosen, imputation may be required to satisfy the 95% threshold.

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

**Status:** BLOCKED — OD-004 unresolved; gate test must not be run
**Blocker for:** iter_009 (adding functional_class to feature set, if accepted)
**Created:** 2026-05-16

**Current state:**
`notebooks/diagnostic/gate_test_functional_class.py` has been rewritten to use the
AADT interval join (StandardRouteCode + BeginMM/EndMM filter) instead of the invalid
road centerline join. Gate 1 is reported dynamically; verdict is withheld pending
OD-004 resolution. Gate 2 and Gate 3 code is present but results must not be
interpreted until Gate 1 is resolved.

All prior Gate 1 figures for FunctionalClass (98.72% overall, 99.43% fatal) derived
from the road centerline source are invalidated and must be disregarded.

**Blocking conditions:**
1. OD-004 must be resolved and signed off before Gate 1 verdict is accepted or rejected.
2. Gate test script must not be run until OD-004 is resolved.

**When unblocked, Gate 2 and Gate 3 results must be:**
1. Run against Phase 1 train/val featured parquets
2. Committed to a formal artifact (experiment directory or iteration_log.md entry)
3. Recorded in config/features.yaml if Gate 3 passes

---

## OD-007 — B2 gap_warning Disposition

**Status:** ACKNOWLEDGED — see protocol_deviations.md PD-003
**Blocker for:** Nothing (audit-only result)
**Created:** 2026-05-16

iter_001 on B2 (2023 data): fatal_recall=0.4625, benchmark_gap=0.0881.
Gap warning fires. Cannot influence modeling decisions per protocol.
Full disposition documented in `docs/technical/protocol_deviations.md` PD-003.
Root cause: temporal drift documented in `docs/technical/06_drift_diagnosis.md`.
