# Open Decisions

**Document:** `docs/technical/open_decisions.md`
**Created:** 2026-05-16
**Purpose:** Living tracker for unresolved methodological decisions. Each entry
must be resolved before dependent modeling work can proceed. Entries are closed
(not deleted) when resolved.

---

## OD-001 — AADT Imputation Strategy

**Status:** SUPERSEDED — see OD-004 resolution
**Blocker for:** N/A
**Created:** 2026-05-16

**Original decision required:**
AADT coverage on the Phase 1 training set is 56.79% overall, 69.96% fatal.
Three options were under consideration: impute, exclude, or reject AADT features.

**Superseded 2026-05-19:** OD-004 resolved via TomTom coordinate join. AADT
imputation is no longer required — road-attribute features are sourced from
TomTom API at 97%+ coverage. This decision is closed without a formal resolution.

---

## OD-002 — FunctionalClass Source

**Status:** SUPERSEDED — see OD-004 resolution
**Blocker for:** N/A
**Created:** 2026-05-16
**Reopened:** 2026-05-16
**Superseded:** 2026-05-19

**History:**
1. Road centerline (Layer 39) selected as source — invalidated due to TWN_LR /
   LRSNUMBER format mismatch. Coverage figures of 98.72% were a fan-out artifact.
2. AADT files selected as fallback — coverage 56.79% overall, 69.96% fatal.
   Failed Gate 1 at 95% threshold.
3. Full VTrans API survey conducted — no public layer resolves the LRS crosswalk
   at ≥95% coverage on the full training set.

**Resolution:** FunctionalClass will be sourced from TomTom road_use field via
coordinate join. LRS crosswalk problem bypassed entirely. See OD-004.

---

## OD-003 — Feature Testing Order for AADT-Derived Features

**Status:** SUPERSEDED — see OD-004 resolution
**Blocker for:** N/A
**Created:** 2026-05-16

**Superseded 2026-05-19:** AADT-derived features (segment_aadt,
crash_rate_per_vmt, is_divided) are deprioritized. Road-attribute features
are now sourced from TomTom API. Feature testing order will be:
1. speed_limit (TomTom) — gate test first
2. road_use (TomTom) — gate test second
3. AADT-derived features — revisit after TomTom features are resolved,
   if additional signal is needed

---

## OD-004 — Gate 1 Threshold Scope

**Status:** RESOLVED — 2026-05-19
**Blocker for:** N/A — all dependent decisions resolved
**Created:** 2026-05-16

**Resolution:** Coordinate-based join via TomTom Search API
(`/search/2/reverseGeocode`) bypasses the LRS coverage problem entirely.
Crash records have 99.17% valid coordinate coverage (99.82% fatal).

TomTom sample test results (100-record stratified sample, all road types):
- speed_limit coverage: 97% overall, 100% fatal
- road_use coverage: 100% overall, 100% fatal
- HTTP 200: 100/100 requests

Gate 1 passes at the 95% threshold on the full training set without scope
restriction, imputation of structural missingness, or threshold exception.
Options A, B, C, and D are all superseded.

See `notebooks/scratch/tomtom_sample_test.py` for full sample results.
See `notebooks/scratch/tomtom_batch_test.py` for batch endpoint validation.

---

## OD-005 — Create experiments/iter_007/ Directory

**Status:** OPEN
**Blocker for:** Nothing (audit/completeness question only)
**Created:** 2026-05-16

**Decision required:**
iter_007 was a class weight sweep (20x, 25x, 30x, 35x, 40x, 50x) for
RandomForest. No model was accepted. The registry records the outcome but no
experiment directory exists. Per-weight sweep metrics (individual recall/flag_rate
for each of the 6 weights) are not preserved anywhere on disk.

**Options:**
1. Create `experiments/iter_007/` with a metrics.json capturing the per-weight
   sweep results, for completeness and future reference.
2. Leave absent — treat the sweep as a single rejected iteration fully documented
   in registry.csv and iteration_log.md.

---

## OD-006 — Road-Attribute Feature Gate Tests

**Status:** UNBLOCKED — OD-004 resolved 2026-05-19
**Blocker for:** iter_009 (first road-attribute feature addition)
**Created:** 2026-05-16
**Updated:** 2026-05-19

**Current state:**
OD-004 resolved via TomTom coordinate join. AADT-based gate test script
(`gate_test_functional_class.py`) is superseded. New gate test scripts required:

1. `gate_test_speed_limit.py` — gate test for speed_limit feature (TomTom)
2. `gate_test_road_use.py` — gate test for road_use feature (TomTom)

One script per feature per CLAUDE.md one-change-per-iteration rule.

**Prerequisites before gate testing:**
1. `data/raw/pull_tomtom.py` — full 109,695 record pull via batch API
2. Speed limit and road_use values stored in `data/raw/tomtom_features.csv`
3. Gate test scripts written and reviewed before running

**When gate tests complete, results must be:**
1. Run against Phase 1 train/val featured parquets
2. Committed to a formal artifact (experiment directory or iteration_log.md)
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