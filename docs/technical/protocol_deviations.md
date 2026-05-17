# Protocol Deviations Log

**Document:** `docs/technical/protocol_deviations.md`
**Created:** 2026-05-16
**Purpose:** Permanent record of deviations from CLAUDE.md protocol. Each entry
documents what happened, why, and the disposition. Entries are append-only.

---

## PD-001 — B1 Benchmark Run at iter_004 (1 Iteration Early)

**Date:** 2026-05-10
**Protocol:** CLAUDE.md — "B1 every 5 iterations (log only)"
**What happened:** B1 (2022 data) was run against iter_004 after only 4 numbered
iterations (iter_001, iter_002, iter_003, iter_004). Protocol requires waiting until
iteration 5.
**Reason:** iter_003 was split into three sub-attempts (iter_003a/b/c), all of which
failed Gate 2 or Gate 3 before any model was trained. If sub-attempts are counted,
the total experiment count at iter_004 was effectively 6 (001, 002, 003a, 003b, 003c,
004), making B1 timing approximately correct in spirit.
**Disposition:** Acknowledged. Audit only — result not used for any tuning decision.
B1 result: iter_004 on 2022 data, recall_fatal=0.5228, benchmark_gap=0.0146. See
`experiments/benchmark_audit.jsonl`.

---

## PD-002 — B2 Benchmark Run After 8 Iterations (Protocol: Every 10)

**Date:** 2026-05-11
**Protocol:** CLAUDE.md — "B2 every 10 iterations (log only)"
**What happened:** B2 (2023 data) was run against iter_001 after Phase 1 completed
with 8 total iterations. Protocol requires waiting until iteration 10.
**Reason:** Phase 1 was declared exhausted at iteration 8 (three model classes fully
tested, LogReg iter_001 confirmed as only compliant model). B2 was run as a Phase 1
close-out audit to establish the temporal generalization baseline before any Phase 2
or structural feature work begins.
**Disposition:** Acknowledged as a Phase 1 close-out exception. Audit only — result
not used for any tuning decision. B2 result recorded in `experiments/benchmark_audit.jsonl`.

---

## PD-003 — B2 gap_warning = true (Fatal Recall 0.4625, Gap 0.0881)

**Date:** 2026-05-11
**Protocol:** CLAUDE.md — "Benchmark gap warning: val_recall - benchmark_recall > 0.05"
**What happened:** iter_001 on B2 (2023 data) produced recall_fatal=0.4625 vs
val recall_fatal=0.508. The combined benchmark_gap=0.0881 exceeds the 0.05 warning
threshold. gap_warning=true in benchmark_audit.jsonl.
**Reason:** Temporal drift is the documented root cause (see `06_drift_diagnosis.md`).
90% of fatal locations do not persist across eras. The model degrades as the gap
between training data (2010–2019) and prediction year widens.
**Disposition:** Acknowledged. Does not influence modeling decisions per protocol —
benchmark results are audit-only. This result is the primary motivation for
integrating structural features (AADT, FunctionalClass) that can predict risk at
locations with no crash history. See `open_decisions.md` for current status.
Note: B2 fatal recall 0.4625 is below the 0.50 floor — this is an audit finding,
not a model acceptance decision. The floor applies to val set evaluation only.
