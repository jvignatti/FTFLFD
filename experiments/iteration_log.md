# FTFLFD — Iteration Log

Master narrative log. Each entry records: what changed, why it changed, the hypothesis, and the qualitative result.

Quantitative metrics live in `experiments/iter_NNN/metrics.json` — do not duplicate them here.

## Logging Rules

1. One entry per iteration. No exceptions.
2. Entries are append-only. Never edit past entries; correct via a new entry that references the original.
3. Every entry must answer: what changed, why, what was the hypothesis, what actually happened.
4. Failures are entries too. Rejected changes still get logged with the rejection rationale.
5. If iteration includes a benchmark run, note it explicitly with the audit warning: "for audit only — no tuning decisions based on this result."

## Entry Template

Copy this template for every new iteration:

### iter_NNN — YYYY-MM-DD

**Change:** [one sentence describing the single change made]

**Hypothesis:** [why we expected this change to help — be specific]

**Result:** [qualitative outcome — did it work? did it surprise us? what did we learn?]

**Decision:** [Accepted / Rejected]

**Notes:** [optional — geographic patterns, feature behavior, anything worth remembering]

---

## Iterations

### iter_001 — 2026-05-02

**Change:** Baseline logistic regression with 8 features (4 segment rates, 3 calendar, 1 segment type)

**Hypothesis:** Historical crash rates per segment combined with seasonal indicators should predict above random chance. This validates the pipeline, not the model.

**Result:** Pipeline validated end-to-end. Model beats random chance on all metrics. Fatal recall at 0.508 on validation is barely above the 0.50 hard stop floor — this is the top priority for iteration 002. Flag rate at 31.7% slightly exceeds the 30% maximum. Generalization gap 0.064 is healthy. segment_injury_rate dominates feature importance by a large margin (1.374 vs next highest 0.172).

**Decision:** Accepted

**Fatal FN review:** 246 fatal windows missed on validation (500 total, 254 caught). Not reviewed individually at baseline — individual review begins at iteration 002.

**Notes:** 41,376 val observations come from segments not seen in training (rates set to 0). This is expected — new crash locations emerge over time. The model relies on calendar features alone for these segments. This is a realistic simulation of deployment conditions.

---