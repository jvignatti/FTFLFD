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

(Entries appear below as iterations occur.)
