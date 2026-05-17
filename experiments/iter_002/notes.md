# iter_002 — Threshold Adjustment (0.50 → 0.52)

**Date:** 2026-05-02

**Change:** Classification threshold raised from 0.50 to 0.52. Model unchanged (same
LogReg weights as iter_001). Best feasible threshold from sweep recorded in
`threshold_sweep.csv`.

**Hypothesis:** Raising threshold slightly would reduce flag rate below 30% while
maintaining fatal recall above 0.50. The model may have sufficient signal but a
suboptimal decision boundary.

## Result

| Metric | iter_002 val | iter_001 val (baseline) | Change |
|---|---|---|---|
| Fatal recall | 0.456 | 0.508 | −0.052 |
| Combined recall | 0.5378 | 0.5688 | −0.031 |
| Precision | 0.0551 | 0.0521 | +0.003 |
| Flag rate | 0.284 | 0.317 | −0.033 |
| Generalization gap | 0.0595 | 0.0642 | — |

Threshold: 0.52

## Decision

**REJECTED** — fatal recall dropped from 0.508 to 0.456, below the 0.50 hard floor.

No feasible threshold exists that satisfies both constraints (fatal recall ≥ 0.50 AND
flag rate ≤ 0.30) simultaneously with the current feature set.

## Key Insight

The tension between flag rate and fatal recall cannot be resolved by threshold adjustment
alone. The model needs features that improve fatal window discrimination. Next iteration
must target feature engineering, not threshold tuning.

*Source: registry.csv, iteration_log.md, metrics.json, threshold_sweep.csv*
