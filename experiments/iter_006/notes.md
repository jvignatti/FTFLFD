# iter_006 — RF class_weight balanced → {0:1, 1:10}

**Date:** 2026-05-10

**Change:** RandomForest class_weight changed from "balanced" (~71x minority weight)
to custom {0:1, 1:10}. All other parameters unchanged (max_depth=6, 200 trees,
min_samples_leaf=50).

**Hypothesis:** Balanced weighting on 1:71 imbalance gives a 71x penalty for missed
positives, causing the model to flag nearly half of all windows. Reducing to 10x
should bring flag rate into compliance (≤ 0.30) while preserving sufficient recall.

## Result

| Metric | iter_006 val | iter_005 val | Change |
|---|---|---|---|
| Fatal recall | 0.102 | 0.576 | **−0.474** |
| Combined recall | 0.2494 | 0.6962 | −0.447 |
| Precision | 0.1385 | 0.0418 | +0.097 |
| Flag rate | 0.0524 | 0.4843 | −0.432 |
| Generalization gap | −0.0021 | 0.1055 | — |

Threshold: 0.5

## Decision

**REJECTED** — fatal recall collapsed to 0.102, far below the 0.50 hard floor.
Worse than the naive random baseline (~0.157 at comparable flag rate).

The model became extremely conservative: high precision but catches almost nothing.

## Key Insight

The relationship between class weight and recall is highly non-linear in Random Forest.
71x flags everything; 10x flags nothing. The feasible weight range lies between 20x–50x.
Next iteration must sweep this range systematically to find (if any) a weight satisfying
all constraints simultaneously.

*Source: registry.csv, iteration_log.md, metrics.json, config_snapshot.yaml*
