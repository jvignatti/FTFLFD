# iter_005 — RF max_depth 10 → 6

**Date:** 2026-05-10

**Change:** RandomForest max_depth reduced from 10 to 6. All other parameters
unchanged (200 trees, min_samples_leaf=50, class_weight=balanced).

**Hypothesis:** Shallower trees will reduce memorization, lower the generalization
gap, and reduce the flag rate by preventing the model from fitting noisy deep splits.

## Result

| Metric | iter_005 val | iter_004 val | Change |
|---|---|---|---|
| Fatal recall | 0.576 | 0.576 | 0.000 |
| Combined recall | 0.6962 | 0.6909 | +0.005 |
| Precision | 0.0418 | 0.0424 | −0.001 |
| Flag rate | 0.4843 | 0.4743 | +0.010 |
| Generalization gap | 0.1055 | 0.1114 | −0.006 |

Threshold: 0.5

## Decision

**REJECTED** — depth is not the cause of over-flagging. Flag rate increased slightly.
Generalization gap marginally improved but still exceeds the 0.10 threshold.
Fatal recall unchanged. No constraint brought into compliance.

## Key Insight

The model makes all meaningful decisions within the first 6 tree levels — depth
reduction had no material effect. The over-flagging is driven by class_weight="balanced"
on a 1:71 imbalance, not by tree depth. Next iteration must directly address class
weight strategy.

*Source: registry.csv, iteration_log.md, metrics.json, config_snapshot.yaml*
