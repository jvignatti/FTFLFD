# iter_008 — Model Advancement: RandomForest → LightGBM

**Date:** 2026-05-11

**Change:** Model advanced from RandomForestClassifier to LGBMClassifier
(300 trees, max_depth=6, is_unbalance=True, learning_rate=0.05,
num_leaves=31, min_child_samples=100).
RF was exhausted: depth tuning (iter_005), class weight range (iter_006), full
sweep 20x–50x (iter_007) — all rejected. No feasible RF configuration exists
on the current 8 features.

**Hypothesis:** Gradient boosting builds trees sequentially, each correcting the
previous. LightGBM's is_unbalance flag may handle the 1:71 imbalance more gracefully
than RF's class_weight, producing a model that satisfies all four constraints.

## Result

| Metric | iter_008 val | iter_001 val (baseline) | Change |
|---|---|---|---|
| Fatal recall | 0.58 | 0.508 | +0.072 |
| Combined recall | 0.6861 | 0.5688 | +0.117 |
| Precision | 0.0426 | 0.0521 | −0.010 |
| Flag rate | 0.4688 | 0.317 | +0.152 |
| Generalization gap | 0.1107 | 0.0642 | +0.046 |

Threshold: 0.5

## Decision

**REJECTED** — same over-flagging profile as RF iter_004. Three constraints violated:

- Flag rate 0.4688 >> 0.30 maximum: **FAIL**
- Precision 0.0426 < 0.05 floor: **FAIL**
- Generalization gap 0.1107 > 0.10: **FAIL** (overfitting warning)

## Feature Importance Shift

LightGBM uses month (1,872 splits) and segment_crash_rate (2,189) more heavily than
RF. segment_injury_rate no longer dominant. Different internal patterns found, same
constraint failure.

## Exhaustion Conclusion

Three model classes tested across 8 iterations:
LogisticRegression (iter_001–003), RandomForest (iter_004–007), LightGBM (iter_008).

**LogReg iter_001 is the only model satisfying all constraints.** It is confirmed as
the Phase 1 MVP production model. All tree-based models over-flag on the current 8
crash-history features. Structural features (AADT, functional class) are required
to improve tree-based model performance.

*Source: registry.csv, iteration_log.md, metrics.json, config_snapshot.yaml, feature_importance.csv*
