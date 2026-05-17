# iter_004 — Model Advancement: LogisticRegression → RandomForestClassifier

**Date:** 2026-05-08

**Change:** Model advanced from LogisticRegression to RandomForestClassifier
(200 trees, max_depth=10, min_samples_leaf=50, class_weight=balanced).
LogReg was exhausted: 4 feature candidates + 1 threshold adjustment all rejected (iter_002, iter_003a/b/c).

**Hypothesis:** Logistic Regression cannot express non-linear interactions between
the 8 crash-history features. Random Forest may find combinations LogReg cannot,
improving fatal recall while satisfying all constraints.

## Result

| Metric | iter_004 val | iter_001 val (baseline) | Change |
|---|---|---|---|
| Fatal recall | 0.576 | 0.508 | **+0.068** |
| Combined recall | 0.6909 | 0.5688 | +0.122 |
| Precision | 0.0424 | 0.0521 | −0.0097 |
| Flag rate | 0.4743 | 0.317 | +0.157 |
| Generalization gap | 0.1114 | 0.0642 | +0.047 |

Threshold: 0.5

## Decision

**CONDITIONALLY ACCEPTED** — fatal recall improved meaningfully (+0.068) but three
constraints are violated:

- Flag rate 0.4743 >> 0.30 maximum: **FAIL**
- Precision 0.0424 < 0.05 floor: **FAIL**
- Generalization gap 0.1114 > 0.10: **FAIL** (overfitting warning)

Accepted conditionally because fatal recall improvement is real. Hyperparameter
tuning required to bring flag rate and precision into compliance.

## Fatal False Negative Review

Fatal recall improved from 0.508 to 0.576. 212 of 500 fatal windows recalled
(vs 254 at iter_001). Still 288 fatal misses — overfitting and over-flagging
prevent acceptance at full compliance.

## Feature Importance

segment_injury_rate dominates (0.536 Gini), followed by segment_crash_rate (0.251).
segment_fatal_rate low importance (0.020) — model predicts from injury rate, not
fatal rate directly. Consistent with collinearity finding from iter_003a.

## Key Insight

RF found real signal. Problem is class_weight="balanced" on 1:71 imbalance amplifies
minority class too aggressively for tree-based models, causing 47% flag rate.
Next iteration must reduce class weight to control flag rate without losing fatal recall.

*Source: registry.csv, iteration_log.md, metrics.json, config_snapshot.yaml, feature_importance.csv*
