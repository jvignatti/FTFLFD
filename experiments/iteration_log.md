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

### iter_002 — 2026-05-02

**Change:** Classification threshold adjusted from 0.50 to 0.52 (best feasible threshold from sweep)

**Hypothesis:** Raising threshold slightly would reduce flag rate below 30% while maintaining fatal recall above 0.50. The model may already have sufficient signal but the decision boundary is suboptimal.

**Result:** Hypothesis rejected. Flag rate improved (0.317 → 0.284) but fatal recall dropped from 0.508 to 0.456 — below the 0.50 hard stop floor. No feasible threshold exists that satisfies both constraints simultaneously. This confirms the problem is insufficient signal for fatal windows, not a suboptimal decision boundary.

**Decision:** Rejected

**Key insight:** The tension between flag rate and fatal recall cannot be resolved by threshold adjustment alone. The model needs features that specifically improve fatal window discrimination. Next iteration must target feature improvement, not model or threshold changes.

### iter_003 — 2026-05-02

**Change:** Add segment_impairment_ratio (proportion of segment's historical crashes involving any impairment)

**Hypothesis:** EDA confirmed impairment is present in 47.5% of fatal crashes vs 10.4% of injury crashes — a 4.5x overrepresentation. A segment's historical impairment ratio should help the model discriminate fatal-risk locations.

**Result:** Passed Gate 1 (coverage 1.0) and Gate 2 (Spearman 0.064). Failed Gate 3 — val recall dropped by 0.0019 instead of improving. The feature is collinear with existing segment rate features (crash_rate, injury_rate, fatal_rate). Segments with high impairment ratios already have high crash rates, so the impairment ratio adds no new information above what the model already knows.

**Decision:** Rejected

**Key insight:** The impairment signal is powerful at the crash level (outcome discriminator) but weak at the segment level (location discriminator). Segment-level aggregates of crash-level attributes are redundant when the model already has segment-level rates. Future features should target dimensions the model has zero access to currently — such as temporal trends, structural road characteristics, or cross-segment spatial patterns.

### iter_003a — 2026-05-02

**Change:** Add segment_impairment_ratio

**Hypothesis:** Impairment is 4.5x overrepresented in fatal crashes. Segment-level impairment ratio should improve fatal discrimination.

**Result:** Passed Gates 1-2. Failed Gate 3 (gain -0.0019). Collinear with existing rate features.

**Decision:** Rejected

---

### iter_003b — 2026-05-02

**Change:** Add segment_crash_trend (slope of annual crash count)

**Hypothesis:** Trend is orthogonal to rate levels — captures whether a segment is getting more dangerous.

**Result:** Passed Gate 1. Failed Gate 2 (MI 0.0095, Spearman 0.026). 70% of segments have zero trend due to insufficient years of data.

**Decision:** Rejected

---

### iter_003c — 2026-05-02

**Change:** Add road_group_risk (FSI rate per road classification)

**Hypothesis:** Road type is a structural property independent of crash history. Different road classes have fundamentally different risk profiles.

**Result:** Passed Gates 1-2 (Spearman 0.051, barely above 0.05). Failed Gate 3 (gain 0.0015, threshold 0.005). The feature adds real signal but not enough incremental value over existing features for a linear model.

**Decision:** Rejected

**Exhaustion conclusion:** Four feature candidates and one threshold adjustment have been tested with Logistic Regression. None improved val recall. The linear model has extracted maximum value from available features. Model advancement to Random Forest is justified per CLAUDE.md Model Advancement Criteria.

### iter_004 — 2026-05-03

**Change:** Model advancement from LogisticRegression to RandomForestClassifier (200 trees, max_depth=10, min_samples_leaf=50, class_weight=balanced)

**Hypothesis:** Logistic Regression was exhausted (4 features + 1 threshold tested, all rejected). Random Forest can capture non-linear interactions between features that a linear model cannot express. The same 8 features may produce better discrimination with a non-linear decision boundary.

**Result:** Fatal recall improved from 0.508 to 0.576 (+0.068) — the largest single improvement so far. Combined recall jumped from 0.569 to 0.691. However, the model overfits (generalization gap 0.111 > 0.10 threshold) and over-flags (flag rate 0.474 >> 0.30 maximum). Precision dropped to 0.042, below the 0.05 floor.

**Decision:** Conditionally Accepted

**Interpretation:** The RF found real signal — fatal recall improved meaningfully. But the hyperparameters are too permissive. max_depth=10 on 6.4M rows allows memorization. class_weight="balanced" on 1:71 imbalance amplifies the minority class too aggressively for a tree-based model. Next iteration must constrain the model (reduce depth, increase min_samples_leaf, or adjust class weights) to bring flag rate and precision into compliance without losing the fatal recall gain.

**Feature importance shift:** segment_injury_rate dominates (0.536 Gini) followed by segment_crash_rate (0.251). Notably, segment_fatal_rate has low importance (0.020) — the model predicts fatal+injury windows primarily from injury rate, not fatal rate. This is consistent with the collinearity finding from iter_003a.

### iter_005 — 2026-05-03

**Change:** max_depth reduced from 10 to 6 (all other RF parameters unchanged)

**Hypothesis:** Shallower trees will reduce memorization, lower generalization gap, and reduce flag rate.

**Result:** No meaningful change. Fatal recall unchanged (0.576). Flag rate slightly increased (0.474 → 0.484). Generalization gap marginally improved (0.111 → 0.106). The model makes all its decisions within the first 6 levels — depth was not the root cause of over-flagging.

**Decision:** Rejected

**Key insight:** The over-flagging is driven by class_weight="balanced" on a 1:71 imbalance, not by tree depth. In Random Forest, balanced weighting adjusts leaf vote counts, making the model aggressively predict positives. Next iteration must address the class weighting strategy directly.

---