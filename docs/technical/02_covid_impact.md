# COVID-19 Impact on Vermont Crash Data (2020–2021)

**Document:** `docs/technical/02_covid_impact.md`
**Version:** 1.0
**Created:** 2026-05-02
**Status:** Locked — changes require version bump

---

## 1. Why This Document Exists

The FTFLFD prediction system uses 2020–2021 as the **validation set** — the data against which model tuning decisions are made. This is deliberate. COVID-19 fundamentally altered traffic patterns during this period, and the validation set must test whether the model can generalize beyond "normal" conditions.

This document records what changed, why it matters for modeling, and how the system accounts for it. It is not a COVID research paper — it is a methodological constraint document.

This document does not modify system behavior; it provides context for interpreting model performance.

---

## 2. What Changed Nationally (2020–2021)

The following patterns were observed across the United States during 2020–2021 and are relevant to traffic fatality prediction.

### 2.1 Volume vs. Risk Paradox

Traffic volume dropped sharply in early 2020 (estimated 40–60% reduction in vehicle miles traveled during April–May 2020). However, the fatality **rate per mile traveled** increased significantly. Fewer vehicles on the road led to higher speeds, more risk-taking behavior, and a higher proportion of severe crashes among those that occurred.

This distinction between count and rate is critical, as models trained on counts alone may misinterpret reduced exposure as reduced risk.

### 2.2 Behavioral Shifts

- **Speed increases:** With less congestion, average speeds rose on highways and arterials
- **Impaired driving pattern changes:** Drinking shifted from bars/restaurants to homes; impaired driving moved to different hours and road types
- **Seatbelt use decline:** Some jurisdictions reported lower compliance during reduced enforcement periods
- **Demographic shift:** The population still driving during lockdowns skewed toward essential workers, delivery drivers, and higher-risk demographics

These shifts affect temporal and categorical feature distributions.

### 2.3 Enforcement Changes

- Many agencies deprioritized traffic stops during the pandemic
- Some agencies reduced staffing on patrol
- Reporting practices may have been inconsistent as agencies adapted to pandemic protocols
- Court closures may have affected how violations and crash reports were processed

Enforcement changes introduce potential bias in observed variables such as impairment and violations.

### 2.4 Road Use Changes

- Urban roads saw disproportionate volume drops (commuter traffic disappeared)
- Rural roads maintained or increased relative usage
- Bicycle and pedestrian activity changed — some areas saw increases as people exercised outdoors
- Commercial vehicle patterns shifted with supply chain disruptions

This spatial redistribution affects how risk is geographically represented in the data.

---

## 3. Vermont-Specific Observations

Vermont experienced trends consistent with the national patterns described above.

### 3.1 Known Vermont Factors

Based on domain expertise, these trends are expected to apply locally:

- Vermont is predominantly rural — the rural road usage shift likely had a proportionally larger effect than in urban states
- Tourism patterns changed significantly (Vermont is a destination state for skiing, foliage, and outdoor recreation)
- Seasonal variation in Vermont is extreme — winter driving conditions interact with pandemic-era behavior changes in ways that differ from temperate states

### 3.2 Unknown Vermont Factors (To Be Investigated)

The following are flagged as open questions. They may affect model interpretation but are not yet verified:

- Whether Vermont State Police or local agencies changed crash reporting procedures during 2020–2021
- Whether enforcement intensity (traffic stops, DUI checkpoints) was formally reduced
- Whether specific towns or corridors saw disproportionate changes in crash patterns
- Whether the tourism-driven crash population changed composition during pandemic travel restrictions

These questions do not block model development. They are documented here so that if anomalous validation results appear during 2020–2021 evaluation, investigators have a starting checklist. These uncertainties should be considered when interpreting anomalous model behavior during validation.

---

## 4. Implications for the FTFLFD System

### 4.1 Why 2020–2021 Is the Validation Set

The validation set is used for model tuning — hyperparameter selection, feature acceptance, and threshold calibration. Placing COVID years in the validation set means every tuning decision is tested against **disrupted conditions**.

The goal is robustness under disruption, not peak accuracy under normal conditions.

This is intentional and serves two purposes:

1. **Robustness testing:** A model that only works under normal traffic conditions is operationally fragile. If the model can predict fatality risk during a pandemic, it can likely handle other disruptions (economic downturns, major construction, extreme weather events, policy changes).

2. **Preventing false confidence:** If COVID years were in training, the model would learn pandemic-specific patterns and potentially overfit to them. If they were in a benchmark set, we would only discover the problem late. Validation is the right place — it forces us to confront the disruption during tuning, not after.

### 4.2 What the Model Should NOT Learn from COVID

The model should not learn that "fewer cars = more fatalities" as a general rule. That relationship was specific to the behavioral context of COVID-19. If the model picks up a strong negative correlation between traffic volume and fatality rate, that is a signal to investigate — not to celebrate high accuracy.

Such relationships should be flagged during model inspection (e.g., feature importance or partial dependence analysis).

### 4.3 The `data_era` Flag

All records carry a `data_era` flag assigned at ingestion:

| Value | Years | Relevance |
|---|---|---|
| `early` | 2010–2014 | Pre-modern reporting era |
| `historical` | 2015–2019 | Training era, pre-COVID baseline |
| `modern` | 2020+ | Includes COVID disruption and recovery |

The validation set (2020–2021) is entirely within the `modern` era. The model can use `data_era` as a contextual feature, allowing it to distinguish between pre-COVID and COVID-era patterns without being told explicitly "this is pandemic data."

This allows the model to condition predictions on temporal context without hard-coding event-specific assumptions.

### 4.4 Fatalities Are Fatalities

A core principle of this project: **a fatal crash is a fatal crash regardless of when or why it happened.** The model does not get to discount COVID-era fatalities as "unusual" or "non-representative." Every death on a Vermont road represents a failure of the system we are trying to improve.

The COVID context explains **why** patterns shifted. It does not excuse the model from predicting **where** risk concentrates, even under disrupted conditions.

---

## 5. Diagnostic Checks for COVID-Era Validation Results

When evaluating model performance on the 2020–2021 validation set, the following diagnostics should be applied.

### 5.1 Expected Patterns

- Lower total crash volume in 2020 compared to 2019 and 2021
- Fatality rate (fatalities per crash) may be **higher** than training era average
- Impairment-related crashes may show different temporal distribution (different hours, different days)
- Geographic distribution may shift toward rural corridors

Observed patterns may exhibit higher variance due to reduced sample sizes.

### 5.2 Warning Signs

| Observation | Possible Cause | Action |
|---|---|---|
| Val recall much lower than train recall (gap > 0.15) | COVID patterns too different from training era | Check if data_era feature is active; consider adding COVID-specific temporal features |
| Val recall much higher than train recall | Model is exploiting a COVID-specific artifact | Investigate which features drive the improvement; check for leakage |
| Model flags urban areas as high-risk but COVID data shows rural shift | Model learned pre-COVID urban patterns that don't transfer | Feature importance audit; consider spatial reweighting |
| Impairment features dominate model | COVID changed impairment patterns significantly | Verify impairment reporting consistency across eras |
| Model performance unstable month-to-month | Increased variance in COVID-era data | Evaluate performance over aggregated periods |

### 5.3 What Is NOT a Problem

- Lower total crash counts in 2020 validation data → expected, not a data quality issue
- Different seasonal patterns in 2020 (e.g., no spring break tourism crashes) → expected
- Higher variance in monthly metrics → expected with lower N

---

## 6. Relationship to Other Documentation

This document should be read in conjunction with the following system components:

| Document | Relationship |
|---|---|
| `docs/technical/01_project_design.md` | Event contract and split strategy that places 2020–2021 in validation |
| `config/splits.yaml` v1.1 | Formal split boundaries with 30-day gaps |
| `config/thresholds.yaml` v1.0 | Generalization gap thresholds that apply to COVID-era validation |
| `data/raw/INGEST_NOTES.md` | Raw data quality issues that may interact with COVID-era patterns |

---

**End of COVID impact documentation.**