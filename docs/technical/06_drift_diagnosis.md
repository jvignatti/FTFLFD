# Temporal Drift Diagnosis

**Document:** `docs/technical/06_drift_diagnosis.md`
**Version:** 1.0
**Created:** 2026-05-15
**Status:** Critical finding — shapes all future modeling decisions

---

## 1. Summary

Fatal crashes do not persist at the segment level. Only 9.92% of segments with fatalities in 2022-2023 also had fatalities during the 2010-2019 training period. This means 90% of future fatal locations cannot be predicted from segment-specific crash history alone.

This is the root cause of all model performance limitations observed in Phase 1 and Phase 2.

## 2. Key Findings

### 2.1 Fatal Segment Persistence (Critical)

| Metric | Value |
|---|---|
| Fatal segments in training (2010-2019) | 537 |
| Fatal segments in recent (2022-2023) | 131 |
| Segments fatal in BOTH eras | 13 |
| Overlap rate | 9.92% |

90% of recent fatal locations are NOT historically fatal locations. A model trained exclusively on historical crash patterns at the segment level faces a fundamental ceiling.

### 2.2 New Segments

17.41% of segments active in 2022-2023 did not exist in the training data. 10.87% of recent fatalities occurred in these completely new segments. The model cannot predict risk for locations it has never observed.

### 2.3 Fatal Rate Shifts by Road Type

| Road Type | Fatal Rate 2010-2019 | Fatal Rate 2022-2023 | Change |
|---|---|---|---|
| State Highway (state owned) | 0.0088 | 0.0172 | +95% |
| Minor Collector | not reported | 0.0213 | new high |
| City/Village streets | 0.0088 | 0.0140 | +59% |
| Federal Aid Secondary | 0.0091 | 0.0116 | +27% |

Fatal risk shifted toward State Highways and Minor Collectors. The relative risk profile of road types changed between eras.

### 2.4 Seasonal Pattern Shift

Monthly fatal rates changed between eras:

| Month | 2010-2019 (per year) | 2022-2023 (per year) | Change |
|---|---|---|---|
| April | 3.5 | 6.0 | +71% |
| May | 4.7 | 10.0 | +113% |
| September | 6.5 | 5.0 | -23% |
| August | 6.1 | 8.5 | +39% |

The summer peak shifted earlier (May instead of September). Spring became significantly more dangerous.

### 2.5 Impairment Increase

Impairment involvement in fatal crashes increased from 47.5% (2010-2019) to 55.8% (2022-2023). More than half of all recent fatals involve impaired driving.

## 3. Implications for Modeling

### 3.1 Historical crash rates have a natural ceiling

The current 8 features are all derived from crash history at the segment level. These features can only predict the ~10% of fatals that recur in historically dangerous locations. The other ~90% require different signal.

### 3.2 Structural road features are the path forward

Features that describe WHY a segment is dangerous — traffic volume (AADT), road classification (functional class), road design (divided/undivided), speed environment — can predict risk at segments with no fatal history. A high-volume, undivided minor collector has inherent risk regardless of whether it has had a fatal crash before.

### 3.3 Expanding the training window does not help

Phase 2 attempted to fix drift by training on 2010-2022. This made performance worse because:
- COVID-era data (2020-2021) introduced disrupted patterns as training signal
- The fundamental problem is not temporal distance but feature type
- More years of crash history does not help if fatals do not persist

### 3.4 The exposure denominator is critical

Without AADT, the model cannot distinguish between:
- High-volume segments where crashes are statistically expected (low rate)
- Low-volume segments where any crash indicates genuine danger (high rate)

This distinction is essential for predicting FUTURE risk rather than summarizing PAST events.

## 4. Revised Strategy

1. Revert to Phase 1 proven configuration (train 2010-2019, val 2020-2021)
2. Integrate AADT as the first structural feature (gate test required)
3. Integrate functional class and divided road status
4. Accept the ~10% persistence ceiling for crash-history-only features
5. Document that the model's strength is structural risk identification, not crash history replay

## 5. What This Does NOT Mean

- It does NOT mean the model is useless. It beats naive baselines by 4x on historical patterns.
- It does NOT mean crash history has no value. The 10% persistence is real signal.
- It does NOT mean we should abandon the current approach. We should augment it with structural features.
- It DOES mean that expectations for recall must account for the persistence ceiling until structural features are integrated.