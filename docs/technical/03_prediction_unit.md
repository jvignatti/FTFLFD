# Prediction Unit Definition (v1.0 — LOCKED)

**Document:** `docs/technical/03_prediction_unit.md`
**Version:** 1.0
**Created:** 2026-05-02
**Status:** Locked — changes require version bump and iteration counter reset

---

## 1. Definition

The prediction unit for the FTFLFD system is:

**(LRSNUMBER, floor(AOTACTUALMILEPOINT), 30-day window)**

This means: for each unique combination of route (identified by LRS number) and 1-mile segment (identified by the integer floor of the milepoint), the model predicts whether a fatal or injury crash will occur within the next 30 days.

## 2. Segment Construction

### 2.1 Spatial Reference

The system uses the Vermont Linear Reference System (LRS) as the canonical spatial framework. Each crash record's position is defined by:

- `LRSNUMBER` — identifies the route
- `AOTACTUALMILEPOINT` — identifies the position along the route

### 2.2 Segment Assignment

Each crash is assigned to exactly one segment using:

    segment_id = (LRSNUMBER, floor(AOTACTUALMILEPOINT))

Examples:
- Crash at LRSNUMBER=20, milepoint 2.85 → segment (20, 2)
- Crash at LRSNUMBER=70, milepoint 0.81 → segment (70, 0)
- Crash at LRSNUMBER=150, milepoint 0.91 → segment (150, 0)

### 2.3 Segment Length

Fixed 1-mile segments. No variable segmentation by road class for MVP.

### 2.4 Exclusions

- Crashes with `AOTACTUALMILEPOINT = 999.99` → sentinel value, excluded (milepoint unknown)
- Crashes with null or missing `LRSNUMBER` → excluded from segment-level modeling
- Crashes with null or missing `AOTACTUALMILEPOINT` → excluded from segment-level modeling

Excluded crashes are documented but not discarded from the dataset. They may be used for temporal or statewide analyses that do not require segment-level assignment.

## 3. Town Mapping

Each segment is mapped to a town for aggregated reporting.

### 3.1 Rules

1. If `CITYORTOWN` exists and is consistent for the segment → use it
2. If `CITYORTOWN` is missing or inconsistent → derive from crash coordinates via spatial join to Vermont town boundaries
3. If no coordinates are available → segment is valid but town is marked as `unknown`

### 3.2 Constraints

- Town mapping must be deterministic (same input always produces same output)
- Town mapping must be documented (method and source of boundary data recorded)
- Town mapping must NOT block segmentation — a segment without a town assignment is still a valid prediction unit

## 4. Reporting Hierarchy

The model predicts at segment level. Reporting aggregates upward:

    Segment (model predicts here)
      → Town (aggregate for local policy)
        → County (aggregate for state policy)
          → Statewide (aggregate for legislature/governor)

Aggregation method: a town/county/state is flagged as high-risk if it contains at least one high-risk segment. Risk scores can be averaged or max-pooled depending on the reporting context. The aggregation method must be documented in each report.

## 5. Label Construction

For each prediction unit (LRSNUMBER, floor(AOTACTUALMILEPOINT), 30-day window):

| Label | Condition |
|---|---|
| 1 (positive) | At least one Fatal or Injury crash occurs in this segment during the 30-day window |
| 0 (negative) | No Fatal or Injury crash occurs in this segment during the 30-day window |

PDO crashes do not contribute to the positive label.
Blank/unknown InjuryType crashes are excluded from label computation entirely.

## 6. Scope Limitations (MVP)

- **State System roads only** for MVP. The LRS has strongest coverage on State Highways and Class 1 Town Highway links. Local roads (Class 2, Class 3) may have incomplete LRS data.
- **Non-LRS crashes** (those without valid LRSNUMBER) cannot be assigned to segments. They are excluded from segment-level modeling but preserved for potential future use.
- **Coverage expansion** to local roads is a future phase dependent on LRS data quality verification.

## 7. This Document Removes the Final Blocker

Per CLAUDE.md Hard Blockers section: "Prediction unit must be defined and locked. No code in src/training/ may execute until this is locked."

This document satisfies that requirement. Iteration 001 may now proceed.

---

**End of prediction unit definition.**