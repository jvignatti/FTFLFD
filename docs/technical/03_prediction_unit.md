# Prediction Unit Definition (v1.1 — LOCKED)

**Document:** `docs/technical/03_prediction_unit.md`
**Version:** 1.1
**Created:** 2026-05-02
**Updated:** 2026-05-02
**Status:** Locked — changes require version bump and iteration counter reset

---

## 1. Definition

The FTFLFD system uses a **hybrid prediction unit** to maximize fatal crash coverage:

**Primary unit (LRS segments):**
(LRSNUMBER, floor(AOTACTUALMILEPOINT), 30-day window)

**Secondary unit (grid cells):**
(grid_row, grid_col, 30-day window) — for crashes without LRS data but with valid coordinates

Both unit types produce the same output: a binary prediction of whether a fatal or injury crash will occur within the next 30 days.

## 2. Why Hybrid

Initial analysis revealed that LRS-only segmentation excludes 54,226 of 154,838 labeled crash records (35%), including 226 of 958 fatalities (23.6%). Per the event contract, zero tolerance for missed fatalities is non-negotiable.

Hybrid segmentation reduces fatal loss from 226 to 5 — the 5 records that lack both LRS data and coordinates (previously documented in INGEST_NOTES.md Section 5.2).

| Coverage | LRS Only | Hybrid |
|---|---|---|
| Total crashes covered | 100,612 (65%) | 153,550 (99.2%) |
| Fatals covered | 732 (76.4%) | 953 (99.5%) |
| Fatals lost | 226 | 5 |

## 3. Primary Unit: LRS Segments

### 3.1 Spatial Reference

Uses the Vermont Linear Reference System:

- `LRSNUMBER` — identifies the route
- `AOTACTUALMILEPOINT` — identifies position along the route

### 3.2 Segment Assignment

    segment_id = "LRS_" + str(LRSNUMBER) + "_" + str(floor(AOTACTUALMILEPOINT))

Examples:
- Crash at LRSNUMBER=20, milepoint 2.85 → segment LRS_20_2
- Crash at LRSNUMBER=70, milepoint 0.81 → segment LRS_70_0

### 3.3 Segment Length

Fixed 1-mile segments via floor(AOTACTUALMILEPOINT).

### 3.4 Exclusions from LRS Segmentation

- Crashes with AOTACTUALMILEPOINT = 999.99 → sentinel value, excluded from LRS (may fall to grid)
- Crashes with null LRSNUMBER or null AOTACTUALMILEPOINT → excluded from LRS (fall to grid if coordinates exist)

## 4. Secondary Unit: Grid Cells

### 4.1 Purpose

Catches all crashes that have valid coordinates but no LRS assignment. This includes local roads, parking lots, urban streets, and any State System crash with missing LRS fields.

### 4.2 Grid Construction

A regular grid overlay across Vermont:

- Cell size: 1km x 1km (approximately 0.009 degrees latitude, 0.012 degrees longitude at Vermont's latitude)
- Grid origin: southwest corner of Vermont bounding box (42.7°N, -73.5°W)
- Cell assignment:

        grid_row = floor((latitude - 42.7) / 0.009)
        grid_col = floor((longitude - (-73.5)) / 0.012)
        segment_id = "GRID_" + str(grid_row) + "_" + str(grid_col)

### 4.3 Grid Cell Properties

- Uniform size regardless of road density
- Every cell with at least one crash becomes a potential prediction unit
- Cells with zero historical crashes are not prediction units (no prediction made for empty cells)

## 5. Unified Segment ID

Every crash receives exactly one segment_id:

1. If valid LRS data exists → assigned to LRS segment (prefix: LRS_)
2. Else if valid coordinates exist → assigned to grid cell (prefix: GRID_)
3. Else → excluded (no segment_id, documented as data loss)

The prefix (LRS_ or GRID_) makes the source of each segment deterministic and traceable.

A `segment_type` column is added:

| Value | Meaning |
|---|---|
| `lrs` | Segment derived from Linear Reference System |
| `grid` | Segment derived from coordinate grid |
| `none` | No segment assignable (excluded from spatial modeling) |

## 6. Town Mapping

Each segment is mapped to a town for aggregated reporting.

### 6.1 Rules

1. If CITYORTOWN exists and is consistent for the segment → use it
2. If CITYORTOWN is missing or inconsistent → derive from crash coordinates via spatial join to Vermont town boundaries
3. If no coordinates are available → segment is valid but town is marked as unknown

### 6.2 Constraints

- Town mapping must be deterministic
- Town mapping must be documented
- Town mapping must NOT block segmentation

## 7. Reporting Hierarchy

    Segment (model predicts here — LRS or GRID)
      → Town (aggregate for local policy)
        → County (aggregate for state policy)
          → Statewide (aggregate for legislature/governor)

Aggregation method: a town/county/state is flagged as high-risk if it contains at least one high-risk segment.

## 8. Label Construction

For each prediction unit (segment_id, 30-day window):

| Label | Condition |
|---|---|
| 1 (positive) | At least one Fatal or Injury crash occurs in this segment during the 30-day window |
| 0 (negative) | No Fatal or Injury crash occurs in this segment during the 30-day window |

PDO crashes do not contribute to the positive label.
Blank/unknown InjuryType crashes are excluded from label computation entirely.

## 9. Scope and Limitations

- LRS segments provide the most operationally meaningful units (tied to state route numbering)
- Grid cells provide coverage where LRS fails but are less operationally intuitive
- The model treats both unit types equally during training — the segment_type column is available as a feature if it proves informative
- 5 fatalities (0.5% of all fatalities) are excluded due to having neither LRS data nor coordinates. These are documented in INGEST_NOTES.md and flagged for retroactive geocoding.

## 10. Data Loss Documentation

| Category | Count | % of Total | Action |
|---|---|---|---|
| Crashes with LRS segment | 100,612 | 65.0% | Primary segmentation |
| Crashes with grid cell only | 52,938 | 34.2% | Secondary segmentation |
| Crashes with no spatial data | 1,288 | 0.8% | Excluded, documented |
| Fatals in LRS segments | 732 | 76.4% of fatals | Covered |
| Fatals in grid cells | 221 | 23.1% of fatals | Covered |
| Fatals excluded | 5 | 0.5% of fatals | Documented, retroactive geocoding planned |

## 11. Version History

- v1.0 (2026-05-02): LRS-only segmentation. Discovered 226 fatal exclusions (23.6%).
- v1.1 (2026-05-02): Hybrid segmentation (LRS + grid). Fatal exclusions reduced to 5 (0.5%). This version is locked.

---

**End of prediction unit definition.**