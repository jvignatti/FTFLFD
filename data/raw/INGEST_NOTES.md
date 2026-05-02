# Vermont Crash Data — Raw Ingest Notes

**File:** `vt_crashes_all.csv`
**Ingested:** 2026-05-02
**Status:** Raw, untransformed, local only (gitignored)

---

## 1. Source

**System:** Vermont Public Crash Query Tool (VTrans)
**URL:** https://apps.vtrans.vermont.gov/CrashPublicQueryTool/
**Authority:** Vermont Agency of Transportation
**Access type:** Public, no authentication required
**Pull method:** ArcGIS REST API workaround. The standard PQT export interface limits results to 2,000 crashes per query. The full dataset was pulled via the underlying ArcGIS REST endpoint to bypass this limitation.
**Pull date:** 2026-05-02
**Pulled by:** Project owner

## 2. Coverage

**Date range:** 2010-01-01 through 2026-05-01 (approximate)
**Total records:** 179,282
**File size:** ~68 MB
**Encoding:** Unknown — to be verified at ingestion (likely UTF-8 or Windows-1252; Vermont state systems vary)
**Header row:** Yes
**Total columns:** 36

## 3. Legal and Use Constraints

Per the VTrans Public Query Tool disclaimers:

- This data is **exempt from discovery or admission under 23 U.S.C. § 409**
- Only **non-personal data** is included
- **Errors and omissions may exist** in the underlying records
- Results may not exactly match VTrans staff queries due to reporting nuances and FARS reconciliation
- **Fatal crash data may be incomplete for the most recent 90 days** — fatal crash investigations take 90+ days; recent fatal records may not yet appear or may be revised
- **Non-Reportable crash data is only available from 2013 onward** (introduced by VSP that year)

## 4. Column Inventory

The file contains 36 columns. Listed below in the order they appear.

### 4.1 Identifiers
- `OBJECTID` — Unique record identifier
- `REPORTNUMBER` — Law enforcement report number
- `REPORTINGAGENCYid` — Agency code (numeric ID)
- `ReportingAgency` — Agency name (human-readable)

### 4.2 Time
- `ACCIDENTDATE` — Crash date and time (format to be verified at ingestion)

### 4.3 Location — Textual
- `STREETADDRESS` — Street address as reported
- `INTERSECTIONWITH` — Cross-street if applicable
- `NonReportableAddress` — Alternative address field for non-reportable crashes
- `Route` — Route designation (text)
- `RDFLNAME` — Unknown semantic. To be clarified at ingestion.

### 4.4 Location — Geographic
- `LATITUDE`, `LONGITUDE` — Coordinates (uppercase columns)
- `latitude`, `longitude` — Coordinates (lowercase columns)
- `EASTING`, `NORTHING` — Projected coordinates (likely Vermont State Plane; CRS to be confirmed)
- `LOC_ERROR` — Geocoding error indicator. **Caveat:** This field is unreliable as a filter. Some records flagged `INVALID LOCATION ROUTE ID` still have valid lat/long. Exclusion must be based on missing coordinates directly, not on this field.
- `HOWMAPPED` — Geocoding method (manual, automatic, route-based, etc.). Critical for understanding spatial reliability.

**Note on duplicate coordinate columns:** The file contains both uppercase (`LATITUDE`/`LONGITUDE`) and lowercase (`latitude`/`longitude`) coordinate columns. Working hypothesis based on user inspection: lowercase values are refined versions of the uppercase originals, with differences typically in meters or less. **Verification required at ingestion:** compute the per-record distance between paired coordinates, flag any record where the difference exceeds 100 meters as a data quality issue, and canonicalize one column as the source of truth based on findings.

### 4.5 Administrative Geography
- `CITYORTOWNid` — Municipality code
- `CITYORTOWN` — Municipality name

### 4.6 Roadway Classification
- `RoadGroup` — Highway system class (State/Federal Aid/Local)
- `AOTROUTEid` — AOT route code
- `AOTROUTE` — AOT route designation
- `LRSNUMBER` — Linear Reference System identifier
- `AOTACTUALMILEPOINT` — Milepoint along AOT route. **Caveat:** Value `999.99` is a sentinel for unknown milepoint. Treat as null at ingestion.
- `AOTROADWAYGROUPid` — Roadway group identifier (relationship to `RoadGroup` to be confirmed at ingestion)
- `RoadCharacteristics` — Intersection type, parking lot, ramp, etc.

### 4.7 Conditions
- `Weather` — Weather at time of crash
- `SurfaceCondition` — Road surface state (dry, wet, icy)
- `RoadCondition` — Road condition descriptor
- `DayNight` — Day vs Night indicator

### 4.8 Crash Characteristics
- `DirOfCollision` — Collision direction/type
- `Animal` — Animal involvement flag
- `Impairment` — Impairment flag
- `Involving` — Vehicle/road user types involved (motorcycle, pedestrian, heavy truck, etc.)

### 4.9 Outcome (Target)
- `InjuryType` — Crash severity. Drives label derivation per event contract:
  - `Fatal` → positive (1)
  - `Injury` → positive (1)
  - `Property Damage Only` → negative (0)
  - blank/unknown → excluded from training

## 5. Known Data Quality Issues

### 5.1 Excel Date-Conversion Corruption

A subset of rows contain Excel auto-converted values where original alphanumeric route or report identifiers were mangled into date strings.

**Examples observed:**
- `Oct-44`, `Oct-15`, `Oct-23`, `Oct-40`, etc. (dozens of variants)
- `D302042`, `10-25064`, `10WT03936`, `187`

**Root cause hypothesis:** Upstream parsing in either the source database or an intermediate Excel-based pipeline interpreted strings like `1-44` (Route 1, Mile 44) or `10-25064` (a 2010 report number) as dates and silently converted them.

**Geographic concentration:** Affected rows cluster around Burlington-area coordinates (latitudes 44.42–44.52, longitudes −73.10 to −73.27), suggesting the issue may originate from a specific reporting agency or specific time window.

**Mitigation strategy (deferred to ingestion):**
- Detect rows where the address or route field matches the pattern `^[A-Za-z]{3}-\d{2}$` (e.g., `Oct-44`)
- Flag affected rows with `data_quality_flag = "excel_date_corruption"`
- Decide per-iteration whether to drop, repair, or use only spatial features for these rows

### 5.2 Coordinate-Missing Records (3,030 of 179,282 — 1.69%)

A subset of records lack lat/long coordinates and cannot be assigned to spatial units. These records have textual location references (addresses, route IDs) but no usable geographic coordinates.

**Severity breakdown of the 3,030 unmappable records:**

| InjuryType | Count | % of unmappable |
|---|---|---|
| Fatal | 5 | 0.17% |
| Injury | 200 | 6.6% |
| Property Damage Only | 1,083 | 35.7% |
| Unknown / blank | 1,742 | 57.5% |
| **Total** | **3,030** | **100%** |

**Observations:**

1. **5 fatalities are excluded** by this filter. This is a real data loss. While statistically small, each excluded fatality represents a missed signal. Documented as a known gap to address in later iterations through retroactive geocoding or alternative location matching.

2. **57.5% of unmappable records have unknown severity** — far higher than the rate among coordinate-present records. This suggests these records originate from reporting pathways with weaker data hygiene (likely Non-Reportable VSP records or specific agency feeds), not random quality loss.

3. **Decision for MVP:** Exclude all coordinate-missing records from the spatial training pool.

4. **Action item logged for future iteration:** Retroactive geocoding of the 5 unmappable fatalities should be prioritized. These represent confirmed events at unknown locations — exactly the kind of data the model should eventually account for.

### 5.3 Recent-Year Incompleteness (Per VTrans Disclosure)

The data is non-stationary at the recent edge:

- **Fatal crashes** for the most recent 90 days may be missing or revised
- **Late-reporting agencies** continue to submit historical records, meaning row counts for any year (especially 2024–2026) may grow over time
- **Verification cycles** mean some recent records may be flagged provisional

This affects how Benchmark 4 (2026) is treated. The "auto" end date in `splits.yaml` v1.1 is correct because it adapts to the data state at runtime, but the ingestion code must record the data state at time of ingestion in `config_snapshot.yaml`.

Per the project owner's domain knowledge: data is always collected regardless of world situation; sometimes it takes years to be normalized but generally reaches the database according to VTrans methodology.

### 5.4 Reporting System Changes Across Years

Per the VTrans glossary:

- **2013:** Non-Reportable crash data introduced by VSP. Pre-2013 vs post-2013 data have structurally different completeness for minor incidents.
- **July 2014:** Updated UCRF (Uniform Crash Report Form) introduced expanded distracted driving categories. Pre-July-2014 distracted-driving features are not directly comparable to post-July-2014.
- **2014 onward:** "Under Influence-Impaired" was clarified as a distinct concept.

**Implication:** Features derived from distraction or impairment fields must be computed with awareness of these reporting regime changes, or restricted to post-2014 data.

## 6. Glossary Anchors (From VTrans Source Documentation)

The following definitions are taken directly from the VTrans Public Query Tool glossary and govern label derivation for this project:

| Term | VTrans Definition |
|---|---|
| Fatal Crash | A crash on a public highway resulting in one or more person fatal injuries. If a crash involved a fatality and a Suspected Serious injury, the crash type is Fatal. |
| Injury-Fatal | Any injury resulting in death within 30 days of the crash. |
| Injury-Suspected Serious | Any injury (other than fatal) preventing the injured person from walking, driving, or normally continuing prior activities. Previously called "Incapacitating." |
| Major Crash | Any crash where at least one person suffered a fatal or serious injury. |
| PDO | Property Damage Only — crash with no person injury or fatality. |
| Unknown Crash Type | Severity not documented; also assigned to Non-Reportable incidents. |

**Project alignment:** Our event contract (`docs/technical/01_project_design.md` v1.0) targets `InjuryType = "Fatal" OR "Injury"`. This corresponds to the VTrans "Major Crash" concept when restricted to Fatal + Serious Injury, broadened to include all Injury severities.

**Open question for ingestion:** The contract currently treats all `Injury` rows as positive, including suspected minor and possible injuries. We may want to revisit this and align strictly with VTrans "Major Crash" (Fatal + Suspected Serious only). **This is a contract amendment candidate, not a tonight decision.**

## 7. Yearly File Splits (To Be Generated)

The master file will be split into yearly slices to enable per-year integrity checks before the data is trusted as a whole. Splits to be generated:

- `vt_crashes_2010.csv` through `vt_crashes_2026.csv` (17 files)

Each split will be hashed (SHA256) and the hashes recorded below once generated.

## 8. File Hashes

To be filled after splitting is complete:

| File | SHA256 |
|---|---|
| vt_crashes_all.csv | `<pending>` |
| vt_crashes_2010.csv | `<pending>` |
| vt_crashes_2011.csv | `<pending>` |
| ... | ... |
| vt_crashes_2026.csv | `<pending>` |

## 9. What Was NOT Done in This Ingest

To preserve the integrity of future leakage checks and contract enforcement, the following operations were explicitly **not performed** during this ingest step:

- No date parsing
- No label derivation (`InjuryType` → `target`)
- No coordinate validation
- No missing-value imputation
- No row exclusion (corrupted rows were observed but not removed)
- No train/val/benchmark splitting
- No feature engineering

These steps belong to the ingestion module (`src/ingestion/`) and will be performed under controlled, version-tracked conditions.

---

**End of ingest notes.**
