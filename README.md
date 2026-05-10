# FTFLFD

Proactive traffic fatality and serious injury prediction system for Vermont.

## Objective

Identify road segments with elevated risk of fatal or injury crashes within the next 30 days using historical crash data and temporal patterns. The goal is proactive observation — getting ahead of the problem, not reacting after it happens.

**Current limitation:** The source dataset does not distinguish between serious and minor injuries. The model treats all injury severities as the positive class. Serious injury granularity will be incorporated when Department of Health data becomes available in a future phase.

## Current Status

- **Dataset:** 179,282 Vermont crash records (2010–2026) from VTrans
- **Pipeline:** Ingestion → Splitting → Window Generation → Feature Engineering → Training → Evaluation
- **Model:** Logistic Regression baseline validated. Random Forest in progress.
- **Fatal coverage:** 99.5% (953 of 958 fatalities assigned to prediction units)
- **Iteration:** 004 (model advancement to Random Forest after LogReg exhaustion)

## How It Works

The system uses hybrid spatial segmentation:

- **Primary:** 1-mile segments on state routes via Vermont Linear Reference System
- **Secondary:** 1km grid cells for roads without LRS coverage

Rolling 30-day prediction windows with 7-day step generate segment-level observations. The model learns from 10 years of training data (2010–2019) and validates against 2020–2021 (including COVID disruption for robustness testing).

## Data Policy

- Raw crash data is **never committed** to this repository
- All data stays local on the machine running the model
- SHA256 hashes of all data files are tracked for audit and reproducibility
- Only aggregated, non-sensitive outputs appear in `data/public/`
- This data is exempt from discovery or admission under 23 U.S.C. § 409

## Methodology

This project follows strict engineering discipline:

- One change per iteration — never change two variables at once
- 3-gate feature acceptance test (availability, signal, incremental gain)
- Recall-prioritized with Fatal recall as primary metric
- Frozen benchmark sets never used for tuning decisions
- Adaptive kill switch (20 iterations maximum per phase)
- Every experiment is fully reproducible (data hashes, config versions, random seeds)

See `CLAUDE.md` for the complete engineering specification.

## Documentation

| Document | Purpose |
|---|---|
| `CLAUDE.md` | Authoritative project rules and constraints |
| `docs/technical/01_project_design.md` | System design (initial version, sync pending) |
| `docs/technical/02_covid_impact.md` | Why 2020–2021 is the validation set |
| `docs/technical/03_prediction_unit.md` | Hybrid segmentation definition |
| `data/raw/INGEST_NOTES.md` | Raw data documentation and quality issues |

## Notebook Policy

Notebooks in `notebooks/` are for exploration only. Any finding must be re-implemented in `src/` before it influences the pipeline. Notebooks are never imported by `src/` code.

## Environment

- Python 3.11
- Dependencies: `pip install -r requirements.txt`
- Runner: `python run.py [install|test|lint|check-leakage]`

## License

Apache 2.0