# FTFLFD

Proactive traffic fatality and serious injury prediction system for Vermont.

## Notebook Policy

Notebooks in `notebooks/` are for exploration only.

- Any finding must be re-implemented in `src/` before it influences the pipeline.
- Notebooks are never imported by `src/` code.
- Scratch notebooks in `notebooks/scratch/` are disposable and not reviewed.

## Project Structure

See `docs/technical/01_project_design.md` for full design documentation.

## Data

Raw and processed data are gitignored. Do not commit crash records.
Public outputs (aggregated reports, maps) live in `data/public/` and are tracked.
