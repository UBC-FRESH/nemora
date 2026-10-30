# Phase 10 plan: calibration and handoff hardening

Parent issue: #36

Branch: `feature/p10-calibration-handoff`

Status: active

## Goal

Improve the Malcolm Knapp transfer and make the FHOPS scenario handoff
reproducible and internally consistent.

## P10.1: stratification

- Inspect local/regional age and height distributions.
- Try an age-first stratified regional pool; use height or site series only when
  the source supports it.
- Record the rule before scoring.
- Rerun the same empirical transfer metrics as P9.
- Report improvement or negative result.

## P10.2: FHOPS hardening

- Automate scenario registration/pin checks.
- Verify inventory and metadata checksums before the run.
- Reconcile the fixture block `work_required` with the exported merchantable
  volume or explain the mismatch.
- Keep FHOPS core unchanged.

## P10.3: rerun and report

- Run the reconciled scenario.
- Add machine-rate sensitivity evidence.
- Compare Gavin and Malcolm Knapp.
- Write `notes/reports/phase10_validation.md` and the close-out report.

## Limits

- Malcolm Knapp has six local mature plots.
- The regional pool is a different population.
- The synthetic inventory is not a tree map.
- No operational, financial, or regulatory claim is made.

## Verification

- `python -m pytest -q`
- `ruff format --check src tests scripts`
- `ruff check src tests scripts`
- `mypy src`
- `sphinx-build -b html docs _build/html -W`
- `pre-commit run --all-files`
