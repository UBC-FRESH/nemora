# Phase 9 plan: cross-forest validation and FHOPS handoff

Parent issue: #31

Branch: `feature/p9-cross-forest-handoff`

Status: active

## Goal

Extend the validated workflow to Malcolm Knapp with an explicit regional pool,
then hand the selected stand to FHOPS through a tested tree-inventory export.

## P9.1: Malcolm Knapp

- Select local Malcolm Knapp plots with the same mature-stand rules.
- Build a CWHdm regional pool within 100 km.
- Record the pooling rule before fitting.
- Keep local and regional samples distinguishable.
- Run the P8 empirical model or a justified fallback.
- Report the same metrics as Gavin Lake plus sample-size limits.

## P9.2: FHOPS handoff

- Confirm the FHOPS tree inventory fields and units.
- Export block/landing context, tree IDs, species, DBH, height, and synthetic
  provenance.
- Add a round-trip test through the FHOPS loader/schema.
- No FHOPS core changes.

## P9.3: run and report

- Run FHOPS on the exported inventory.
- Capture command, inputs, outputs, and versions.
- Compare Gavin and Malcolm Knapp metrics.
- Write `notes/reports/phase9_validation.md` and the close-out report.

## Limits

- The local Malcolm Knapp sample has only six mature plots.
- The regional pool is a different population, not local replication.
- The synthetic stand is not a surveyed stand.
- No operational, financial, or regulatory claim is made.

## Verification

- `python -m pytest -q`
- `ruff format --check src tests scripts`
- `ruff check src tests scripts`
- `mypy src`
- `sphinx-build -b html docs _build/html -W`
- `pre-commit run --all-files`
