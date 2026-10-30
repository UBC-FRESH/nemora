# Phase 11 plan: multi-block validation and release readiness

Parent issue: #41

Branch: `feature/p11-multi-block-readiness`

Status: active

## Scope

- Build separate FHOPS scenarios for Gavin and Malcolm Knapp validated
  inventories. The FHOPS ground system remains one block/one landing per run;
  the catalog describes the set.
- Add an independent volume estimate and compare it with `work_required` and
  FHOPS processed volume.
- Run rate-model and truck-count sensitivity.
- Update release/readiness docs.

## P11.1: scenario set

- Reuse the Phase 9 Gavin export; create a Malcolm Knapp export.
- Register each scenario with checksum-verified pins.
- Write a machine-readable catalog with block, landing, inventory checksum, and
  provenance.

## P11.2: independent yield check

- Use a documented simple volume estimator based on DBH and height.
- Compare with FHOPS processed volume and registered `work_required`.
- Report differences; do not force agreement.

## P11.3: sensitivity

- Rate model: calibrated vs reference.
- Truck count: at least one alternative.
- Report makespan, volume, utilisation, and warnings.

## P11.4: release/readiness

- Update docs and limitations.
- Write the close-out report.
- Queue the phase PR; no tag/release without separate approval.

## Limits

- One block/one landing per FHOPS run; no simultaneous multi-block optimization.
- The synthetic inventories are not tree maps.
- The simple volume estimator is not a taper model.
- No operational, financial, or regulatory claim.

## Verification

- `python -m pytest -q`
- `ruff format --check src tests scripts`
- `ruff check src tests scripts`
- `mypy src`
- `sphinx-build -b html docs _build/html -W`
- `pre-commit run --all-files`
