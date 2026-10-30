# Phase 11 independent yield check

Date: 2026-10-30

Phase: P11

Task: P11.2

Status: complete.

## Estimator

Independent paraboloid-style check:

```text
V = pi/4 * D^2 * H_m * form_factor
```

- `D` is DBH in metres.
- `H_m` is merchantable height; default 70% of total height when not supplied.
- `form_factor` is 0.5.
- This is not a taper model and is not an operational yield estimate.

## Results

### Gavin

- Independent estimate: **417.64 m³**
- FHOPS processed volume: **530.54 m³**
- Registered `work_required`: 530.543340 m³
- Estimate vs processed: **-21.3%**

The independent check is lower than FHOPS. That is a useful discrepancy, not a
match target; FHOPS's merchantable-volume model and this simple form factor
answer different questions.

### Malcolm Knapp

- Independent estimate: **471.61 m³**
- Registered `work_required` updated to 471.614268 m³.
- The scenario pin remains valid; FHOPS validation passes.

## Limits

- The formula is a coarse geometric estimate.
- The 70% merchantable-height ratio and 0.5 form factor are assumptions.
- The check is independent of FHOPS, but not a valid taper model.
- No operational yield claim is made.

## Artifacts

- `notes/reports/phase11_yield_check.json`
- `scripts/check_phase11_yield.py`
- `notes/reports/phase11_scenario_catalog.json` updated with the MKRf estimate.
