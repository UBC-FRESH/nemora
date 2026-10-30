# Phase 11 multi-block validation report

Date: 2026-10-30

Phase: P11 Multi-block validation and release readiness

Status: complete; sensitivity and independent-yield checks are reported.

## Scenario set

| Scenario | Block | Landing | Trees | Source |
| --- | --- | --- | ---: | --- |
| gavin-empirical | B01 | L1 | 1,978 | P8 empirical Gavin stand |
| mkrf-local-empirical | B02 | L2 | 1,856 | Malcolm Knapp local six-plot empirical |

FHOPS runs one block/one landing at a time. The catalog describes the set; this
is not a simultaneous multi-block optimization.

## Independent yield check

Paraboloid-style estimate, not a taper model:

| Block | Independent estimate | FHOPS processed | Relative |
| --- | ---: | ---: | ---: |
| Gavin | 417.64 m³ | 530.54 m³ | -21.3% |
| Malcolm Knapp | 471.61 m³ | 599.27 m³ | -21.3% |

The independent estimate is consistently lower than FHOPS. The discrepancy is
retained. Gavin is registered at the FHOPS volume; Malcolm Knapp is registered
at the independent estimate and therefore retains the expected
`work_required` warning.

## Sensitivity

| Run | Block | Rate | Trucks | Trees | Volume m³ | Makespan h | Unreachable | Warnings |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gavin_cal_t1 | B01 | calibrated | 1 | 1,978 | 530.54 | 115.456 | 0 | 0 |
| gavin_ref_t1 | B01 | reference | 1 | 1,978 | 530.54 | 115.774 | 0 | 0 |
| gavin_cal_t2 | B01 | calibrated | 2 | 1,978 | 530.54 | 58.304 | 0 | 0 |
| mkrf_cal_t1 | B02 | calibrated | 1 | 1,856 | 599.27 | 129.756 | 0 | 1 |
| mkrf_ref_t1 | B02 | reference | 1 | 1,856 | 599.27 | 129.723 | 0 | 1 |
| mkrf_cal_t2 | B02 | calibrated | 2 | 1,856 | 599.27 | 65.839 | 0 | 1 |

- Truck count dominates makespan: doubling trucks roughly halves it.
- Rate model changes are small at one truck: +0.28% Gavin, -0.03% MKRf.
- The MKRf warning is the independent-yield/`work_required` mismatch, not a
  simulation failure.
- Gavin runs are warning-free.

## Interpretation

The scenario set is reproducible and the handoff is internally consistent for
Gavin. Malcolm Knapp provides a useful transfer/demo case, but its local
inventory is not independently hold-out validated and its registered
`work_required` intentionally reflects the independent estimate.

Trucking is the bottleneck in every run; comparing stands on makespan without
fixing fleet assumptions would be misleading.

## Limits

- One block/one landing per run; no simultaneous multi-block optimization.
- Synthetic inventories, not tree maps.
- The independent volume check is a geometric estimate, not a taper model.
- Machine rates are fixture defaults/reference values, not field-calibrated.
- No operational, financial, or regulatory claim is supported.

## Artifacts

- `notes/reports/phase11_sensitivity.json`
- `notes/reports/phase11_yield_check.json`
- `notes/reports/phase11_scenario_catalog.json`
