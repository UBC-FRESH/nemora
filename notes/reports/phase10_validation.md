# Phase 10 validation report: calibration and handoff hardening

Date: 2026-10-30

Phase: P10 Calibration and handoff hardening

Status: complete; transfer is still limited, but the FHOPS scenario is internally consistent.

## Malcolm Knapp stratification

Age-first rule: CWHdm regional plots 50–100 km from the centre with
`PROJ_AGE_1` 80–110. This yields 15 regional plots vs 43 in P9.

| Metric | P9 base | P10 age 80–110 |
| --- | ---: | ---: |
| Stems/ha error | -20.5% | -38.8% |
| BA error | -7.4% | -18.1% |
| QMD error | -2.3% | +4.8% |
| Mean DBH error | -18.0% | -0.4% |
| Mean height error | -26.3% | -11.1% |
| DBH KS | 0.277 | 0.116 |
| Species TV | 0.504 | 0.495 |
| Composite score | 0.929 | **0.758** |

The distributional metrics improve; density and BA worsen. The stratified model
is not used for system comparison without a density constraint.

## FHOPS scenario

The scenario pin is reproducible, checksum-verified, and block `work_required`
is reconciled to 530.543340 m³. The prior work-required warning is gone.

## Rate sensitivity

FHOPS 1.2.0 on the 1,978-tree inventory:

| Rate model | Processed m³ | Makespan h | Trees | Unreachable | Warnings |
| --- | ---: | ---: | ---: | ---: | --- |
| calibrated | 530.54 | 115.456 | 1,978 | 0 | none |
| reference | 530.54 | 115.774 | 1,978 | 0 | none |

Makespan is nearly invariant (+0.27%) and truck-bound. Machine utilisation
differs materially: skidder utilisation rises from ~1.2% to ~7.6% under the
reference rates, while loader/processor utilisation changes less. Do not rank
the stands on makespan alone; the bottleneck is the single truck.

## Limits

- Malcolm Knapp has six local hold-out plots; the regional ring is a different
  population.
- The empirical model is not a process model.
- The FHOPS scenario is a fixture with uncalibrated machine rates.
- The work-required reconciliation is not an independent yield estimate.
- No operational, financial, or regulatory claim is supported.

## Artifacts

- `notes/reports/phase10_mkrf_stratified.md`
- `notes/reports/phase10_fhops_scenario.md`
- `notes/reports/phase10_fhops_system_summary.json`
- `notes/reports/phase10_fhops_scenario_registration.json`

## Reproduce

```bash
python scripts/validate_phase10_mkrf.py --faib-root data/external/faib/raw \
  --report notes/reports/phase10_mkrf_stratified.json

python scripts/register_phase10_fhops.py \
  --scenario-template path/to/fhops/examples/tree_system/scenario \
  --inventory-dir build/phase9/fhops \
  --output-dir build/phase10/fhops/scenario \
  --work-required-m3 530.5433400362283 \
  --report notes/reports/phase10_fhops_scenario_registration.json

fhops tree system build/phase10/fhops/scenario/scenario.yaml \
  --block B01 --feller H1 --skidder H2 --skidder H3 \
  --processor H5 --loader H4 --truck-capacity 5 --truck-count 1 \
  --landing-capacity 15 --out build/phase10/fhops/system
```
