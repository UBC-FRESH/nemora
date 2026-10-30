# Phase 10 Malcolm Knapp stratification

Date: 2026-10-30

Phase: P10

Task: P10.1

Status: complete with a mixed result.

## Rule

Pre-registered before scoring:

- local hold-out: Malcolm Knapp CWHdm plots within 50 km;
- regional training: CWHdm plots in the 50–100 km ring with
  `PROJ_AGE_1` between 80 and 110 years;
- model: P8 empirical; 100 bootstrap replicates.

This is an age-first stratification. Height is reported as a secondary
diagnostic, but no height-matched rule was adopted.

## Before/after

| Metric | P9 base (43 regional) | P10 age 80–110 (15 regional) |
| --- | ---: | ---: |
| Stems/ha error | -20.5% | -38.8% |
| BA error | -7.4% | -18.1% |
| QMD error | -2.3% | +4.8% |
| Mean DBH error | -18.0% | -0.4% |
| Mean height error | -26.3% | -11.1% |
| DBH KS | 0.277 | 0.116 |
| Species TV | 0.504 | 0.495 |
| Composite score | 0.929 | **0.758** |

## Decision

Age stratification improves the DBH distribution, mean DBH, mean height, and DBH
KS. It worsens stems/ha and BA because the age-filtered regional pool has lower
density. The composite score improves, but the density/trade-off is material.

Do not use the age-stratified model for the FHOPS system comparison without a
density constraint. Keep the P9 base result as the density baseline and report
the age-stratified result as a distributional improvement with a density
limitation.

Height matching was also tested informally: it improved height/BA but worsened
DBH/QMD and stems, so it was not adopted.

## Limits

- The local hold-out has only six plots.
- The age-filtered pool has 15 plots.
- Age is a projected source attribute, not field-verified.
- The empirical model cannot extrapolate beyond the selected regional support.
- No operational or real-tree-map claim is made.

## Reproduce

```bash
python scripts/validate_phase10_mkrf.py \
  --faib-root data/external/faib/raw \
  --report notes/reports/phase10_mkrf_stratified.json
```
