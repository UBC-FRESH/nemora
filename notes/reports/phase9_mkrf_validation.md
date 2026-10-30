# Phase 9 Malcolm Knapp regional-pool validation

Date: 2026-10-30

Phase: P9

Task: P9.1

Status: complete with a negative transfer result.

## Pooling rule

- Local target: Malcolm Knapp centre, 50 km radius, CWHdm, mature (≥60 projected
  years), first visit, sample type F. Result: **6 plots**.
- Regional training pool: CWHdm plots in the **50–100 km ring** around the same
  centre. Result: **43 plots**.
- Local plots are held out. The regional ring is a different population, not
  local replication.
- Model: P8 empirical resampling; stand seed 20261030; 100 bootstrap replicates.

## Results

| Metric | Regional training | Local hold-out | Synthetic | Relative error |
| --- | ---: | ---: | ---: | ---: |
| Stems/ha | 1,476.05 | 1,855.92 | 1,476.00 | -20.5% |
| Basal area (m²/ha) | 48.64 | 52.04 | 48.18 | -7.4% |
| QMD (cm) | 24.79 | 20.87 | 20.39 | -2.3% |
| Mean DBH (cm) | 22.15 | 18.71 | 15.35 | -18.0% |
| Mean height (m) | 20.62 | 20.83 | 15.36 | -26.3% |

100 empirical bootstrap replicates:

- Stems/ha: 1,038.8–1,938.2 (5th–95th), median 1,407.
- Basal area: 42.56–53.31, median 48.28.
- QMD: 17.32–24.42, median 21.10.
- Mean DBH: 12.66–20.53, median 16.27.
- Mean height: 12.89–19.86, median 16.30.

## Interpretation

The regional training pool does not transfer cleanly to the six local plots.
BA and QMD are close, but DBH and height are strongly underestimated, and stems
are low. The regional ring is not a substitute for local calibration.

This is a useful negative result: a CWHdm label alone is too coarse for a
transferable stand model. A future pass should stratify by site series, age, or
height class, or use a hierarchical local adjustment.

## Limits

- The local hold-out has only six plots.
- The regional pool is a different population.
- Empirical resampling cannot extrapolate to local structure.
- No LiDAR or spatial competition is included in this comparison.
- No operational or real-tree-map claim is made.

## Reproduce

```bash
python scripts/validate_phase9_mkrf.py \
  --faib-root data/external/faib/raw \
  --release-root build/phase9 \
  --report notes/reports/phase9_mkrf_validation.json
```
