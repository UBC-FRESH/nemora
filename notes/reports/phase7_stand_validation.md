# Phase 7 stand validation: Gavin Lake

Date: 2026-10-30

Phase: P7

Task: P7.2

Status: first validation pass complete; species-stratified refinement remains.

## Setup

- Forest: Alex Fraser Research Forest, Gavin Lake.
- Selection: 20 mature plots, 15 training and 5 hold-out.
- Training trees: 2,086 live stems across 15 plots.
- Hold-out trees: 784 live stems across 5 plots.
- Split seed: 20261030.
- Stand seed: 20261030.
- DBH model: grouped Weibull fit to training `PHF_TREE` tallies.
- Height model: log-log height–DBH fit on training measured trees.
- Synthetic stand: 1,978 trees in a one-hectare equivalent, species sampled
  from training proportions.

The training and hold-out `nemora.inventory.v1` releases validate and were
written to a local `build/phase7/gavin/` directory. They are not tracked because
they contain plot coordinates and are regenerable from public inputs.

## Results

| Metric | Training | Hold-out | Synthetic | Relative error |
| --- | ---: | ---: | ---: | ---: |
| Stems/ha | 1,977.84 | 2,250.57 | 1,978.00 | -12.11% |
| Basal area (m²/ha) | 48.15 | 40.72 | 52.43 | +28.76% |
| QMD (cm) | 20.30 | 16.01 | 18.37 | +14.73% |
| Mean DBH (cm) | 17.85 | 14.36 | 13.54 | -5.74% |
| Mean height (m) | 17.80 | 14.58 | 14.24 | -2.34% |

Distribution and composition:

- Weighted DBH KS distance: **0.201**.
- Species total-variation distance: **0.217**.

Weibull fit:

- `a`: 1.0275
- `beta`: 13.4341
- `s`: 69,564.15

Height fit:

- `intercept`: 0.4388
- `slope`: 0.8581
- `residual_std`: 0.2082
- measured trees: 2,086

## Uncertainty

Plot-resampling bootstrap, 100 successful replicates:

| Synthetic metric | 5th percentile | Median | 95th percentile |
| --- | ---: | ---: | ---: |
| Stems/ha | 1,422.75 | 2,030.50 | 2,492.15 |
| Basal area (m²/ha) | 23.52 | 52.06 | 74.21 |
| QMD (cm) | 11.26 | 18.07 | 25.43 |
| Mean DBH (cm) | 8.81 | 13.22 | 19.97 |
| Mean height (m) | 10.27 | 13.72 | 19.39 |

The wide intervals show that the 15-plot training sample does not tightly
constrain these metrics. The point estimates should not be read as precise
stand estimates.

## Interpretation

The first pass is not a clean validation:

- Basal area and QMD are high relative to the hold-out plots.
- Mean DBH and mean height are slightly low.
- The DBH KS distance of 0.201 and species TV distance of 0.217 indicate
  material distribution and composition differences.
- The hold-out plots are more variable than the pooled training fit; one
  Weibull by pooled tally does not capture species- or plot-level structure.

This is a useful negative result. The next refinement should compare:

- species-stratified DBH fits;
- a mixture or plot-level random-effect model;
- a height model that uses species and respects measured height coverage;
- possibly a density constraint rather than generating from the pooled mean.

## Limits

- The synthetic stand is a one-hectare statistical reconstruction, not a real
  tree map.
- The fit ignores spatial structure, competition, defects, and species-specific
  allometry.
- The hold-out has only five plots.
- No LiDAR/terrain context is included yet; P7.3 covers that.
- No operational or regulatory conclusion is supported.

## Reproduction

```bash
python scripts/validate_phase7_stand.py \
  --forest gavin \
  --faib-root data/external/faib/raw \
  --release-root build/phase7 \
  --report notes/reports/phase7_stand_validation_gavin.json \
  --bootstrap-replicates 100
```

The JSON result contains the metric table, fit parameters, per-plot hold-out
summaries, and bootstrap intervals.
