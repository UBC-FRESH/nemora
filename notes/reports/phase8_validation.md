# Phase 8 validation report: Gavin Lake refinement

Date: 2026-10-30

Phase: P8 Validation refinement and LiDAR coverage expansion

Status: refined model and expanded coverage reported; no operational claim.

## Selected model

The empirical resampling candidate was selected from P8.1 by held-out score.
It draws training tree rows with replacement, weighted by `PHF_TREE`, preserving
observed DBH/height/species triples. It is a non-parametric alternative to the
pooled Weibull and does not extrapolate beyond the training support.

Training: 15 plots / 2,086 trees. Hold-out: 5 plots / 784 trees.

## Stand validation

| Metric | Hold-out | P7 pooled | P8 empirical | P8 relative error |
| --- | ---: | ---: | ---: | ---: |
| Stems/ha | 2,250.57 | 1,978.00 | 1,978.00 | -12.1% |
| Basal area (m²/ha) | 40.72 | 52.43 | 47.54 | +16.8% |
| QMD (cm) | 16.01 | 18.37 | 17.49 | +9.2% |
| Mean DBH (cm) | 14.36 | 13.54 | 14.02 | -2.4% |
| Mean height (m) | 14.58 | 14.24 | 14.67 | +0.6% |

Distribution/composition:

- Weighted DBH KS: **0.094** (P7: 0.201).
- Species TV distance: **0.143** (P7: 0.217).

The empirical candidate improves the DBH distribution, species mix, BA, QMD,
mean DBH, and mean height. Stems/ha remains unchanged because both models use
the training mean density; the hold-out density is 12.1% higher.

## Uncertainty

100 empirical plot-resampling replicates:

| Metric | 5th percentile | Median | 95th percentile |
| --- | ---: | ---: | ---: |
| Stems/ha | 1,498.70 | 1,970.00 | 2,609.60 |
| Basal area (m²/ha) | 42.06 | 48.42 | 53.68 |
| QMD (cm) | 14.91 | 17.60 | 21.05 |
| Mean DBH (cm) | 11.72 | 14.20 | 17.37 |
| Mean height (m) | 12.95 | 14.67 | 17.40 |

Intervals remain wide, but the selected model is more stable than the pooled
Weibull for BA and QMD.

## LiDAR and terrain context

Public LidarBC 2024 1 m DEM/DSM coverage was materialized through FEMIC in P7.
P8.2 resolved the one nodata plot using the alternate tile `093a031_3_2_1`.

- Valid canopy/slope samples: **16/20** plots.
- Residual missing plots: **4/20**; no indexed tile footprint contains them even
  at 25 km.
- Mean canopy cover (≥2 m): **0.683**.
- Mean canopy height: **12.59 m**.
- Mean canopy p95 height: **26.07 m**.
- Mean slope: **17.91%**.

Synthetic-stand centre (100 m × 100 m grid):

- Canopy cover: **0.879**.
- Mean canopy height: **9.82 m**.
- Canopy p95: **17.29 m**.
- Mean slope: **17.23%**.

This is context evidence with partial coverage; it is not a stand-level accuracy
claim.

## Observed, fitted, assumed, synthetic

- **Observed:** FAIB plot locations, projected ages, DBH/height, expansion
  tallies, species codes; LidarBC DEM/DSM-derived CHM/slope at sampled points.
- **Fitted:** none for the selected empirical model; the pooled and species
  Weibull candidates were comparison-only.
- **Assumed:** BEC/age/radius selection; weighted recomposition of training trees;
  one-hectare-equivalent density; 5×5 plot sample grids.
- **Synthetic:** the resampled tree list and stand metrics.
- **Unavailable:** canopy/slope evidence for four residual plots.

## Limits

- Empirical resampling cannot extrapolate to stands outside the training support.
- It is not a mechanistic growth, treatment, or spatial-competition model.
- The hold-out set has only five plots.
- Four plots lack LidarBC coverage; the remaining LiDAR evidence is context, not
  full-plot validation.
- No operational, financial, or regulatory conclusion is supported.

## Reproduce

```bash
python scripts/compare_phase8_models.py \
  --forest gavin \
  --faib-root data/external/faib/raw \
  --report notes/reports/phase8_model_comparison.json

python scripts/validate_phase8_stand.py \
  --forest gavin \
  --faib-root data/external/faib/raw \
  --release-root build/phase8 \
  --report notes/reports/phase8_stand_validation_gavin.json \
  --bootstrap-replicates 100
```

LiDAR metrics: `notes/reports/phase8_lidar_metrics.json`.
