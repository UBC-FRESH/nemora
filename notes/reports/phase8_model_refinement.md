# Phase 8 model refinement comparison

Date: 2026-10-30

Phase: P8

Task: P8.1

Status: complete. Empirical resampling selected for P8.3.

## Candidate models

- **Pooled Weibull**: P7 baseline. One grouped Weibull fit to all training
  trees, weighted by `PHF_TREE`, with a log-log height model.
- **Species Weibull**: separate Weibull per training species, with a pooled
  fallback below 25 training trees; global height model.
- **Empirical**: draw training tree rows with replacement, weighted by
  `PHF_TREE`, preserving the observed DBH/height/species triples.

All candidates use the same Gavin Lake split: 15 training plots / 2,086 trees
and 5 hold-out plots / 784 trees. The split seed and stand seed are 20261030.

## Results

| Model | Score | DBH KS | Species TV | BA error | QMD error | Mean DBH error | Mean height error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pooled Weibull | 0.545 | 0.201 | 0.217 | +28.8% | +14.7% | -5.7% | -2.3% |
| Species Weibull | 0.734 | 0.224 | 0.223 | +79.9% | +35.6% | +7.6% | +8.2% |
| Empirical | **0.319** | **0.094** | **0.143** | **+16.8%** | **+9.2%** | **-2.4%** | **+0.6%** |

Score = mean absolute relative error over the five stand metrics + weighted DBH
KS + species total-variation distance. Lower is better.

## Decision

Select the **empirical** candidate for P8.3.

It improves every reported comparison against the pooled baseline and avoids
the species-stratified model's small-sample overfit. The species model falls
back to pooled for one rare species and still produces a much heavier BA/QMD
tail, which is a negative result worth retaining.

## Height refinement

The selected empirical candidate preserves observed DBH/height pairs, so no
separate height model is fitted for it. A species-specific parametric height
model remains a future candidate; the P7 global log-log fit is retained only for
the pooled and species Weibull candidates.

## Limits

- Empirical resampling preserves the training support; it cannot extrapolate to
  stands outside that support.
- It is not a mechanistic growth or treatment-response model.
- The hold-out set has only five plots.
- The empirical candidate currently has no bootstrap interval; P8.3 must add
  plot-resampling uncertainty before reporting final metrics.
- No LiDAR, spatial competition, defect, or operational claim is included here.

## Reproduce

```bash
python scripts/compare_phase8_models.py \
  --forest gavin \
  --faib-root data/external/faib/raw \
  --report notes/reports/phase8_model_comparison.json
```

The JSON includes all candidate metrics and the selection score.
