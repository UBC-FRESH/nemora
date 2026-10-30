# Plot-conditioned stand validation

`nemora.validation` answers a narrow question: how close is a synthetic stand,
conditioned only on training plots, to held-out plots from the same selection?

It does not claim that the synthetic stand is the real stand. It is a
reproducible check on the fitted structure.

## Inputs

Use the training and hold-out tree rows from `resolve_tree_detail`:

```python
from nemora.validation import validate_plot_conditioned_stand

result = validate_plot_conditioned_stand(
    training_trees,
    holdout_trees,
    bin_width_cm=2.0,
    area_ha=1.0,
    bootstrap_replicates=100,
    seed=20261030,
)
```

Required tree columns are `plot_id`, `species_code`, `dbh_cm`, `height_m`, and
`tally`. For FAIB tree detail, `tally` is `PHF_TREE`.

## What it fits

- **DBH:** grouped Weibull distribution over training DBH bins, weighted by
  `PHF_TREE`.
- **Height:** log-log height–DBH regression on training measured trees, with
  residual noise when generating heights.
- **Species:** training species proportions, sampled independently of DBH.

The synthetic stand has one hectare-equivalent of trees at the training mean
stems/ha. Each synthetic tree has `tally=1`.

## Metrics

- Stems/ha, basal area m²/ha, QMD, mean DBH, mean height.
- Weighted DBH Kolmogorov–Smirnov distance.
- Species total-variation distance.
- Relative error against the mean of the hold-out plot metrics.
- Bootstrap intervals from resampling training plots.

`result.comparison_frame()` returns the metric table. `result.to_dict()` returns
the JSON-ready payload used by the Phase 7 report.

## Compare candidate models

Use `compare_stand_models` when more than one model is plausible:

```python
from nemora.validation import compare_stand_models

comparison = compare_stand_models(training_trees, holdout_trees, seed=20261030)
print(comparison.comparison_frame())
print(comparison.selected)
```

The current candidates are pooled Weibull, species-stratified Weibull, and
empirical resampling. The score combines mean absolute relative error over the
stand metrics with weighted DBH KS and species TV distance. Selection is by
held-out score, not by visual fit.

## Limits

The fit currently ignores species-specific DBH and height relationships, spatial
structure, competition, and defects. Wide bootstrap intervals are a result, not a
formatting problem. Report them and refine the model rather than hiding them.
