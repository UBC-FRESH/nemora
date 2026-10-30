# `nemora.validation`

`nemora.validation` compares a plot-conditioned synthetic stand with held-out
inventory plots. It fits a DBH distribution and a height–DBH relation on
training trees, generates a seeded one-hectare-equivalent stand, and reports
metrics, distribution distance, species mix, and plot-resampling uncertainty.

This is a validation layer, not a spatial tree map or an operational planning
tool.

```{automodule} nemora.validation
:members:
:undoc-members:
:show-inheritance:
```

```{seealso}
- [`docs/howto/stand_validation.md`](../howto/stand_validation.md) for the
  workflow and metric definitions.
- [`docs/howto/plot_selection.md`](../howto/plot_selection.md) for the plot and
  tree-detail input.
```
