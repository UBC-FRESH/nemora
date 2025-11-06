# Overview

`dbhdistfit` streamlines fitting probability density functions to diameter-at-breast-height (DBH)
inventories. The package packages workflows for:

- horizontal point sampling (HPS) tallies with size-bias corrections handled through weighting, and
- fixed-area inventories where left/right censoring implies truncated support.

The project grew out of research reproducibility stacks in the UBC FRESH Lab, and is now being
generalised into a reusable toolkit with Python, CLI, and R interfaces.

```{note}
This documentation is pre-alpha. Expect rapid iteration while the API stabilises.
```

## Relationship to prior work
Several R libraries—most prominently
[`ForestFit`](https://cran.r-project.org/package=ForestFit)—already support extensive parametric and
mixture-based DBH modelling. `dbhdistfit` is positioned as a complementary, workflow-oriented
toolkit:

- Horizontal point sampling (HPS) weighting, censored workflows, and manuscript parity datasets are
  included out of the box.
- The Python-first stack (Typer CLI, pandas integration, and planned `dbhdistfitr` bridge) enables
  the same pipelines to run in notebooks, batch jobs, or mixed-language projects.
- Candidate features that originated in ForestFit (finite mixtures, JSB variations, EM initialisers)
  are tracked in `candidate-import-from-ForestFit-features.md` so upstream credit is explicit while
  we extend the Python implementation.

Future roadmap items include mixture and piecewise models inspired by the forestry literature cited
in the README. Contributions comparing the toolkits or proposing cross-language examples are very
welcome.

## Current Caveats
- Grouped Weibull currently relies on a least-squares fit (matching the manuscript parity workflow),
  while Johnson SB and Birnbaum-Saunders use grouped maximum-likelihood optimisation with numerical
  Hessian estimates. We plan to consolidate these approaches once the MLE path reproduces the legacy
  parity results.
- Johnson SB grouped fits fall back to SciPy’s continuous MLE when the grouped optimiser fails, so
  diagnostics will report a non-converged status in those cases.
- Generalised secant (GSM) mixtures are exposed as `gsmN` distributions; while the grouped estimator
  supports any component count `N >= 2`, convergence behaviour for large `N` still needs empirical
  evaluation.
