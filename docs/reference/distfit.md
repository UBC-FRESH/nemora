# Distribution Fitting (`nemora.distfit`)

`nemora.distfit` hosts the shared estimators, grouped-data solvers, and helper utilities that power
the current alpha milestone. The module is designed to work with the canonical distribution registry
in `nemora.distributions` and the shared dataclasses in `nemora.core`.

## Key Concepts

- **`FitConfig`** encapsulates starting values, optional bounds, and weights used during optimisation.
- **`fit_inventory`** accepts an `InventorySpec` and evaluates one or more candidate distributions.
  Grouped estimators are selected automatically when `inventory.metadata["grouped"]` evaluates to
  `True`.
- **Grouped estimators** for Weibull, Johnson SB, Birnbaum–Saunders, and generalised secant mixtures
  live under `nemora.distfit.grouped`. They expose diagnostics describing the solver path
  (`grouped-ls`, `grouped-em`, `grouped-mle`).
- **Mixture utilities** (`fit_mixture_grouped`, `fit_mixture_samples`, `mixture_pdf`, `mixture_cdf`,
  `sample_mixture`) support finite mixtures with grouped tallies or sample-level data.

Module-level functions return `FitResult` or `MixtureFitResult` instances from `nemora.core`.

## API Reference

```{automodule} nemora.distfit
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} nemora.distfit.grouped
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} nemora.distfit.mixture
:members:
:undoc-members:
:show-inheritance:
```

.. todo:: Expand this page with grouped-fitting theory notes and worked examples once the ingest and sampling modules are in place.
