# Programmatic HPS Analysis

This guide shows how to drive the Horizontal Point Sampling (HPS) workflow directly from Python.
It highlights the richer goodness-of-fit metrics exposed by `fit_hps_inventory` and demonstrates
how to turn `FitResult` objects into tidy reports for downstream analysis.

## Prerequisites

- Install `nemora` in an environment with pandas and numpy available (`pip install -e ".[dev]"`).
- Obtain an HPS tally CSV with `dbh_cm`, `tally`, and the associated basal area factor (`baf`)
  metadata. The repository ships examples under `examples/hps_baf12/`.

```python
from pathlib import Path

import pandas as pd

from nemora.core import FitSummary, InventorySpec
from nemora.workflows import fit_hps_inventory

csv_path = Path("examples/hps_baf12/4000002_PSP1_v1_p1.csv")
data = pd.read_csv(csv_path)
dbh_cm = data["dbh_cm"].to_numpy()
tally = data["tally"].to_numpy()
baf = 12.0
```

## Fit candidate distributions

Pass the tallies into `fit_hps_inventory`. The helper expands the tallies to stand tables, applies
the HPS compression weights, and returns a list of `FitResult` objects – one entry per candidate.

```python
results = fit_hps_inventory(
    dbh_cm=dbh_cm,
    tally=tally,
    baf=baf,
    distributions=("weibull", "gamma", "gb2", "gsm3", "gsm6"),
)
```

Inspect the goodness-of-fit metrics (RSS, AIC/AICc, BIC, chi-square, KS, CvM, AD) and residual
summaries that are now populated automatically; any `gsmN` string (with `N ≥ 2`) will route through
the generalized secant grouped estimator:

```python
for result in results:
    gof = result.gof
    residuals = result.diagnostics["residual_summary"]
    print(
        f"{result.distribution:8s}"
        f" RSS={gof['rss']:.2f}"
        f" AICc={gof.get('aicc', float('nan')):.2f}"
        f" Chi^2={gof.get('chisq', float('nan')):.2f}"
        f" KS={gof.get('ks', float('nan')):.3f}"
        f" CvM={gof.get('cvm', float('nan')):.3f}"
        f" AD={gof.get('ad', float('nan')):.3f}"
        f" Max|res|={residuals.get('max_abs', float('nan')):.2f}"
    )
```

Choose the best candidate (e.g. by minimum RSS or AICc) and inspect the fitted parameters:

```python
best = min(results, key=lambda item: item.gof["rss"])
print(best.distribution, best.parameters)
```

## Summarise results as a table

`FitSummary` can collate the results into a tidy DataFrame for notebooks, CSV exports, or further
analysis.

```python
summary = FitSummary(
    inventory=InventorySpec(
        name="psp-4000002",
        sampling="hps",
        bins=dbh_cm,
        tallies=tally,
        metadata={"baf": baf, "source": csv_path.name},
    ),
    results=results,
    best=best,
)

summary_frame = summary.to_frame()
summary_frame.loc[:, ["distribution", "gof_rss", "gof_aicc", "gof_chisq"]]
```

Save the table or plot the residuals from `result.diagnostics["residuals"]` to match the parity
notebooks.

## Next steps

- Run `nemora fit-hps --show-parameters` for a CLI view that mirrors the report above.
- Use [the HPS dataset guide](hps_dataset.md) to fetch the manuscript dataset via
  `nemora fetch-reference-data` when DataLad is available.
- Combine the scripted workflow with `matplotlib` or `polars` to build automated QA reports for
  multiple plots.
