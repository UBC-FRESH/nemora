# Fit HPS Inventories

This guide walks through fitting the horizontal point sampling (HPS) workflow using the
`dbhdistfit` CLI and Python API. It extends the weighted estimator described in the UBC FRESH Lab
manuscript and relies on the probability distributions catalogued in
{doc}`reference/distributions`.

## Prerequisites

- An HPS tally file with columns `dbh_cm` (bin midpoints) and `tally` (per-plot counts).
- The basal area factor (`BAF`) used during cruise design.
- Python environment with `dbhdistfit` installed. One approach:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## CLI Workflow

1. Inspect available distributions:

```bash
dbhdistfit registry
```

2. Fit the HPS inventory with the default candidate set (`weibull`, `gamma`):

```bash
dbhdistfit fit-hps data/hps_tally.csv --baf 2.0
```

Pass additional `--distribution` options (planned) to try alternative PDFs as the CLI evolves.

To explore the public PSP example bundle prepared with
`scripts/prepare_hps_dataset.py`, start with one of the manifests in
`examples/hps_baf12`:

```bash
dbhdistfit fit-hps examples/hps_baf12/4000002_PSP1_v1_p1.csv --baf 12
```

## Python API Example

```python
import pandas as pd
from dbhdistfit.workflows import fit_hps_inventory

data = pd.read_csv("data/hps_tally.csv")
results = fit_hps_inventory(
    dbh_cm=data["dbh_cm"].to_numpy(),
    tally=data["tally"].to_numpy(),
    baf=2.0,
    distributions=["weibull", "gamma", "gb2"],
)

best = min(results, key=lambda r: r.gof["rss"])
print(best.distribution, best.parameters)
```

`fit_hps_inventory` expands tallies to stand tables, applies the HPS compression factors as
weights, and auto-generates starting values. Override the defaults through `FitConfig` for
specialised scenarios.

### Reference Fit (BC PSP 4000002-PSP1)

Using the public bundle prepared in {doc}`howto/hps_dataset`, `fit_hps_inventory`
identifies the Weibull distribution as the best fit for plot `4000002_PSP1_v1_p1`
with BAF 12. This mirrors the methodology from the EarthArXiv preprint of the HPS
manuscript (Paradis, 2025). The regression test in `tests/test_hps_parity.py`
locks these targets:

| Metric | Value |
| --- | --- |
| Distribution | `weibull` |
| RSS | `4.184770291568501e+07` |
| Parameters | `a=2.762844978640213`, `beta=13.778112123083137`, `s=69732.71124303175` |

Re-run the check locally with:

```bash
pytest tests/test_hps_parity.py
```

### Censored Meta-Plot Demo

For a pooled view, the notebook `examples/hps_bc_psp_demo.ipynb` aggregates the public BC PSP files,
censors stems below 9 cm, and fits the censored workflow. This serves as a deployment example on new
data rather than a reproduction of the manuscript figures. Run the notebook (or
`pytest tests/test_censored_workflow.py`) to verify the gamma fit parameters recorded in the summary
table.

### Parity Summary

- Regression tests `tests/test_hps_parity.py` and `tests/test_censored_workflow.py` lock the
  manuscript-aligned Weibull and censored gamma fits.
- `tests/test_cli.py::test_fit_hps_command_outputs_weibull_first` exercises the Typer CLI against
  the PSP tallies to ensure the command-line workflow mirrors the Python API.
- `examples/hps_bc_psp_demo.ipynb` demonstrates the workflow on the public BC PSP dataset, while
  `examples/hps_parity_reference.ipynb` runs the parity analysis on the manuscript meta-plot dataset
  included under `examples/data/reference_hps/binned_meta_plots.csv` (see Figure 1). The notebook also
  exports `docs/_static/reference_hps_parity_table.csv` summarising RSS, AICc, chi-square, and
  parameter deltas for each meta-plot.

.. figure:: /_static/reference_hps_parity.png
   :alt: Comparison of size-biased and weighted fits for the manuscript meta-plots.
   :width: 100%

   Figure 1 — Size-biased control vs. weighted `dbhdistfit` curves for the manuscript meta-plots. The
   dashed line shows residuals on the HPS tally scale.

## Diagnostics

- Inspect `result.diagnostics["residuals"]` for shape or bias.
- Compare `result.gof` metrics (RSS, AICc when available) across candidates.
- Plot the empirical stand table alongside fitted curves to confirm agreement with the manuscript
  workflow.

## Next Steps

- Expose distribution filters and parameter previews in the CLI.
- Add worked examples for censored inventories and DataLad-backed datasets.
- Integrate notebook tutorials mirroring the published reproducibility bundles.
- Expand FAIR dataset coverage (see {doc}`howto/hps_dataset`) with additional PSP plots and
  censored variants to support end-to-end parity tests.
