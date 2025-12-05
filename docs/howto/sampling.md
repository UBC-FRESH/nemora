# Sampling Utilities (Draft)

The `nemora.sampling` module provides helpers for converting registered PDFs
into CDFs, drawing random variates, sampling fitted mixtures, and bootstrapping
stand tables from `nemora.fit` results.

## Convert a PDF to a CDF

```python
import numpy as np

from nemora.sampling import pdf_to_cdf

cdf = pdf_to_cdf("weibull", {"a": 2.5, "beta": 12.0, "s": 1.0}, method="numeric")
x = np.linspace(0.0, 40.0, 100)
y = cdf(x)
```

When a distribution exposes an analytic CDF, `pdf_to_cdf(..., method="analytic")`
delegates to it; otherwise the helper falls back to numeric integration.

### Configuring numeric integration

`pdf_to_cdf` accepts a `SamplingConfig` so you can control grid density,
integration backend, and quadrature tolerances:

```python
from nemora.sampling import SamplingConfig, pdf_to_cdf

cfg = SamplingConfig(grid_points=2048, integration_method="quad", quad_rel_tol=1e-7)
cdf = pdf_to_cdf(
    "gamma",
    {"beta": 4.0, "p": 3.0, "s": 1.0},
    method="numeric",
    config=cfg,
)
```

The default uses a trapezoid grid; switching to `"quad"` delegates to
`scipy.integrate.quad` with the tolerances above. You can also set
`integration_method="simpson"` to integrate via Simpson's rule.
Set `cache_numeric_cdf=True` if you plan to evaluate the same numeric CDF repeatedly—Nemora caches
the computed grid/integral pair in-memory so sampling calls avoid rerunning the integrator.

## Sample from a distribution

```python
from nemora.sampling import sample_distribution

draws = sample_distribution("gamma", {"beta": 4.0, "p": 3.0, "s": 1.0}, size=500)
```

Distributions with closed-form inverse CDFs (Weibull, exponential, Pareto,
uniform, lognormal) use analytic inversion internally for improved accuracy.
Logistic/Fisk currently fall back to the numeric pathway described above; this is documented in
`notes/sampling_inverse_matrix.md` and will only change once synthesis requires a closed-form helper.

## Sample from a mixture fit

```python
from nemora.fit import MixtureComponentFit, MixtureFitResult
from nemora.sampling import sample_mixture_fit

components = [
    MixtureComponentFit(name="gamma", weight=0.6, parameters={"beta": 3.0, "p": 2.0}),
    MixtureComponentFit(name="gamma", weight=0.4, parameters={"beta": 8.0, "p": 5.0}),
]
mixture = MixtureFitResult(
    distribution="mixture",
    components=components,
    log_likelihood=-100.0,
    iterations=10,
    converged=True,
)
draws = sample_mixture_fit(mixture, size=1000, random_state=np.random.default_rng(42))
```

Pass a `numpy.random.Generator` (or integer seed) via `random_state` to obtain reproducible mixture draws.

Use the optional `lower`/`upper` parameters to truncate the draws (rejection sampling ensures all values fall inside the interval), and `weight_overrides=[...]` when you need to re-weight components dynamically (e.g., mixture-of-experts routing):

```python
draws = sample_mixture_fit(
    mixture,
    size=250,
    random_state=123,
    lower=5.0,
    upper=25.0,
    weight_overrides=[1.0, 0.0],  # ignore the second component for this scenario
)
```

Invalid overrides (negative weights, wrong lengths, or zero totals) raise `ValueError` so calling code can fall back gracefully.

## Bootstrap a fitted inventory

```python
import numpy as np
from nemora.core import FitResult
from nemora.sampling import BootstrapResult, bootstrap_inventory

fit = FitResult(distribution="gamma", parameters={"beta": 5.0, "p": 2.5, "s": 1.0})
bins = np.array([10.0, 20.0, 30.0])
tallies = np.array([5, 3, 2], dtype=float)
result: BootstrapResult = bootstrap_inventory(
    fit,
    bins,
    tallies,
    resamples=5,
    sample_size=25,
    return_result=True,
)
samples = result.samples
stacked = result.stacked()
```

Passing `return_result=True` yields a `BootstrapResult` containing the sampled
arrays and metadata (distribution, parameters, bins, tallies, RNG seed).

.. warning::
   These APIs are experimental. Expect refinements (additional configuration,
   performance tuning) as we integrate them with downstream modules.

## Bootstrap result metadata

`bootstrap_inventory(..., return_result=True)` returns a `BootstrapResult`
containing:

- `samples`: list of `(bin, draw)` arrays
- `distribution` / `parameters`: metadata from the fit
- `bins`, `tallies`, `resamples`, `sample_size`, `rng_seed`
- Convenience helpers: `stacked()` to concatenate samples and `to_dataframe()` to return a DataFrame with `resample`, `bin`, and `draw` columns.

Use the metadata when passing bootstrap outputs into synthesis or the simulation module.

## Using bootstrap outputs downstream

- **Synthesis (`nemora.synthesis`)** expects paired `(bin, draw)` arrays plus the originating fit metadata so stem or
  stand generators can report provenance. Use
  `nemora.synthesis.helpers.bootstrap_payload(result)` to obtain both the stacked array and a
  `pandas.DataFrame` with metadata attached in `frame.attrs["nemora_bootstrap"]`.
  Prefer the CLI helper (`nemora sampling-describe-bootstrap <stand-table.csv>`) when you want to
  preview metadata from an existing stand table or emit JSON for automation.
- **Simulation (`nemora.simulation`)** can persist the entire `BootstrapResult` (including RNG seed) to regenerate
  uncertainty studies or re-run Monte Carlo workflows deterministically.
- **DBH helpers** use `nemora.sampling.helpers.bootstrap_dbh_vectors(...)` (or the CLI shown below) to produce per-stand DBH arrays + metadata suitable for synthesis/simulation. See `docs/examples/faib_manifest_parquet.md` for an end-to-end manifest walkthrough.
- **Ingest + benchmarking notebooks** should write the stacked output to Parquet so future steps can
  slice by distribution, BAF, or inventory metadata without re-running the bootstrap sampling step.

Downstream modules should rely on the metadata provided here rather than reconstructing provenance
manually.

## Export DBH vectors for synthesis

Turn a `BootstrapResult` into per-resample DBH vectors (and an optional long-form table) via the helper:

```python
from nemora.sampling import bootstrap_dbh_vectors

payload = bootstrap_dbh_vectors(result, stand_id="psp-stand-001")
print(payload.dbh_vectors[0][:5])
payload.frame.head()
```

The helper preserves metadata (`distribution`, `parameters`, `bins`, `tallies`, RNG seed) and annotates each row with the stand identifier plus the original tally-derived weights. When you prefer a no-code path, the CLI mirrors this workflow:

```bash
nemora sampling-export-bootstrap-dbh tests/fixtures/hps_psp_stand_table.csv \
    --stand-id psp-stand-001 \
    --output tmp/bootstrap_dbh.json \
    --table-output tmp/bootstrap_dbh.parquet \
    --resamples 3 \
    --sample-size 25 \
    --seed 2025
```

The JSON file captures per-resample DBH arrays and metadata, while the table export (Parquet or CSV) stores every `(resample, bin, dbh)` row for downstream analysis.

## Sampling directly from ingest-created manifests

After generating manifests (CSV + Parquet by default) via `nemora faib-manifest`,
select an entry, fit a distribution, and draw samples while tuning the numeric
integration settings. Use `--no-parquet` if you prefer CSV-only outputs.
`docs/examples/faib_manifest_parquet.md` contains a complete example that:

1. Loads the Parquet manifest and resolves the stand-table path.
2. Wraps the DBH bins/tallies in an `InventorySpec`.
3. Fits a distribution via `fit_inventory`.
4. Calls `sample_distribution(..., config=SamplingConfig(...))` to test
   trapezoid/Simpson/quad integration modes, grid densities, and cache settings.

Use this pattern when validating ingest outputs or when you need to benchmark how
numeric integration tolerances impact downstream sampling accuracy.
