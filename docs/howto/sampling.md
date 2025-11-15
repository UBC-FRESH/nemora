# Sampling Utilities (Draft)

The `nemora.sampling` module provides helpers for converting registered PDFs
into CDFs, drawing random variates, sampling fitted mixtures, and bootstrapping
stand tables from `nemora.distfit` results.

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

## Sample from a distribution

```python
from nemora.sampling import sample_distribution

draws = sample_distribution("gamma", {"beta": 4.0, "p": 3.0, "s": 1.0}, size=500)
```

Distributions with closed-form inverse CDFs (Weibull, exponential, Pareto,
uniform, lognormal) use analytic inversion internally for improved accuracy.

## Sample from a mixture fit

```python
from nemora.distfit import MixtureComponentFit, MixtureFitResult
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
draws = sample_mixture_fit(mixture, size=1000)
```

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
