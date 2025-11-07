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

## Sample from a distribution

```python
from nemora.sampling import sample_distribution

draws = sample_distribution("gamma", {"beta": 4.0, "p": 3.0, "s": 1.0}, size=500)
```

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
from nemora.sampling import bootstrap_inventory

fit = FitResult(distribution="gamma", parameters={"beta": 5.0, "p": 2.5, "s": 1.0})
bins = np.array([10.0, 20.0, 30.0])
tallies = np.array([5, 3, 2], dtype=float)
samples = bootstrap_inventory(fit, bins, tallies, resamples=5, sample_size=25)
```

.. warning::
   These APIs are experimental. Expect refinements (additional configuration,
   performance tuning) as we integrate them with downstream modules.
