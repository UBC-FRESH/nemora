# Register Custom Distributions

`nemora` ships with a core registry (Weibull, Gamma, Johnson SB, Birnbaum-Saunders, and the
generalized beta and generalized secant families). Generalised secant entries are available as
`gsmN` for `N >= 2` (e.g. `gsm3`, `gsm5`) and can be extended via the same plugin hooks. The registry can be
extended at runtime. You can plug in additional probability density functions either by calling the
Python API directly, by installing a plugin that exposes an entry point, or by pointing the toolkit
at a YAML configuration file.

## Python API

```python
from typing import Mapping

import numpy as np

from nemora.distributions import Distribution, register_distribution


def truncated_normal_pdf(x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    mean = params["mu"]
    sigma = params["sigma"]
    scale = params.get("s", 1.0)
    z = (x - mean) / sigma
    return scale * np.exp(-0.5 * z**2) / (sigma * np.sqrt(2 * np.pi))

register_distribution(
    Distribution(
        name="truncnorm",
        parameters=("mu", "sigma", "s"),
        pdf=truncated_normal_pdf,
        notes="Example plugin distribution."
    ),
    overwrite=True,
)
```

Once registered, the new distribution appears in `nemora registry` and is available to the HPS
workflow.

## Entry points (recommended for plugins)

Third-party packages can expose a callable via the `nemora.distributions` entry-point group. The
callable should return either a single `Distribution` or an iterable of them. Example
`pyproject.toml` snippet:

```toml
[project.entry-points."nemora.distributions"]
truncnorm = "my_plugin.distributions:create_truncnorm"
```

Within `my_plugin/distributions.py`:

```python
from nemora.distributions import Distribution

from ._pdf import truncated_normal_pdf


def create_truncnorm() -> Distribution:
    return Distribution(
        name="truncnorm",
        parameters=("mu", "sigma", "s"),
        pdf=truncated_normal_pdf,
        notes="Truncated Normal distribution shipped by my_plugin.",
    )
```

`nemora` discovers the entry point at import time and registers the distribution automatically.

## YAML configuration

Supply a YAML file that references callables or defines distributions inline. Point the environment
variable `DBHDISTFIT_DISTRIBUTIONS` at one or more files separated by the OS path separator, or copy
files into `config/distributions/` inside a source checkout.

```yaml
metadata:
  title: "Custom additions"
  version: "1.0"
distributions:
  - name: truncnorm
    parameters: ["mu", "sigma", "s"]
    pdf: my_plugin.pdf:truncated_normal_pdf
    notes: "Truncated normal PDF from my_plugin."
  - callable: my_plugin.factory:create_mixture
    overwrite: true
    args: [2]
```

Each item either provides a `callable` (invoked with optional `args`/`kwargs`) or an inline
definition with `name`, `parameters`, and a `pdf` import path. If the callable returns multiple
`Distribution` instances they are all registered. The loader skips invalid entries but reports
warnings via the logger to aid debugging.

## Inspecting the registry

```bash
nemora registry
```

The CLI lists built-in and plugin distributions in alphabetical order. Use `nemora --verbose
distribution-name` (planned) to inspect parameters and metadata.

## Testing custom distributions

When adding a new distribution, unit tests should exercise the PDF across representative DBH vectors
and ensure parameters remain valid for grouped tallies. The helper `register_distribution` accepts
`overwrite=True` so test suites can inject temporary definitions without polluting the global
registry.
