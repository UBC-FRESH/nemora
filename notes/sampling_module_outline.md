# Sampling Module Outline

Date: 2025-11-07
Status: Draft — first pass at `nemora.sampling` interfaces.

## Goals

- Provide reusable helpers that turn registered PDFs into CDFs (analytic when
  closed-form is available, numerical otherwise).
- Support bootstrap/Monte Carlo sampling from fitted distributions and finite
  mixtures produced by `nemora.distfit`.
- Keep the public API Pydantic/typing-friendly, mirroring the approach used in
  `nemora.distfit`.

## Proposed API Surface

```python
# src/nemora/sampling/__init__.py
from nemora.core import FitResult, MixtureFitResult

def pdf_to_cdf(
    distribution: str,
    params: Mapping[str, float],
    *,
    method: Literal["analytic", "numeric"] = "analytic",
    grid: np.ndarray | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    ...

def sample_distribution(
    distribution: str,
    params: Mapping[str, float],
    size: int,
    *,
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    ...

def sample_mixture_fit(
    fit: MixtureFitResult,
    size: int,
    *,
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    ...

def bootstrap_inventory(
    fit: FitResult,
    bins: np.ndarray,
    tallies: np.ndarray,
    *,
    resamples: int,
    sample_size: int,
    random_state: np.random.Generator | None = None,
) -> list[np.ndarray]:
    ...
```

## Implementation Notes

- For distributions with known CDFs (registered `cdf` callable), `pdf_to_cdf`
  should delegate to the analytic form. When absent, fall back to numeric
  integration (Simpson / trapezoidal rule) across a supplied grid.
- `sample_distribution` should honour the `distribution.cdf` when available.
  Otherwise use inverse transform sampling against the numeric CDF.
- `sample_mixture_fit` can reuse existing utilities in `nemora.distfit.mixture`
  but should expose a friendlier API for notebooks/CLI commands.
- `bootstrap_inventory` ought to resample tallies with replacement,
  then call `sample_distribution` for each resample to propagate uncertainty.
  Return a list/array of sampled stand tables for downstream metrics.

## Tests & Fixtures

- Create deterministic sampling tests using fixed random seeds and simple
  distributions (exponential, gamma).
- Add regression tests for mixtures mirroring the existing fixture data,
  ensuring weights remain normalised and component distributions are sampled.
- Property-based tests (Hypothesis) can validate that numeric CDF integration
  matches analytic counterparts within tolerance.

## Documentation TODOs

- Draft `docs/howto/sampling.md` referencing the new helpers once implemented.
- Update README/roadmap to surface the sampling module as it lands.
