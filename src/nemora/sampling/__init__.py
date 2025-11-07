"""Sampling utilities built on top of the Nemora distribution registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..core import FitResult, MixtureFitResult
from ..distfit.mixture import sample_mixture as distfit_sample_mixture
from ..distributions import Pdf, get_distribution

__all__ = [
    "SamplingConfig",
    "pdf_to_cdf",
    "sample_distribution",
    "sample_mixture_fit",
    "bootstrap_inventory",
]


@dataclass(slots=True)
class SamplingConfig:
    """Configuration controlling numerical inversion/sampling behaviour."""

    grid_points: int = 512
    integration_method: Literal["trapezoid", "simpson"] = "trapezoid"


def _numeric_cdf(xs: np.ndarray, pdf_values: np.ndarray) -> np.ndarray:
    if xs.size < 2:
        return np.zeros_like(xs, dtype=float)
    diffs = np.diff(xs)
    integrand = 0.5 * (pdf_values[:-1] + pdf_values[1:]) * diffs
    cdf = np.concatenate(([0.0], np.cumsum(integrand)))
    cdf = np.clip(cdf, 0.0, None)
    total = float(cdf[-1]) if cdf.size else 1.0
    if total > 0:
        cdf /= total
    return cdf


def pdf_to_cdf(
    distribution: str,
    params: Mapping[str, float],
    *,
    method: Literal["analytic", "numeric"] = "analytic",
    grid: np.ndarray | None = None,
    config: SamplingConfig | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable CDF for the requested distribution."""

    dist = get_distribution(distribution)
    if method == "analytic" and dist.cdf is not None:
        cdf_callable: Pdf = dist.cdf
        return lambda values: cdf_callable(np.asarray(values, dtype=float), params)

    cfg = config or SamplingConfig()
    if grid is None:
        upper = params.get("beta") or params.get("scale") or 200.0
        grid = np.linspace(0.0, float(upper) * 5.0, cfg.grid_points)
    grid = np.asarray(grid, dtype=float)
    pdf_vals = dist.pdf(grid, params)
    cdf_vals = _numeric_cdf(grid, pdf_vals)

    def numeric_cdf(values: np.ndarray) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        return np.interp(vals, grid, cdf_vals, left=0.0, right=1.0)

    return numeric_cdf


def sample_distribution(
    distribution: str,
    params: Mapping[str, float],
    size: int,
    *,
    random_state: np.random.Generator | None = None,
    config: SamplingConfig | None = None,
) -> np.ndarray:
    """Draw samples from a registered distribution."""
    rng = random_state or np.random.default_rng()
    dist = get_distribution(distribution)
    if dist.cdf is not None:
        cfg = config or SamplingConfig()
        upper = params.get("beta") or params.get("scale") or 200.0
        xs = np.linspace(0.0, float(upper) * 5.0, cfg.grid_points)
        cdf_vals = dist.cdf(xs, params)
    else:
        cfg = config or SamplingConfig()
        xs = np.linspace(0.0, 200.0, cfg.grid_points)
        cdf_fn = pdf_to_cdf(distribution, params, method="numeric", grid=xs, config=cfg)
        cdf_vals = cdf_fn(xs)
    cdf_vals = np.clip(cdf_vals, 0.0, 1.0)
    u = rng.random(size)
    return np.interp(u, cdf_vals, xs)


def sample_mixture_fit(
    fit: MixtureFitResult,
    size: int,
    *,
    random_state: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample from a :class:`MixtureFitResult`."""
    rng = random_state or np.random.default_rng()
    return distfit_sample_mixture(size, fit.components, random_state=rng)


def bootstrap_inventory(
    fit: FitResult,
    bins: np.ndarray,
    tallies: np.ndarray,
    *,
    resamples: int,
    sample_size: int,
    random_state: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Bootstrap stand tables by sampling from a fitted distribution."""
    rng = random_state or np.random.default_rng()
    tallies = np.asarray(tallies, dtype=float)
    bins = np.asarray(bins, dtype=float)
    if tallies.sum() <= 0:
        raise ValueError("Tallies must sum to a positive value.")
    samples: list[np.ndarray] = []
    for _ in range(resamples):
        weights = tallies / tallies.sum()
        indices = rng.choice(np.arange(bins.size), size=sample_size, p=weights, replace=True)
        selected_bins = bins[indices]
        draws = sample_distribution(
            fit.distribution,
            fit.parameters,
            size=sample_size,
            random_state=rng,
        )
        samples.append(np.column_stack((selected_bins, draws)))
    return samples
