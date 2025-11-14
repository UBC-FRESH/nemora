"""Sampling utilities built on top of the Nemora distribution registry."""

from __future__ import annotations

import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import integrate

from ..core import FitResult, MixtureFitResult
from ..distfit.mixture import sample_mixture as distfit_sample_mixture
from ..distributions import Pdf, get_distribution

__all__ = [
    "BootstrapResult",
    "SamplingConfig",
    "pdf_to_cdf",
    "sample_distribution",
    "sample_mixture_fit",
    "bootstrap_inventory",
]


@dataclass(slots=True)
class SamplingConfig:
    """Configuration controlling numerical inversion/sampling behaviour."""

    grid_points: int = 2048
    support_multiplier: float = 5.0
    integration_method: Literal["trapezoid", "quad"] = "trapezoid"
    quad_abs_tol: float = 1e-8
    quad_rel_tol: float = 1e-6


@dataclass(slots=True)
class BootstrapResult:
    """Container returned by :func:`bootstrap_inventory` when metadata is requested."""

    samples: list[np.ndarray]
    distribution: str
    parameters: Mapping[str, float]
    bins: np.ndarray
    tallies: np.ndarray
    resamples: int
    sample_size: int
    rng_seed: int | None = None

    def stacked(self) -> np.ndarray:
        """Return all bootstrap samples concatenated along axis 0."""

        if not self.samples:
            return np.empty((0, 2), dtype=float)
        return np.concatenate(self.samples, axis=0)


def _grid_from_params(
    params: Mapping[str, float],
    *,
    cfg: SamplingConfig,
    grid: np.ndarray | None = None,
) -> np.ndarray:
    if grid is not None:
        return np.asarray(grid, dtype=float)
    loc = float(params.get("s") or params.get("loc") or 0.0)
    scale_like = params.get("beta") or params.get("scale") or params.get("sigma") or 40.0
    upper = loc + float(scale_like) * cfg.support_multiplier
    return np.linspace(loc, upper, cfg.grid_points)


def _numeric_cdf(
    xs: np.ndarray,
    pdf_callable: Callable[[np.ndarray], np.ndarray],
    *,
    cfg: SamplingConfig,
) -> np.ndarray:
    if xs.size < 2:
        return np.zeros_like(xs, dtype=float)
    if cfg.integration_method == "trapezoid":
        pdf_values = pdf_callable(xs)
        cdf = integrate.cumulative_trapezoid(pdf_values, xs, initial=0.0)
    else:

        def scalar_pdf(value: float) -> float:
            arr = np.asarray([value], dtype=float)
            return float(pdf_callable(arr)[0])

        cdf = np.zeros_like(xs, dtype=float)
        for idx in range(1, xs.size):
            area, _ = integrate.quad(
                scalar_pdf,
                float(xs[idx - 1]),
                float(xs[idx]),
                epsabs=cfg.quad_abs_tol,
                epsrel=cfg.quad_rel_tol,
            )
            cdf[idx] = cdf[idx - 1] + area
    cdf = np.clip(cdf, 0.0, None)
    total = float(cdf[-1]) if cdf.size else 1.0
    if total > 0:
        cdf /= total
    return cdf


def _prepare_cdf_grid(
    distribution: str,
    params: Mapping[str, float],
    *,
    method: Literal["analytic", "numeric"],
    grid: np.ndarray | None,
    cfg: SamplingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    dist = get_distribution(distribution)
    xs = _grid_from_params(params, cfg=cfg, grid=grid)
    if method == "analytic" and dist.cdf is not None:
        cdf_vals = dist.cdf(xs, params)
    else:
        pdf_callable: Pdf = dist.pdf
        cdf_vals = _numeric_cdf(xs, lambda values: pdf_callable(values, params), cfg=cfg)
    cdf_vals = np.clip(cdf_vals, 0.0, 1.0)
    return xs, cdf_vals


def pdf_to_cdf(
    distribution: str,
    params: Mapping[str, float],
    *,
    method: Literal["analytic", "numeric"] = "analytic",
    grid: np.ndarray | None = None,
    config: SamplingConfig | None = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a callable CDF for the requested distribution."""

    cfg = config or SamplingConfig()
    dist = get_distribution(distribution)
    chosen_method = method
    if method == "analytic" and dist.cdf is None:
        chosen_method = "numeric"
    xs, cdf_vals = _prepare_cdf_grid(
        distribution,
        params,
        method=chosen_method,
        grid=grid,
        cfg=cfg,
    )

    def cdf_func(values: np.ndarray) -> np.ndarray:
        vals = np.asarray(values, dtype=float)
        return np.interp(vals, xs, cdf_vals, left=0.0, right=1.0)

    return cdf_func


def sample_distribution(
    distribution: str,
    params: Mapping[str, float],
    size: int,
    *,
    random_state: np.random.Generator | None = None,
    config: SamplingConfig | None = None,
) -> np.ndarray:
    """Draw samples from a registered distribution."""
    cfg = config or SamplingConfig()
    rng: np.random.Generator
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    elif isinstance(random_state, numbers.Integral):
        rng = np.random.default_rng(int(random_state))
    else:
        rng = random_state or np.random.default_rng()
    dist = get_distribution(distribution)
    method: Literal["analytic", "numeric"] = "analytic" if dist.cdf is not None else "numeric"
    xs, cdf_vals = _prepare_cdf_grid(
        distribution,
        params,
        method=method,
        grid=None,
        cfg=cfg,
    )
    u = rng.random(size)
    return np.interp(u, cdf_vals, xs)


def sample_mixture_fit(
    fit: MixtureFitResult,
    size: int,
    *,
    random_state: np.random.Generator | int | None = None,
) -> np.ndarray:
    """Sample from a :class:`MixtureFitResult`."""
    seed: int | None
    if isinstance(random_state, np.random.Generator):
        seed = int(random_state.integers(0, 2**32, dtype=np.uint64))
    elif isinstance(random_state, numbers.Integral):
        seed = int(random_state)
    else:
        seed = None
    return distfit_sample_mixture(size, fit.components, random_state=seed)


def bootstrap_inventory(
    fit: FitResult,
    bins: np.ndarray,
    tallies: np.ndarray,
    *,
    resamples: int,
    sample_size: int,
    random_state: np.random.Generator | numbers.Integral | None = None,
    return_result: bool = False,
) -> list[np.ndarray] | BootstrapResult:
    """Bootstrap stand tables by sampling from a fitted distribution.

    When ``return_result`` is ``True`` a :class:`BootstrapResult` with metadata is returned
    instead of a bare list of arrays.
    """
    rng: np.random.Generator
    seed_meta: int | None = None
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    elif isinstance(random_state, numbers.Integral):
        seed_meta = int(random_state)
        rng = np.random.default_rng(seed_meta)
    else:
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
    if return_result:
        return BootstrapResult(
            samples=samples,
            distribution=fit.distribution,
            parameters=dict(fit.parameters),
            bins=bins,
            tallies=tallies,
            resamples=resamples,
            sample_size=sample_size,
            rng_seed=seed_meta,
        )
    return samples
