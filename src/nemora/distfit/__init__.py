"""Shared fitting strategies for nemora."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from inspect import Parameter, Signature
from typing import Any, cast

import numpy as np
from lmfit import Model
from scipy.optimize import curve_fit

from ..core import FitResult, InventorySpec
from ..distributions import Distribution, get_distribution
from .grouped import get_grouped_estimator

Objective = Callable[[np.ndarray, np.ndarray, Mapping[str, float]], float]


@dataclass(slots=True)
class FitConfig:
    distribution: str
    initial: dict[str, float]
    bounds: dict[str, tuple[float | None, float | None]] | None = None
    weights: np.ndarray | None = None


def _moment_summary(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    if x.size == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.0
    weights = np.clip(y, 1e-8, None)
    total = float(weights.sum())
    mean = float(np.sum(weights * x) / total) if total > 0 else float(np.mean(x))
    if total > 0:
        variance = float(np.sum(weights * np.square(x - mean)) / total)
    else:
        variance = float(np.var(x))
    variance = max(variance, 1e-6)
    std = float(np.sqrt(variance))
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    return mean, variance, std, xmin, xmax


def _positive(value: float, fallback: float = 1.0) -> float:
    return float(value) if value > 0 else float(fallback)


def _default_bounds(parameters: tuple[str, ...]) -> dict[str, tuple[float | None, float | None]]:
    lower_positive = {
        "a",
        "b",
        "beta",
        "p",
        "q",
        "sigma2",
        "d",
        "u",
        "v",
        "df",
        "s",
        "alpha",
        "scale",
    }
    bounds: dict[str, tuple[float | None, float | None]] = {}
    for name in parameters:
        lower: float | None = None
        upper: float | None = None
        if name in lower_positive or name.startswith("omega"):
            lower = 1e-6
        if name.startswith("omega"):
            upper = 1.0
        if lower is not None or upper is not None:
            bounds[name] = (lower, upper)
    return bounds


def _residual_summary(residuals: np.ndarray) -> dict[str, float]:
    if residuals.size == 0:
        return {}
    abs_res = np.abs(residuals)
    return {
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals, ddof=0)),
        "max_abs": float(np.max(abs_res)),
        "median_abs": float(np.median(abs_res)),
    }


def _gof_metrics(
    y: np.ndarray,
    fitted: np.ndarray,
    residuals: np.ndarray,
    rss: float,
    param_count: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {"rss": float(rss)}
    n = y.size
    if n == 0:
        return metrics

    rss_safe = max(float(rss), 1e-12)
    k = max(param_count, 1)
    metrics["aic"] = float(n * np.log(rss_safe / n) + 2 * k)
    denom = n - k - 1
    if denom > 0:
        metrics["aicc"] = float(metrics["aic"] + (2 * k * (k + 1)) / denom)
    metrics["bic"] = float(n * np.log(rss_safe / n) + k * np.log(n))

    expected = np.clip(fitted, 1e-8, None)
    chisq = float(np.sum(np.square(residuals) / expected))
    metrics["chisq"] = chisq
    return metrics


def default_fit_config(name: str, x: np.ndarray, y: np.ndarray) -> FitConfig:
    """Construct a heuristic FitConfig for the requested distribution."""
    dist = get_distribution(name)
    mean, variance, std, xmin, xmax = _moment_summary(x, y)
    scale = float(np.max(y)) if y.size else 1.0
    initial: dict[str, float] = {}
    omega_params = [param for param in dist.parameters if param.startswith("omega")]
    omega_components = len(omega_params) + (1 if omega_params else 0)

    if "s" in dist.parameters:
        initial["s"] = _positive(scale, 1.0)

    for param in dist.parameters:
        if param == "s":
            continue
        guess = 1.0
        if param == "a":
            guess = 2.0
        elif param == "b":
            if name.lower() in {"pareto", "p"}:
                guess = _positive(min(xmin * 0.9, xmax), 1.0)
            else:
                guess = _positive(xmax * 1.1, 1.0)
        elif param == "beta":
            guess = _positive(max(mean, std), 1.0)
        elif param in {"p", "q"}:
            guess = 2.0
        elif param == "mu":
            guess = float(np.log(_positive(mean, 1.0)))
        elif param == "sigma2":
            guess = _positive(std**2, 0.5)
        elif param == "alpha":
            guess = 1.5
        elif param == "loc":
            guess = xmin
        elif param == "scale":
            guess = _positive(std, 1.0)
        elif param == "d":
            guess = 2.0
        elif param == "u":
            guess = 5.0
        elif param == "v":
            guess = 10.0
        elif param == "df":
            guess = 6.0
        elif param.startswith("omega"):
            component_count = omega_components if omega_components else 2
            guess = 1.0 / component_count
        initial[param] = _positive(guess, 1.0)

    bounds = _default_bounds(dist.parameters)
    return FitConfig(distribution=name, initial=initial, bounds=bounds or None)


def _curve_fit_distribution(
    x: np.ndarray,
    y: np.ndarray,
    distribution: Distribution,
    config: FitConfig,
) -> FitResult:
    """Fit a distribution using SciPy curve_fit with optional weights."""

    def wrapped(x_vals: np.ndarray, *params: float) -> np.ndarray:
        values = dict(zip(distribution.parameters, params, strict=False))
        return distribution.pdf(x_vals, values)

    p0 = [config.initial.get(name, 1.0) for name in distribution.parameters]
    sigma = config.weights if config.weights is not None else None
    params, cov = curve_fit(
        wrapped,
        x,
        y,
        p0=p0,
        sigma=sigma,
        maxfev=int(2e5),
    )
    param_dict = dict(zip(distribution.parameters, params, strict=False))
    fitted = distribution.pdf(x, param_dict)
    residuals = y - fitted
    rss = float(np.sum(np.square(residuals)))
    gof = _gof_metrics(y, fitted, residuals, rss, len(distribution.parameters))
    diagnostics = {
        "fitted": fitted,
        "residuals": residuals,
        "residual_summary": _residual_summary(residuals),
    }
    return FitResult(
        distribution=distribution.name,
        parameters=param_dict,
        covariance=cov,
        gof=gof,
        diagnostics=diagnostics,
    )


def fit_with_lmfit(
    x: np.ndarray,
    y: np.ndarray,
    distribution: Distribution,
    config: FitConfig,
) -> FitResult:
    """Fit using lmfit Model for more advanced scenarios (e.g., truncation)."""

    param_names = list(distribution.parameters)

    def func(x_vals: np.ndarray, **params: float) -> np.ndarray:
        return distribution.pdf(x_vals, params)

    signature = Signature(
        parameters=[
            Parameter("x_vals", kind=Parameter.POSITIONAL_OR_KEYWORD),
            *(Parameter(name, kind=Parameter.KEYWORD_ONLY) for name in param_names),
        ]
    )
    cast(Any, func).__signature__ = signature

    model = Model(func)
    params = model.make_params()
    for name in distribution.parameters:
        start = config.initial.get(name, 1.0)
        params[name].set(value=start)
        if config.bounds and name in config.bounds:
            lower, upper = config.bounds[name]
            params[name].set(min=lower, max=upper)
    weights = config.weights
    result = model.fit(y, params, x_vals=x, weights=weights)
    param_dict = {name: result.params[name].value for name in distribution.parameters}
    cov = result.covar
    fitted = result.best_fit
    residuals = result.residual
    rss = float(np.sum(np.square(residuals)))
    gof = _gof_metrics(y, fitted, residuals, rss, len(distribution.parameters))
    # Overwrite AIC/BIC with lmfit's values when available to keep parity.
    if hasattr(result, "aic"):
        gof["aic"] = float(result.aic)
    if hasattr(result, "bic"):
        gof["bic"] = float(result.bic)
    return FitResult(
        distribution=distribution.name,
        parameters=param_dict,
        covariance=cov,
        gof=gof,
        diagnostics={
            "result": result,
            "fitted": fitted,
            "residuals": residuals,
            "residual_summary": _residual_summary(residuals),
        },
    )


def fit_inventory(
    inventory: InventorySpec,
    distributions: Iterable[str],
    configs: Mapping[str, FitConfig],
    *,
    fitter: Callable[
        [np.ndarray, np.ndarray, Distribution, FitConfig],
        FitResult,
    ] = _curve_fit_distribution,
) -> list[FitResult]:
    """Fit a collection of candidate distributions to an inventory."""
    x = np.asarray(inventory.bins, dtype=float)
    y = np.asarray(inventory.tallies, dtype=float)
    configs = dict(configs)
    results: list[FitResult] = []
    grouped = bool(inventory.metadata.get("grouped"))
    for name in distributions:
        dist = get_distribution(name)
        config = configs.get(name)
        if config is None:
            config = default_fit_config(name, x, y)
            configs[name] = config
        estimator = get_grouped_estimator(name) if grouped else None
        if estimator is not None:
            try:
                results.append(estimator(inventory, config))
                continue
            except ValueError:
                pass
        results.append(fitter(x, y, dist, config))
    return results


from .mixture import (  # noqa: E402
    MixtureComponentSpec,
    fit_mixture_grouped,
    fit_mixture_samples,
    mixture_cdf,
    mixture_pdf,
    sample_mixture,
)

__all__ = [
    "FitConfig",
    "default_fit_config",
    "fit_inventory",
    "MixtureComponentSpec",
    "fit_mixture_grouped",
    "fit_mixture_samples",
    "mixture_pdf",
    "mixture_cdf",
    "sample_mixture",
]
