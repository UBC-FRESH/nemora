"""Finite mixture fitting utilities."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ..core import InventorySpec, MixtureComponentFit, MixtureFitResult
from ..distributions import get_distribution
from . import default_fit_config, fit_inventory


@dataclass(slots=True)
class MixtureComponentSpec:
    """Describe one component of a finite mixture."""

    distribution: str
    initial: dict[str, float] | None = None


def fit_mixture_grouped(
    bins: ArrayLike,
    tallies: ArrayLike,
    components: Sequence[MixtureComponentSpec],
    *,
    max_iter: int = 200,
    tol: float = 1e-6,
    min_weight: float = 1e-4,
    random_state: int | None = None,
) -> MixtureFitResult:
    """Fit a finite mixture to grouped (binned) tallies via EM."""

    if len(components) < 2:
        raise ValueError("Mixture fitting requires at least two component specifications.")

    rng = np.random.default_rng(random_state)
    x = np.asarray(bins, dtype=float)
    y = np.asarray(tallies, dtype=float)
    if x.size != y.size:
        raise ValueError("Bins and tallies must be the same length.")
    if np.any(y < 0):
        raise ValueError("Tallies must be non-negative.")
    total = float(np.sum(y))
    if total <= 0:
        raise ValueError("Tallies must sum to a positive value.")

    n_components = len(components)
    weights = np.full(n_components, 1.0 / n_components, dtype=float)

    parameter_sets: list[dict[str, float]] = []
    component_results: list[MixtureComponentFit] = []

    init_weights = []
    for index, spec in enumerate(components):
        if spec.initial:
            params = dict(spec.initial)
            init_weight = total / n_components
        else:
            params, init_weight = _initial_parameters(
                x, y, spec.distribution, index, n_components, rng
            )
        parameter_sets.append(params)
        component_results.append(
            MixtureComponentFit(name=spec.distribution, weight=weights[index], parameters=params)
        )
        init_weights.append(init_weight)

    init_weights_array = np.clip(np.asarray(init_weights, dtype=float), min_weight, None)
    init_total = float(np.sum(init_weights_array))
    if init_total > 0:
        weights = init_weights_array / init_total

    log_likelihood = -np.inf
    converged = False
    for iteration in range(1, max_iter + 1):
        pdf_matrix = np.vstack(
            [
                _evaluate_pdf(spec.distribution, x, params)
                for spec, params in zip(components, parameter_sets, strict=True)
            ]
        )
        weighted_pdf = weights[:, None] * pdf_matrix
        denom = np.clip(np.sum(weighted_pdf, axis=0), 1e-12, None)
        responsibilities = weighted_pdf / denom

        expected_tallies = responsibilities * y
        dirichlet_prior = 1.1
        posterior = expected_tallies.sum(axis=1) + (dirichlet_prior - 1.0)
        normaliser = total + n_components * (dirichlet_prior - 1.0)
        new_weights = np.clip(posterior / normaliser, min_weight, None)
        new_weights /= new_weights.sum()

        new_params: list[dict[str, float]] = []
        new_results: list[MixtureComponentFit] = []
        max_param_delta = 0.0
        for idx, (spec, expected, old_params) in enumerate(
            zip(components, expected_tallies, parameter_sets, strict=True)
        ):
            updated = _update_component(spec.distribution, x, expected, old_params)
            param_delta = max(
                abs(old_params.get(key, 0.0) - updated.parameters.get(key, 0.0))
                for key in updated.parameters
            )
            max_param_delta = max(max_param_delta, param_delta)
            new_params.append(updated.parameters)
            new_results.append(
                MixtureComponentFit(
                    name=spec.distribution,
                    weight=new_weights[idx],
                    parameters=updated.parameters,
                    gof=updated.gof,
                    diagnostics=updated.diagnostics,
                )
            )

        new_log_likelihood = float(np.sum(y * np.log(denom)))
        weight_delta = float(np.max(np.abs(new_weights - weights)))
        if max(weight_delta, max_param_delta, abs(new_log_likelihood - log_likelihood)) < tol:
            weights = new_weights
            parameter_sets = new_params
            component_results = new_results
            log_likelihood = new_log_likelihood
            converged = True
            iteration += 0  # keep iteration count consistent
            break

        weights = new_weights
        parameter_sets = new_params
        component_results = new_results
        log_likelihood = new_log_likelihood

    return MixtureFitResult(
        distribution="mixture",
        components=component_results,
        log_likelihood=log_likelihood,
        iterations=iteration,
        converged=converged,
        diagnostics={
            "responsibilities": responsibilities,
            "expected_tallies": expected_tallies,
        },
    )


def fit_mixture_samples(
    samples: ArrayLike,
    components: Sequence[MixtureComponentSpec],
    *,
    bins: int | Sequence[float] | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    min_weight: float = 1e-4,
    random_state: int | None = None,
) -> MixtureFitResult:
    """Wrapper that converts raw samples into grouped tallies before fitting."""

    data = np.asarray(samples, dtype=float)
    if data.size == 0:
        raise ValueError("Samples array must contain at least one observation.")

    if bins is None:
        bin_count = max(10, int(np.sqrt(data.size)))
        hist, edges = np.histogram(data, bins=bin_count)
    else:
        hist, edges = np.histogram(data, bins=bins)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    return fit_mixture_grouped(
        midpoints,
        hist,
        components,
        max_iter=max_iter,
        tol=tol,
        min_weight=min_weight,
        random_state=random_state,
    )


def _evaluate_pdf(distribution_name: str, x: np.ndarray, params: dict[str, float]) -> np.ndarray:
    distribution = get_distribution(distribution_name)
    values = distribution.pdf(x, _normalise_parameters(params))
    return np.clip(values, 1e-12, None)


def _initial_parameters(
    x: np.ndarray,
    y: np.ndarray,
    distribution_name: str,
    index: int,
    total_components: int,
    rng: np.random.Generator,
) -> tuple[dict[str, float], float]:
    indices = np.argsort(x)
    chunks = np.array_split(indices, total_components)
    subset_idx = chunks[index]
    if subset_idx.size > 0 and np.sum(y[subset_idx]) > 0:
        subset_x = x[subset_idx]
        subset_y = y[subset_idx]
        config = default_fit_config(distribution_name, subset_x, subset_y)
        init_weight = float(np.sum(subset_y))
    else:
        config = default_fit_config(distribution_name, x, y)
        init_weight = float(np.sum(y)) / float(total_components)
    initial = _normalise_parameters(dict(config.initial))
    jitter = 0.9 + 0.2 * rng.random(len(initial))
    for scale, key in zip(jitter, initial.keys(), strict=True):
        initial[key] = max(initial[key] * scale, 1e-6)
    return initial, init_weight


def _update_component(
    distribution_name: str,
    x: np.ndarray,
    tallies: np.ndarray,
    previous_params: dict[str, float],
) -> MixtureComponentFit:
    total = float(np.sum(tallies))
    name_lower = distribution_name.lower()

    if total <= 0:
        return MixtureComponentFit(
            name=distribution_name,
            weight=0.0,
            parameters=dict(previous_params),
            diagnostics={"note": "no weight assigned"},
        )

    if name_lower == "gamma":
        mean = float(np.sum(tallies * x) / total)
        variance = float(np.sum(tallies * (x - mean) ** 2) / total)
        variance = max(variance, 1e-6)
        shape = max((mean**2) / variance, 1e-6)
        scale = max(variance / mean, 1e-6)
        params = {"beta": scale, "p": shape}
        return MixtureComponentFit(
            name=distribution_name,
            weight=0.0,
            parameters=params,
            diagnostics={"method": "moments"},
        )

    inventory = InventorySpec(
        name=f"mixture-{distribution_name}",
        sampling="mixture",
        bins=x,
        tallies=tallies,
    )
    config = default_fit_config(distribution_name, x, np.clip(tallies, 1e-12, None))
    if "s" in config.initial:
        config.initial["s"] = 1.0
    fit = fit_inventory(inventory, [distribution_name], {distribution_name: config})[0]
    parameters = _normalise_parameters(fit.parameters)
    return MixtureComponentFit(
        name=distribution_name,
        weight=0.0,
        parameters=parameters,
        gof=fit.gof,
        diagnostics=fit.diagnostics,
    )


def _normalise_parameters(params: dict[str, float]) -> dict[str, float]:
    cleaned = dict(params)
    if "s" in cleaned:
        cleaned.pop("s")
    return cleaned


def mixture_pdf(
    x: ArrayLike,
    components: Sequence[MixtureComponentFit],
) -> np.ndarray:
    """Evaluate the PDF of a fitted mixture at points ``x``."""
    points = np.asarray(x, dtype=float)
    pdf_values = np.zeros_like(points, dtype=float)
    for component in components:
        dist = get_distribution(component.name)
        pdf_values += component.weight * dist.pdf(
            points, _normalise_parameters(component.parameters)
        )
    return pdf_values


def mixture_cdf(
    x: ArrayLike,
    components: Sequence[MixtureComponentFit],
) -> np.ndarray:
    """Evaluate the mixture CDF at points ``x``."""
    points = np.asarray(x, dtype=float)
    cdf_values = np.zeros_like(points, dtype=float)
    for component in components:
        dist = get_distribution(component.name)
        if dist.cdf is not None:
            cdf_values += component.weight * dist.cdf(
                points, _normalise_parameters(component.parameters)
            )
        else:
            cdf_values += component.weight * _numeric_cdf(points, dist.name, component.parameters)
    return np.clip(cdf_values, 0.0, 1.0)


def sample_mixture(
    size: int,
    components: Sequence[MixtureComponentFit],
    *,
    random_state: int | None = None,
) -> np.ndarray:
    """Draw random samples from a fitted mixture."""
    rng = np.random.default_rng(random_state)
    weights = np.array([component.weight for component in components], dtype=float)
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Component weights must sum to one.")
    if np.any(weights < 0):
        raise ValueError("Component weights must be non-negative.")
    choices = rng.choice(len(components), size=size, p=weights)
    samples = np.empty(size, dtype=float)
    for idx, component in enumerate(components):
        count = int(np.sum(choices == idx))
        if count == 0:
            continue
        dist = get_distribution(component.name)
        samples[choices == idx] = _sample_from_distribution(
            dist.name, count, component.parameters, rng
        )
    return samples


def _sample_from_distribution(
    name: str,
    size: int,
    params: Mapping[str, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample from a known distribution using numpy or scipy helpers."""
    name_lower = name.lower()
    if name_lower == "gamma":
        shape = params["p"]
        scale = params["beta"]
        return rng.gamma(shape=shape, scale=scale, size=size)
    if name_lower == "weibull":
        shape = params["a"]
        scale = params["beta"]
        return scale * np.power(-np.log(rng.random(size)), 1.0 / shape)
    message = (
        f"Sampling not implemented for distribution '{name}'. "
        "Consider overriding _sample_from_distribution."
    )
    raise NotImplementedError(message)


def _numeric_cdf(points: np.ndarray, name: str, params: Mapping[str, float]) -> np.ndarray:
    if name.lower() == "gamma":
        from scipy.stats import gamma as scipy_gamma

        shape = params["p"]
        scale = params["beta"]
        return scipy_gamma.cdf(points, shape, scale=scale)
    if name.lower() == "weibull":
        from scipy.stats import weibull_min

        shape = params["a"]
        scale = params["beta"]
        return weibull_min.cdf(points, shape, scale=scale)
    raise NotImplementedError(f"Numerical CDF not implemented for distribution '{name}'.")


__all__ = [
    "MixtureComponentSpec",
    "fit_mixture_grouped",
    "fit_mixture_samples",
    "mixture_pdf",
    "mixture_cdf",
    "sample_mixture",
]
