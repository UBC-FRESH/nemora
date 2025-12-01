"""Distribution registry and canonical implementations."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.special import ndtri
from scipy.stats import fatiguelife, johnsonsb, norm
from scipy.stats import gamma as gamma_dist

from .base import (
    Distribution,
    Pdf,
    clear_registry,
    get_distribution,
    list_distributions,
    load_entry_points,
    load_yaml_config,
    register_distribution,
)
from .generalized_beta import GENERALIZED_BETA_DISTRIBUTIONS

__all__ = [
    "Distribution",
    "Pdf",
    "get_distribution",
    "list_distributions",
    "register_distribution",
    "clear_registry",
    "GENERALIZED_BETA_DISTRIBUTIONS",
    "GENERALIZED_SECANT_DISTRIBUTIONS",
    "default_parameter_bounds",
    "list_registry_metadata",
]


def generalized_gamma_pdf(x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Generalized gamma with optional scaling constant."""
    arr = np.asarray(x, dtype=float)
    a = params["a"]
    beta = params["beta"]
    p = params["p"]
    scale = params.get("s", 1.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        y = (
            scale
            * (a * np.power(arr, a * p - 1.0) * np.exp(-np.power(arr / beta, a)))
            / (np.power(beta, a * p) * gamma_fn(p))
        )
    return np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)


def weibull_pdf(x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    return generalized_gamma_pdf(
        x,
        {"a": params["a"], "beta": params["beta"], "p": 1.0, "s": params.get("s", 1.0)},
    )


def gamma_pdf(x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    return generalized_gamma_pdf(
        x,
        {"a": 1.0, "beta": params["beta"], "p": params["p"], "s": params.get("s", 1.0)},
    )


def _clip_unit(u: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(u, dtype=float)
    return np.clip(arr, 1e-12, 1 - 1e-12)


def weibull_cdf(x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, 0.0, None)
    a = params["a"]
    beta = params["beta"]
    with np.errstate(over="ignore"):
        values = 1.0 - np.exp(-np.power(arr / beta, a))
    return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)


def weibull_inverse_cdf(u: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    arr = _clip_unit(u)
    a = params["a"]
    beta = params["beta"]
    with np.errstate(divide="ignore"):
        return beta * np.power(-np.log1p(-arr), 1.0 / a)


def exponential_cdf(x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, 0.0, None)
    beta = params["beta"]
    values = 1.0 - np.exp(-arr / beta)
    return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)


def exponential_inverse_cdf(u: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    arr = _clip_unit(u)
    beta = params["beta"]
    with np.errstate(divide="ignore"):
        return -beta * np.log1p(-arr)


def uniform_cdf(x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    b = params["b"]
    values = arr / b
    values = np.clip(values, 0.0, 1.0)
    return values


def uniform_inverse_cdf(u: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    arr = _clip_unit(u)
    b = params["b"]
    return b * arr


def pareto_cdf(x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, params["b"], None)
    b_val = params["b"]
    p_val = params["p"]
    values = 1.0 - np.power(b_val / arr, p_val)
    return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0)


def pareto_inverse_cdf(u: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    arr = _clip_unit(u)
    b_val = params["b"]
    p_val = params["p"]
    return b_val * np.power(1.0 - arr, -1.0 / p_val)


def lognormal_cdf(x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    arr = np.clip(arr, 1e-12, None)
    mu = params["mu"]
    sigma = float(np.sqrt(max(params["sigma2"], 1e-12)))
    return norm.cdf((np.log(arr) - mu) / sigma)


def lognormal_inverse_cdf(u: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    arr = _clip_unit(u)
    mu = params["mu"]
    sigma = float(np.sqrt(max(params["sigma2"], 1e-12)))
    return np.exp(mu + sigma * ndtri(arr))


STANDARD_DISTRIBUTIONS = [
    Distribution(
        name="weibull",
        parameters=("a", "beta", "s"),
        pdf=weibull_pdf,
        cdf=weibull_cdf,
        inverse_cdf=weibull_inverse_cdf,
        notes="Complete-form Weibull via generalized gamma representation.",
    ),
    Distribution(
        name="gamma",
        parameters=("beta", "p", "s"),
        pdf=gamma_pdf,
        notes="Gamma distribution with optional scaling factor.",
    ),
    Distribution(
        name="johnsonsb",
        parameters=("a", "b", "loc", "scale"),
        pdf=lambda x, params: np.nan_to_num(
            johnsonsb.pdf(
                np.asarray(x, dtype=float),
                a=params["a"],
                b=params["b"],
                loc=params.get("loc", 0.0),
                scale=params.get("scale", 1.0),
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        cdf=lambda x, params: np.nan_to_num(
            johnsonsb.cdf(
                np.asarray(x, dtype=float),
                a=params["a"],
                b=params["b"],
                loc=params.get("loc", 0.0),
                scale=params.get("scale", 1.0),
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        bounds={
            "a": (1e-6, None),
            "b": (1e-6, None),
            "scale": (1e-6, None),
        },
        notes="Johnson SB distribution with bounded support.",
    ),
    Distribution(
        name="birnbaum_saunders",
        parameters=("alpha", "beta"),
        pdf=lambda x, params: np.nan_to_num(
            fatiguelife.pdf(
                np.asarray(x, dtype=float),
                c=params["alpha"],
                scale=params["beta"],
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        cdf=lambda x, params: np.nan_to_num(
            fatiguelife.cdf(
                np.asarray(x, dtype=float),
                c=params["alpha"],
                scale=params["beta"],
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ),
        bounds={
            "alpha": (1e-6, None),
            "beta": (1e-6, None),
        },
        notes="Birnbaum-Saunders (fatigue life) distribution.",
    ),
]


def _build_generalized_secant_distributions() -> list[Distribution]:
    def _weights(params: Mapping[str, float], components: int) -> np.ndarray:
        raw = [float(params.get(f"omega{i}", 1.0 / components)) for i in range(1, components)]
        tail = 1.0 - float(np.sum(raw))
        weights = np.array(raw + [tail], dtype=float)
        weights = np.clip(weights, 1e-12, None)
        total = float(np.sum(weights))
        if total <= 0:
            return np.full(components, 1.0 / components, dtype=float)
        return weights / total

    def _pdf_factory(components: int):
        def pdf(x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
            arr = np.asarray(x, dtype=float)
            beta = max(float(params["beta"]), 1e-8)
            scale = 1.0 / beta
            weights = _weights(params, components)
            out = np.zeros_like(arr)
            for idx, weight in enumerate(weights, start=1):
                if weight <= 0:
                    continue
                out += weight * gamma_dist.pdf(arr, a=idx, scale=scale)
            return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        return pdf

    def _cdf_factory(components: int):
        def cdf(x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
            arr = np.asarray(x, dtype=float)
            beta = max(float(params["beta"]), 1e-8)
            scale = 1.0 / beta
            weights = _weights(params, components)
            out = np.zeros_like(arr)
            for idx, weight in enumerate(weights, start=1):
                if weight <= 0:
                    continue
                out += weight * gamma_dist.cdf(arr, a=idx, scale=scale)
            return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        return cdf

    distributions: list[Distribution] = []
    for components in range(2, 7):
        param_names = ("beta",) + tuple(f"omega{i}" for i in range(1, components))
        bounds: dict[str, tuple[float | None, float | None]] = {"beta": (1e-6, None)}
        for i in range(1, components):
            bounds[f"omega{i}"] = (1e-6, 1.0)
        distributions.append(
            Distribution(
                name=f"gsm{components}",
                parameters=param_names,
                pdf=_pdf_factory(components),
                cdf=_cdf_factory(components),
                bounds=bounds,
                notes=f"Generalised secant mixture with {components} gamma components.",
            )
        )
    return distributions


GENERALIZED_SECANT_DISTRIBUTIONS = _build_generalized_secant_distributions()

STANDARD_DISTRIBUTIONS.extend(GENERALIZED_SECANT_DISTRIBUTIONS)


_LOWER_BOUNDED_PARAMS: dict[str, tuple[float | None, float | None]] = {
    "a": (1e-6, None),
    "b": (1e-6, None),
    "beta": (1e-6, None),
    "p": (1e-6, None),
    "q": (1e-6, None),
    "sigma2": (1e-6, None),
    "d": (1e-6, None),
    "u": (1e-6, None),
    "v": (1e-6, None),
    "df": (1e-6, None),
    "s": (1e-6, None),
    "alpha": (1e-6, None),
    "scale": (1e-6, None),
    "beta1": (1e-6, None),
}


def default_parameter_bounds(
    parameters: Iterable[str],
) -> dict[str, tuple[float | None, float | None]]:
    """Return heuristic bounds for standard Nemora distribution parameters."""

    bounds: dict[str, tuple[float | None, float | None]] = {}
    for name in parameters:
        if name.startswith("omega"):
            bounds[name] = (1e-6, 1.0)
            continue
        param_bounds = _LOWER_BOUNDED_PARAMS.get(name)
        if param_bounds:
            bounds[name] = param_bounds
    return bounds


def _apply_inverse_metadata(dist: Distribution) -> None:
    """Attach analytic CDF/inverse metadata where closed forms exist."""

    name = dist.name.lower()
    if name in {"weibull", "w"}:
        dist.cdf = weibull_cdf
        dist.inverse_cdf = weibull_inverse_cdf
    elif name == "exp":
        dist.cdf = exponential_cdf
        dist.inverse_cdf = exponential_inverse_cdf
    elif name == "u":
        dist.cdf = uniform_cdf
        dist.inverse_cdf = uniform_inverse_cdf
    elif name == "pareto":
        dist.cdf = pareto_cdf
        dist.inverse_cdf = pareto_inverse_cdf
    elif name == "ln":
        dist.cdf = lognormal_cdf
        dist.inverse_cdf = lognormal_inverse_cdf


def _register_builtin() -> None:
    for dist in STANDARD_DISTRIBUTIONS + GENERALIZED_BETA_DISTRIBUTIONS:
        _apply_inverse_metadata(dist)
        register_distribution(dist, overwrite=True)


def _load_config_files() -> None:
    project_root = Path(__file__).resolve().parents[3]
    config_dir = project_root / "config" / "distributions"
    if config_dir.exists():
        for path in sorted(config_dir.glob("*.yaml")):
            load_yaml_config(path)

    env_paths = os.environ.get("DBHDISTFIT_DISTRIBUTIONS")
    if env_paths:
        for item in env_paths.split(os.pathsep):
            load_yaml_config(item)


_register_builtin()
load_entry_points()
_load_config_files()


def list_registry_metadata(
    *,
    names: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    """Return structured metadata for registered distributions.

    Parameters
    ----------
    names:
        Optional iterable of distribution names to filter the registry results. When omitted,
        all distributions are returned. Matching is case-insensitive.
    """

    if names is None:
        requested = set(list_distributions())
    else:
        requested = {name.lower() for name in names}
    metadata: list[dict[str, object]] = []
    for name in list_distributions():
        if requested and name.lower() not in requested:
            continue
        dist = get_distribution(name)
        entry_bounds = default_parameter_bounds(dist.parameters)
        if dist.bounds:
            entry_bounds.update(dist.bounds)
        metadata.append(
            {
                "name": dist.name,
                "parameters": dist.parameters,
                "bounds": entry_bounds,
                "notes": dist.notes,
                "extras": dist.extras,
            }
        )
    return metadata
