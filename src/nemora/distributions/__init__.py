"""Distribution registry and canonical implementations."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.special import gamma as gamma_fn
from scipy.stats import fatiguelife, johnsonsb
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


STANDARD_DISTRIBUTIONS = [
    Distribution(
        name="weibull",
        parameters=("a", "beta", "s"),
        pdf=weibull_pdf,
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


def _register_builtin() -> None:
    for dist in STANDARD_DISTRIBUTIONS + GENERALIZED_BETA_DISTRIBUTIONS:
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
