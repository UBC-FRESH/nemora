"""Distribution registry and canonical implementations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.special import gamma as gamma_fn

from .base import (
    Distribution,
    Pdf,
    clear_registry,
    get_distribution,
    list_distributions,
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
]


def _register_builtin() -> None:
    for dist in STANDARD_DISTRIBUTIONS + GENERALIZED_BETA_DISTRIBUTIONS:
        register_distribution(dist, overwrite=True)


_register_builtin()
