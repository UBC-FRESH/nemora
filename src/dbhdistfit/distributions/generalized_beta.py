"""Generalized beta family distribution implementations."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.special import beta as beta_fn
from scipy.special import gamma as gamma_fn

from .base import Distribution

EPS = 1e-12


def _as_array(x: np.ndarray | float) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _clean(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _gb1_core(arr: np.ndarray, a: float, b: float, p: float, q: float, s: float) -> np.ndarray:
    if a == 0:
        raise ValueError("Parameter 'a' must be non-zero for GB1.")
    if a > 0:
        mask = (arr > 0) & (arr < b)
    else:
        mask = arr > b
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        inner = 1.0 - np.power(arr / b, a)
        inner = np.clip(inner, 0.0, None)
        base = (
            s
            * (np.abs(a) * np.power(arr, a * p - 1.0) * np.power(inner, q - 1.0))
            / (np.power(b, a * p) * beta_fn(p, q))
        )
    out = np.zeros_like(arr)
    out[mask] = base[mask]
    return _clean(out)


def _gb2_core(arr: np.ndarray, a: float, b: float, p: float, q: float, s: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        denom = np.power(1.0 + np.power(arr / b, a), p + q)
        base = (
            s
            * (np.abs(a) * np.power(arr, a * p - 1.0))
            / (np.power(b, a * p) * beta_fn(p, q) * denom)
        )
    mask = arr > 0
    out = np.zeros_like(arr)
    out[mask] = base[mask]
    return _clean(out)


def _gg_core(arr: np.ndarray, a: float, beta: float, p: float, s: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        base = (
            s
            * (a * np.power(arr, a * p - 1.0) * np.exp(-np.power(arr / beta, a)))
            / (np.power(beta, a * p) * gamma_fn(p))
        )
    mask = arr > 0
    out = np.zeros_like(arr)
    out[mask] = base[mask]
    return _clean(out)


def gb1_pdf(
    x: np.ndarray | float, *, a: float, b: float, p: float, q: float, s: float = 1.0
) -> np.ndarray:
    arr = _as_array(x)
    return _gb1_core(arr, a, b, p, q, s)


def gb2_pdf(
    x: np.ndarray | float, *, a: float, b: float, p: float, q: float, s: float = 1.0
) -> np.ndarray:
    arr = _as_array(x)
    return _gb2_core(arr, a, b, p, q, s)


def gg_pdf(x: np.ndarray | float, *, a: float, beta: float, p: float, s: float = 1.0) -> np.ndarray:
    arr = _as_array(x)
    return _gg_core(arr, a, beta, p, s)


def _wrap(func, param_order: tuple[str, ...]):
    def wrapper(x: np.ndarray | float, params: Mapping[str, float]) -> np.ndarray:
        kwargs = {name: params[name] for name in param_order}
        return func(x, **kwargs)

    return wrapper


def _distribution(name: str, parameters: tuple[str, ...], func) -> Distribution:
    return Distribution(
        name=name,
        parameters=parameters,
        pdf=_wrap(func, parameters),
        notes="Generalized beta family distribution.",
    )


def _build_generalized_beta_distributions() -> list[Distribution]:
    lim0 = 1e-10

    def gb1_func(x, a, b, p, q, s=1.0):
        return gb1_pdf(x, a=a, b=b, p=p, q=q, s=s)

    def gb2_func(x, a, b, p, q, s=1.0):
        return gb2_pdf(x, a=a, b=b, p=p, q=q, s=s)

    def gg_func(x, a, beta, p, s=1.0):
        return gg_pdf(x, a=a, beta=beta, p=p, s=s)

    distributions = [
        _distribution("gb1", ("a", "b", "p", "q", "s"), gb1_func),
        _distribution("gb2", ("a", "b", "p", "q", "s"), gb2_func),
        _distribution("gg", ("a", "beta", "p", "s"), gg_func),
    ]

    def add(name: str, params: tuple[str, ...], func) -> None:
        distributions.append(_distribution(name, params, func))

    add("ib1", ("b", "p", "q", "s"), lambda x, b, p, q, s: gb1_pdf(x, a=-1.0, b=b, p=p, q=q, s=s))
    add(
        "ug",
        ("b", "d", "q", "s"),
        lambda x, b, d, q, s: gb1_pdf(x, a=lim0, b=b, p=d / lim0, q=q, s=s),
    )
    add("b1", ("b", "p", "q", "s"), lambda x, b, p, q, s: gb1_pdf(x, a=1.0, b=b, p=p, q=q, s=s))
    add("b2", ("b", "p", "q", "s"), lambda x, b, p, q, s: gb2_pdf(x, a=1.0, b=b, p=p, q=q, s=s))
    add("sm", ("a", "b", "q", "s"), lambda x, a, b, q, s: gb2_pdf(x, a=a, b=b, p=1.0, q=q, s=s))
    add("dagum", ("a", "b", "p", "s"), lambda x, a, b, p, s: gb2_pdf(x, a=a, b=b, p=p, q=1.0, s=s))
    add("pareto", ("b", "p", "s"), lambda x, b, p, s: gb1_pdf(x, a=-1.0, b=b, p=p, q=1.0, s=s))
    add("p", ("b", "p", "s"), lambda x, b, p, s: gb1_pdf(x, a=1.0, b=b, p=p, q=1.0, s=s))
    add(
        "ln",
        ("mu", "sigma2", "s"),
        lambda x, mu, sigma2, s: gg_pdf(
            x,
            a=lim0,
            beta=np.power(sigma2 * lim0**2, 1.0 / lim0),
            p=(lim0 * mu + 1.0) / (sigma2 * lim0**2),
            s=s,
        ),
    )
    add("ga", ("beta", "p", "s"), lambda x, beta, p, s: gg_pdf(x, a=1.0, beta=beta, p=p, s=s))
    add("w", ("a", "beta", "s"), lambda x, a, beta, s: gg_pdf(x, a=a, beta=beta, p=1.0, s=s))
    add(
        "f",
        ("u", "v", "s"),
        lambda x, u, v, s: gb2_pdf(x, a=1.0, b=v / u, p=u / 2.0, q=v / 2.0, s=s),
    )
    add("l", ("b", "q", "s"), lambda x, b, q, s: gb2_pdf(x, a=1.0, b=b, p=1.0, q=q, s=s))
    add("il", ("b", "p", "s"), lambda x, b, p, s: gb2_pdf(x, a=1.0, b=b, p=p, q=1.0, s=s))
    add("fisk", ("a", "b", "s"), lambda x, a, b, s: gb2_pdf(x, a=a, b=b, p=1.0, q=1.0, s=s))
    add("u", ("b", "s"), lambda x, b, s: gb1_pdf(x, a=1.0, b=b, p=1.0, q=1.0, s=s))
    add("halfn", ("sigma2", "s"), lambda x, sigma2, s: gg_pdf(x, a=2.0, beta=sigma2, p=0.5, s=s))
    add("chisq", ("p", "s"), lambda x, p, s: gg_pdf(x, a=1.0, beta=2.0, p=p, s=s))
    add("exp", ("beta", "s"), lambda x, beta, s: gg_pdf(x, a=1.0, beta=beta, p=1.0, s=s))
    add("r", ("beta", "s"), lambda x, beta, s: gg_pdf(x, a=2.0, beta=beta, p=1.0, s=s))
    add(
        "halft",
        ("df", "s"),
        lambda x, df, s: gb2_pdf(x, a=2.0, b=np.sqrt(df), p=0.5, q=df / 2.0, s=s),
    )
    add("ll", ("b", "s"), lambda x, b, s: gb2_pdf(x, a=1.0, b=b, p=1.0, q=1.0, s=s))

    return distributions


GENERALIZED_BETA_DISTRIBUTIONS = _build_generalized_beta_distributions()

__all__ = [
    "GENERALIZED_BETA_DISTRIBUTIONS",
    "gb1_pdf",
    "gb2_pdf",
    "gg_pdf",
]
