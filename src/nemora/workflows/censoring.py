"""Workflow for censored or truncated tallies using two-stage scaling."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from ..core import FitResult, InventorySpec
from ..distfit import FitConfig, default_fit_config, fit_inventory, fit_with_lmfit


def fit_censored_inventory(
    dbh_cm: np.ndarray,
    density: np.ndarray,
    *,
    support: tuple[float, float],
    distributions: Iterable[str] | None = None,
    configs: Mapping[str, FitConfig] | None = None,
) -> list[FitResult]:
    """Fit complete-form PDFs to censored tallies with a two-stage scaler."""
    dbh = np.asarray(dbh_cm, dtype=float)
    values = np.asarray(density, dtype=float)
    scale_guess = float(values.max() if values.size else 1.0)
    inventory = InventorySpec(
        name="censored-inventory",
        sampling="fixed-area",
        bins=dbh,
        tallies=values,
        metadata={"support": support},
    )
    chosen = tuple(distributions) if distributions is not None else ("weibull", "gamma")
    configs = dict(configs or {})
    for name in chosen:
        config = configs.get(name)
        if config is None:
            config = default_fit_config(name, dbh, values)
            configs[name] = config
        if "s" in config.initial:
            config.initial["s"] = config.initial.get("s", scale_guess) or scale_guess
        config.bounds = config.bounds or {"s": (1e-6, None)}
    return fit_inventory(inventory, chosen, configs, fitter=fit_with_lmfit)
