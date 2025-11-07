"""Horizontal point sampling fitting workflow."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np

from ..core import FitResult, InventorySpec
from ..distfit import FitConfig, default_fit_config, fit_inventory
from ..weighting import hps_compression_factor, hps_expansion_factor


def fit_hps_inventory(
    dbh_cm: np.ndarray,
    tally: np.ndarray,
    *,
    baf: float,
    distributions: Iterable[str] | None = None,
    configs: Mapping[str, FitConfig] | None = None,
    grouped_weibull_mode: str = "auto",
) -> list[FitResult]:
    """Fit HPS tallies using weighted stand-table expansion."""
    dbh = np.asarray(dbh_cm, dtype=float)
    tallies = np.asarray(tally, dtype=float)
    stand_table = tallies * hps_expansion_factor(dbh, baf=baf)
    weights = hps_compression_factor(dbh, baf=baf)
    mode = str(grouped_weibull_mode).lower()
    if mode not in {"auto", "ls", "mle"}:
        raise ValueError("grouped_weibull_mode must be one of 'auto', 'ls', or 'mle'.")
    inventory = InventorySpec(
        name="hps-inventory",
        sampling="hps",
        bins=dbh,
        tallies=stand_table,
        metadata={
            "baf": baf,
            "original_tally": tallies,
            "grouped": True,
            "grouped_weibull_mode": mode,
        },
    )
    chosen = tuple(distributions) if distributions is not None else ("weibull", "gamma")
    configs = dict(configs or {})
    for name in chosen:
        config = configs.get(name)
        if config is None:
            config = default_fit_config(name, dbh, stand_table)
            configs[name] = config
        config.initial.setdefault("s", float(np.max(stand_table)) if stand_table.size else 1.0)
        config.weights = weights
    return fit_inventory(inventory, chosen, configs)
