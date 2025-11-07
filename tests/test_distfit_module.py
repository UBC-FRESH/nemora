import numpy as np
import pytest

import nemora
from nemora.core import InventorySpec
from nemora.distfit import default_fit_config, fit_inventory


def test_package_exports_distfit_module() -> None:
    assert hasattr(nemora, "distfit"), "nemora.distfit should be available from the package root"
    assert hasattr(nemora.distfit, "default_fit_config")


def test_default_fit_config_weibull_seed_values() -> None:
    x = np.linspace(5.0, 40.0, num=8)
    y = np.array([5, 12, 18, 24, 20, 12, 6, 3], dtype=float)
    config = default_fit_config("weibull", x, y)
    assert config.distribution == "weibull"
    assert config.initial["a"] > 0
    assert config.initial["beta"] > 0
    assert config.initial["s"] > 0
    assert config.bounds is not None
    assert config.bounds["a"][0] == pytest.approx(1e-6)


def test_fit_inventory_returns_fit_results() -> None:
    x = np.linspace(5.0, 45.0, num=10)
    y = np.array([5, 10, 18, 22, 20, 15, 10, 6, 3, 2], dtype=float)
    inventory = InventorySpec(
        name="stand-table",
        sampling="fixed",
        bins=x,
        tallies=y,
    )
    config = default_fit_config("gamma", x, y)
    results = fit_inventory(inventory, distributions=("gamma",), configs={"gamma": config})
    assert len(results) == 1
    fit = results[0]
    assert fit.distribution == "gamma"
    assert fit.parameters["beta"] > 0
    assert fit.parameters["p"] > 0
    assert fit.gof["rss"] >= 0
    assert fit.diagnostics.get("method") == "curve-fit"
