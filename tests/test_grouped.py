import numpy as np
import pytest
from scipy.stats import fatiguelife

from nemora.core import InventorySpec
from nemora.distfit import fit_inventory
from nemora.workflows.hps import fit_hps_inventory


def test_grouped_weibull_estimator_applied() -> None:
    bins = np.linspace(5, 45, num=10)
    counts = np.array([5, 12, 18, 25, 20, 15, 10, 6, 4, 2], dtype=float)
    result = fit_hps_inventory(bins, counts, baf=1.0, distributions=("weibull",))
    assert result
    fit = result[0]
    assert fit.diagnostics.get("method") in {"grouped-ls", "grouped-mle"}
    assert fit.parameters["a"] > 0
    assert fit.parameters["beta"] > 0
    assert fit.parameters["s"] > 0
    assert "ks" in fit.gof
    assert "cvm" in fit.gof


def test_grouped_johnsonsb_estimator_applied() -> None:
    bins = np.linspace(5, 40, num=12)
    counts = np.array([4, 8, 10, 18, 22, 20, 15, 9, 6, 4, 3, 2], dtype=float)
    result = fit_hps_inventory(bins, counts, baf=1.0, distributions=("johnsonsb",))
    assert result
    fit = result[0]
    assert fit.diagnostics.get("method") == "grouped-em"
    assert "method_detail" in fit.diagnostics
    assert "iterations" in fit.diagnostics
    assert fit.parameters["a"] > 0
    assert fit.parameters["b"] > 0
    assert fit.parameters["scale"] > 0


def test_grouped_birnbaum_saunders_estimator_applied() -> None:
    bins = np.linspace(10, 60, num=10)
    counts = np.array([2, 6, 12, 18, 25, 20, 14, 8, 4, 2], dtype=float)
    result = fit_hps_inventory(
        bins,
        counts,
        baf=1.0,
        distributions=("birnbaum_saunders",),
    )
    assert result
    fit = result[0]
    assert fit.diagnostics.get("method") in {"grouped-em", "grouped-mle"}
    assert "method_detail" in fit.diagnostics
    assert fit.parameters["alpha"] > 0
    assert fit.parameters["beta"] > 0


def test_grouped_birnbaum_saunders_em_on_synthetic_counts() -> None:
    edges = np.linspace(5, 70, num=16)
    bins = 0.5 * (edges[:-1] + edges[1:])
    alpha_true = 1.1
    beta_true = 30.0
    probabilities = fatiguelife.cdf(edges[1:], c=alpha_true, scale=beta_true) - fatiguelife.cdf(
        edges[:-1], c=alpha_true, scale=beta_true
    )
    counts = np.clip(np.round(probabilities * 2000.0), 1.0, None)
    inventory = InventorySpec(
        name="synthetic-birnbaum",
        sampling="grouped",
        bins=bins,
        tallies=counts,
        metadata={"grouped": True},
    )
    results = fit_inventory(inventory, distributions=("birnbaum_saunders",), configs={})
    fit = results[0]
    assert fit.diagnostics.get("method") == "grouped-em"
    assert fit.parameters["alpha"] > 0
    assert fit.parameters["beta"] > 0


@pytest.mark.parametrize("distribution", ["gsm3", "gsm6"])
def test_grouped_gsm_estimator_applied(distribution: str) -> None:
    bins = np.linspace(5, 35, num=8)
    counts = np.array([8, 14, 20, 24, 18, 12, 6, 3], dtype=float)
    result = fit_hps_inventory(bins, counts, baf=1.0, distributions=(distribution,))
    assert result
    fit = result[0]
    assert fit.distribution == distribution
    assert fit.diagnostics.get("method") == "grouped-mle"
    assert fit.parameters["beta"] > 0
    assert "omega1" in fit.parameters
    weights = fit.diagnostics.get("component_weights")
    assert weights is not None
    assert float(weights[-1]) > 0
    assert np.isclose(float(np.sum(weights)), 1.0, atol=1e-6)
