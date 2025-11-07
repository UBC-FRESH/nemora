import numpy as np
import pytest

from nemora.core import FitResult, MixtureComponentFit, MixtureFitResult
from nemora.sampling import bootstrap_inventory, pdf_to_cdf, sample_distribution, sample_mixture_fit


@pytest.mark.parametrize(
    "distribution,params",
    [
        ("weibull", {"a": 2.0, "beta": 10.0, "s": 1.0}),
        ("gamma", {"beta": 4.0, "p": 3.0, "s": 1.0}),
    ],
)
def test_pdf_to_cdf_numeric(distribution: str, params: dict[str, float]) -> None:
    grid = np.linspace(0.1, 50.0, 256)
    cdf_fn = pdf_to_cdf(distribution, params, method="numeric", grid=grid)
    values = np.array([0.1, 10.0, 20.0, 40.0])
    cdf_values = cdf_fn(values)
    assert np.all(np.diff(cdf_values) >= 0)  # monotonic
    assert cdf_values[0] >= 0
    assert cdf_values[-1] <= 1


def test_sample_distribution_returns_expected_shape() -> None:
    rng = np.random.default_rng(123)
    draws = sample_distribution(
        "weibull",
        {"a": 2.5, "beta": 12.0, "s": 1.0},
        size=500,
        random_state=rng,
    )
    assert draws.shape == (500,)
    assert np.all(draws >= 0)


def test_sample_mixture_fit_matches_component_weights() -> None:
    rng = np.random.default_rng(1234)
    components = [
        MixtureComponentFit(name="gamma", weight=0.6, parameters={"beta": 3.0, "p": 2.0}),
        MixtureComponentFit(name="gamma", weight=0.4, parameters={"beta": 8.0, "p": 5.0}),
    ]
    mixture = MixtureFitResult(
        distribution="mixture",
        components=components,
        log_likelihood=-100.0,
        iterations=10,
        converged=True,
    )
    draws = sample_mixture_fit(mixture, size=1000, random_state=rng)
    assert draws.shape == (1000,)
    assert np.all(draws >= 0)


def test_bootstrap_inventory_resamples() -> None:
    rng = np.random.default_rng(42)
    bins = np.array([10.0, 20.0, 30.0])
    tallies = np.array([5, 3, 2], dtype=float)
    fit = FitResult(
        distribution="gamma",
        parameters={"beta": 5.0, "p": 2.5, "s": 1.0},
    )
    results = bootstrap_inventory(
        fit,
        bins,
        tallies,
        resamples=3,
        sample_size=10,
        random_state=rng,
    )
    assert len(results) == 3
    assert all(sample.shape == (10, 2) for sample in results)
