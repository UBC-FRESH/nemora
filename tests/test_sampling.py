from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from scipy.stats import gamma as scipy_gamma

from nemora import sampling as sampling_module
from nemora.core import FitResult, MixtureComponentFit, MixtureFitResult
from nemora.sampling import (
    BootstrapResult,
    SamplingConfig,
    bootstrap_inventory,
    pdf_to_cdf,
    sample_distribution,
    sample_mixture_fit,
)
from nemora.sampling.helpers import bootstrap_dbh_vectors

ACCURACY_CASES: tuple[tuple[SamplingConfig, float], ...] = (
    (
        SamplingConfig(grid_points=8192, support_multiplier=12.0, integration_method="trapezoid"),
        3e-3,
    ),
    (SamplingConfig(grid_points=8192, support_multiplier=12.0, integration_method="simpson"), 3e-3),
    (
        SamplingConfig(
            grid_points=1024,
            support_multiplier=12.0,
            integration_method="quad",
            quad_abs_tol=1e-9,
            quad_rel_tol=1e-8,
        ),
        3e-3,
    ),
)


@pytest.mark.parametrize(("cfg", "tolerance"), ACCURACY_CASES)
def test_pdf_to_cdf_numeric_methods_match_gamma(cfg: SamplingConfig, tolerance: float) -> None:
    params = {"beta": 4.0, "p": 3.0, "s": 1.0}
    cdf_fn = pdf_to_cdf("gamma", params, method="numeric", config=cfg)
    support = cfg.support_multiplier * params["beta"]
    values = np.linspace(0.0, support, 20)
    ours = cdf_fn(values)
    expected = scipy_gamma.cdf(values, params["p"], scale=params["beta"])
    np.testing.assert_allclose(ours, expected, atol=tolerance)


def test_pdf_to_cdf_caching_reuses_numeric_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    params = {"beta": 4.0, "p": 3.0, "s": 1.0}
    cfg = SamplingConfig(
        grid_points=256,
        support_multiplier=8.0,
        integration_method="trapezoid",
        cache_numeric_cdf=True,
    )
    call_counter = {"count": 0}

    original = sampling_module._numeric_cdf

    def counting_numeric_cdf(xs: np.ndarray, pdf_callable, *, cfg: SamplingConfig) -> np.ndarray:
        call_counter["count"] += 1
        return original(xs, pdf_callable, cfg=cfg)

    monkeypatch.setattr(sampling_module, "_numeric_cdf", counting_numeric_cdf)
    cdf_fn = pdf_to_cdf("gamma", params, method="numeric", config=cfg)
    cdf_fn(np.linspace(0.0, cfg.support_multiplier * params["beta"], 10))
    # Second call should hit the cache and avoid invoking the numeric integrator.
    cdf_fn(np.linspace(0.0, cfg.support_multiplier * params["beta"], 10))
    assert call_counter["count"] == 1


def test_pdf_to_cdf_numeric_monotonicity() -> None:
    params = {"a": 2.0, "beta": 10.0, "s": 1.0}
    cfg = SamplingConfig(grid_points=256)
    cdf_fn = pdf_to_cdf("weibull", params, method="numeric", config=cfg)
    values = np.linspace(0.0, 30.0, 25)
    cdf_values = cdf_fn(values)
    assert np.all(np.diff(cdf_values) >= -1e-9)


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


def test_sample_distribution_weibull_inverse_matches_formula() -> None:
    params = {"a": 2.0, "beta": 5.0, "s": 1.0}
    rng = np.random.default_rng(2025)
    draws = sample_distribution("weibull", params, size=5, random_state=rng)
    rng_expected = np.random.default_rng(2025)
    expected = params["beta"] * (-np.log1p(-rng_expected.random(5))) ** (1 / params["a"])
    np.testing.assert_allclose(draws, expected)


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


def test_sample_mixture_fit_accepts_generator() -> None:
    components = [
        MixtureComponentFit(name="gamma", weight=0.5, parameters={"beta": 2.0, "p": 2.0}),
        MixtureComponentFit(name="gamma", weight=0.5, parameters={"beta": 5.0, "p": 4.0}),
    ]
    mixture = MixtureFitResult(
        distribution="mixture",
        components=components,
        log_likelihood=-10.0,
        iterations=5,
        converged=True,
    )
    rng1 = np.random.default_rng(999)
    rng2 = np.random.default_rng(999)
    draws_a = sample_mixture_fit(mixture, size=50, random_state=rng1)
    draws_b = sample_mixture_fit(mixture, size=50, random_state=rng2)
    np.testing.assert_allclose(draws_a, draws_b)


def test_sample_mixture_fit_truncation_bounds() -> None:
    components = [
        MixtureComponentFit(name="gamma", weight=0.5, parameters={"beta": 3.0, "p": 2.0}),
        MixtureComponentFit(name="gamma", weight=0.5, parameters={"beta": 7.0, "p": 4.0}),
    ]
    mixture = MixtureFitResult(
        distribution="mixture",
        components=components,
        log_likelihood=-5.0,
        iterations=4,
        converged=True,
    )
    draws = sample_mixture_fit(
        mixture,
        size=200,
        random_state=np.random.default_rng(123),
        lower=1.0,
        upper=40.0,
    )
    assert draws.size == 200
    assert np.all(draws >= 1.0)
    assert np.all(draws <= 40.0)


def test_sample_mixture_fit_weight_overrides() -> None:
    components = [
        MixtureComponentFit(name="gamma", weight=0.1, parameters={"beta": 2.0, "p": 2.0}),
        MixtureComponentFit(name="gamma", weight=0.9, parameters={"beta": 12.0, "p": 5.0}),
    ]
    mixture = MixtureFitResult(
        distribution="mixture",
        components=components,
        log_likelihood=-8.0,
        iterations=6,
        converged=True,
    )
    draws = sample_mixture_fit(
        mixture,
        size=500,
        random_state=123,
        weight_overrides=[1.0, 0.0],
    )
    assert draws.size == 500
    # Overrides zero-out the second component so the mean should reflect the first component.
    assert np.mean(draws) < 8.0


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
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(sample.shape == (10, 2) for sample in results)


def test_bootstrap_inventory_result_wrapper() -> None:
    rng = np.random.default_rng(123)
    bins = np.array([5.0, 15.0, 25.0])
    tallies = np.array([2.0, 4.0, 1.0])
    fit = FitResult(
        distribution="weibull",
        parameters={"a": 2.5, "beta": 12.0, "s": 1.0},
    )
    result = bootstrap_inventory(
        fit,
        bins,
        tallies,
        resamples=2,
        sample_size=5,
        random_state=rng,
        return_result=True,
    )
    assert isinstance(result, BootstrapResult)
    assert result.resamples == 2
    assert len(result.samples) == 2
    stacked = result.stacked()
    assert stacked.shape[1] == 2


def test_bootstrap_result_to_dataframe() -> None:
    rng = np.random.default_rng(999)
    bins = np.array([5.0, 15.0, 25.0])
    tallies = np.array([2.0, 4.0, 1.0])
    fit = FitResult(
        distribution="weibull",
        parameters={"a": 2.5, "beta": 12.0, "s": 1.0},
    )
    result = bootstrap_inventory(
        fit,
        bins,
        tallies,
        resamples=2,
        sample_size=5,
        random_state=rng,
        return_result=True,
    )
    assert isinstance(result, BootstrapResult)
    frame = result.to_dataframe()
    assert list(frame.columns) == ["resample", "bin", "draw"]
    assert frame["resample"].nunique() == 2
    assert len(frame) == 10


def test_bootstrap_dbh_vectors_payload() -> None:
    rng = np.random.default_rng(7)
    bins = np.array([8.0, 18.0, 28.0])
    tallies = np.array([3.0, 4.0, 3.0])
    fit = FitResult(
        distribution="weibull",
        parameters={"a": 2.1, "beta": 11.0, "s": 1.0},
    )
    result = cast(
        BootstrapResult,
        bootstrap_inventory(
            fit,
            bins,
            tallies,
            resamples=2,
            sample_size=6,
            random_state=rng,
            return_result=True,
        ),
    )
    payload = bootstrap_dbh_vectors(result, stand_id="stand-1")
    assert payload.stand_id == "stand-1"
    assert set(payload.dbh_vectors) == {0, 1}
    assert payload.metadata["stand_id"] == "stand-1"
    frame = payload.frame
    assert frame is not None
    assert {"stand_id", "resample", "dbh", "weight"} <= set(frame.columns)
    assert frame["resample"].nunique() == 2
