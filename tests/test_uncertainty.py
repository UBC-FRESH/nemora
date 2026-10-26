"""Tests for multiplicative curve uncertainty bands."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemora.uncertainty import (
    CurveBand,
    apply_ratio_band,
    band_coverage,
    fit_ratio_band,
    read_validation_csv,
    write_band_csv,
)


def _synthetic(seed: int = 42, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    predicted = np.full(n, 12.0)
    ratios = 1.4 * rng.weibull(3.0, n)
    return predicted * ratios, predicted


def test_fit_ratio_band_returns_a_band() -> None:
    observed, predicted = _synthetic()

    band = fit_ratio_band(observed, predicted, distribution="weibull", bins=25, source="synthetic")

    assert band.distribution == "weibull"
    assert band.sample_size == 500
    assert band.source == "synthetic"
    assert band.ratio_min > 0
    assert band.ratio_max > band.ratio_min
    assert band.median_ratio > 0
    lower, upper = band.interval()
    assert lower < band.median_ratio < upper
    assert lower > 0
    payload = band.to_dict()
    assert payload["lower_ratio"] == pytest.approx(lower)
    assert payload["upper_ratio"] == pytest.approx(upper)


def test_band_coverage_is_near_the_nominal_level() -> None:
    observed, predicted = _synthetic(seed=42)
    band = fit_ratio_band(observed, predicted, level=0.90)
    holdout_observed, holdout_predicted = _synthetic(seed=7)

    coverage = band_coverage(holdout_observed, holdout_predicted, band)

    assert 0.75 <= coverage <= 0.99


def test_apply_ratio_band_returns_lower_and_upper() -> None:
    observed, predicted = _synthetic()
    band = fit_ratio_band(observed, predicted)

    lower, upper = apply_ratio_band(np.array([10.0, 20.0]), band)

    assert lower.shape == (2,)
    assert upper.shape == (2,)
    assert np.all(lower < upper)


def test_band_can_be_built_from_a_manual_curve_band() -> None:
    band = CurveBand(
        distribution="weibull",
        parameters={"a": 3.0, "beta": 1.5, "s": 1.0},
        level=0.90,
        sample_size=10,
        median_ratio=1.4,
    )

    lower, upper = band.interval()

    assert lower < 1.4 < upper


def test_invalid_level_is_rejected() -> None:
    observed, predicted = _synthetic()

    with pytest.raises(ValueError, match="level must be between 0 and 1"):
        fit_ratio_band(observed, predicted, level=1.0)


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="same length"):
        fit_ratio_band([1.0, 2.0], [1.0])


def test_empty_arrays_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        fit_ratio_band([], [])


def test_non_positive_predictions_are_rejected() -> None:
    with pytest.raises(ValueError, match="predicted values must be positive"):
        fit_ratio_band([1.0, 2.0], [1.0, 0.0])


def test_distribution_without_inverse_cdf_is_rejected() -> None:
    observed, predicted = _synthetic(n=100)

    with pytest.raises(ValueError, match="no inverse CDF"):
        fit_ratio_band(observed, predicted, distribution="gamma")


def test_read_validation_csv_and_write_band(tmp_path) -> None:
    table = pd.DataFrame(
        {
            "stratum": ["HW"] * 10 + ["BA"],
            "age": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 10],
            "observed": [11.0, 26.0, 27.0, 60.0, 60.0, 84.0, 56.0, 128.0, 94.5, 125.0, 9.0],
            "predicted": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 10.0],
        }
    )
    source = tmp_path / "validation.csv"
    table.to_csv(source, index=False)

    observed, predicted = read_validation_csv(
        source,
        observed_column="observed",
        predicted_column="predicted",
        filter_column="stratum",
        filter_value="HW",
    )
    assert observed.tolist() == [11.0, 26.0, 27.0, 60.0, 60.0, 84.0, 56.0, 128.0, 94.5, 125.0]
    assert predicted.tolist() == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]

    band = fit_ratio_band(observed, predicted, bins=5)
    target = tmp_path / "band.csv"
    write_band_csv(
        target,
        age=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        predicted=predicted,
        band=band,
    )

    result = pd.read_csv(target)
    assert list(result.columns) == ["age", "predicted", "lower", "upper"]
    assert (result["lower"] <= result["upper"]).all()


def test_read_validation_csv_rejects_missing_columns(tmp_path) -> None:
    source = tmp_path / "validation.csv"
    source.write_text("observed,predicted\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing column"):
        read_validation_csv(source, observed_column="observed", predicted_column="missing")
