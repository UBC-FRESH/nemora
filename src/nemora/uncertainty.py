"""Curve uncertainty from observed/predicted ratios.

This module quantifies the uncertainty of an existing curve family. It does not
fit a new growth model. The input is a paired validation table: an observed
value and the curve's prediction at the same stand. The ratio
``observed / predicted`` is fitted with a positive-support distribution from the
Nemora registry, and the resulting interval becomes a multiplicative band.

The band is empirical. It inherits the validation sample's bias, species mix,
age range, and measurement error; it is not a parameter covariance from the
original curve fit.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .core import InventorySpec
from .distributions import get_distribution
from .fit import default_fit_config, fit_inventory

__all__ = [
    "CurveBand",
    "apply_ratio_band",
    "band_coverage",
    "fit_ratio_band",
    "read_validation_csv",
    "write_band_csv",
]


@dataclass(slots=True)
class CurveBand:
    """A multiplicative curve band fitted to observed/predicted ratios.

    Attributes
    ----------
    distribution : str
        Registry name of the fitted positive-support distribution.
    parameters : dict[str, float]
        Fitted distribution parameters.
    level : float
        Nominal coverage level, for example ``0.90``.
    sample_size : int
        Number of validation pairs used for the fit.
    source : str | None
        Optional label for the validation source.
    ratio_min, ratio_max, median_ratio : float
        Observed ratio range and median, for diagnostics.
    gof : dict[str, float]
        Goodness-of-fit metrics returned by :func:`nemora.fit.fit_inventory`.
    """

    distribution: str
    parameters: dict[str, float]
    level: float
    sample_size: int
    source: str | None = None
    ratio_min: float = 0.0
    ratio_max: float = 0.0
    median_ratio: float = 0.0
    gof: dict[str, float] = field(default_factory=dict)

    def interval(self) -> tuple[float, float]:
        """Return the lower and upper ratio quantiles for the band.

        Returns
        -------
        tuple[float, float]
            The ``(1 - level) / 2`` and ``1 - (1 - level) / 2`` quantiles of the
            fitted distribution.

        Raises
        ------
        ValueError
            If the fitted distribution has no inverse CDF or the level is
            outside ``(0, 1)``.
        """
        if not 0.0 < self.level < 1.0:
            raise ValueError("level must be between 0 and 1")
        distribution = get_distribution(self.distribution)
        if distribution.inverse_cdf is None:
            raise ValueError(f"distribution {self.distribution!r} has no inverse CDF")
        tail = (1.0 - self.level) / 2.0
        quantiles = distribution.inverse_cdf(
            np.array([tail, 1.0 - tail], dtype=float), self.parameters
        )
        return float(quantiles[0]), float(quantiles[1])

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable description of the band."""
        lower, upper = self.interval()
        return {
            "distribution": self.distribution,
            "parameters": dict(self.parameters),
            "level": self.level,
            "sample_size": self.sample_size,
            "source": self.source,
            "ratio_min": self.ratio_min,
            "ratio_max": self.ratio_max,
            "median_ratio": self.median_ratio,
            "lower_ratio": lower,
            "upper_ratio": upper,
            "gof": dict(self.gof),
        }


def _paired_arrays(observed: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return paired one-dimensional arrays."""
    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    if observed_array.ndim != 1 or predicted_array.ndim != 1:
        raise ValueError("observed and predicted must be one-dimensional")
    if observed_array.size != predicted_array.size:
        raise ValueError("observed and predicted must have the same length")
    if observed_array.size == 0:
        raise ValueError("at least one observed/predicted pair is required")
    if not np.all(np.isfinite(observed_array)) or not np.all(np.isfinite(predicted_array)):
        raise ValueError("observed and predicted must be finite")
    if np.any(predicted_array <= 0):
        raise ValueError("predicted values must be positive")
    if np.any(observed_array < 0):
        raise ValueError("observed values must be non-negative")
    return observed_array, predicted_array


def fit_ratio_band(
    observed: ArrayLike,
    predicted: ArrayLike,
    *,
    distribution: str = "weibull",
    bins: int | Sequence[float] = 25,
    level: float = 0.90,
    source: str | None = None,
) -> CurveBand:
    """Fit a multiplicative band to observed/predicted ratios.

    Parameters
    ----------
    observed : array-like
        Observed values from the validation sample.
    predicted : array-like
        Curve predictions for the same observations.
    distribution : str, optional
        Positive-support distribution from the Nemora registry. Defaults to
        ``"weibull"``.
    bins : int or sequence of float, optional
        Histogram bins used to turn the raw ratios into grouped tallies for the
        existing fitting stack. Defaults to 25.
    level : float, optional
        Nominal coverage level. Defaults to ``0.90``.
    source : str | None, optional
        Optional validation-source label.

    Returns
    -------
    CurveBand
        Fitted band and diagnostics.

    Raises
    ------
    ValueError
        If the arrays are invalid, the level is outside ``(0, 1)``, the
        distribution has no inverse CDF, or the fit fails.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between 0 and 1")
    observed_array, predicted_array = _paired_arrays(observed, predicted)
    ratios = observed_array / predicted_array
    if not np.all(np.isfinite(ratios)):
        raise ValueError("observed/predicted ratios must be finite")

    histogram, edges = np.histogram(ratios, bins=bins)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    if np.count_nonzero(histogram) < 5:
        raise ValueError("not enough distinct ratios for a stable fit")

    inventory = InventorySpec(
        name=source or "observed/predicted ratio",
        sampling="histogram",
        bins=midpoints,
        tallies=histogram.astype(float),
    )
    config = default_fit_config(distribution, midpoints, histogram.astype(float))
    results = fit_inventory(
        inventory, distributions=(distribution,), configs={distribution: config}
    )
    if not results:
        raise ValueError(f"fit produced no result for {distribution!r}")
    result = results[0]
    fitted = get_distribution(distribution)
    if fitted.inverse_cdf is None:
        raise ValueError(f"distribution {distribution!r} has no inverse CDF")

    return CurveBand(
        distribution=distribution,
        parameters={key: float(value) for key, value in result.parameters.items()},
        level=level,
        sample_size=int(observed_array.size),
        source=source,
        ratio_min=float(np.min(ratios)),
        ratio_max=float(np.max(ratios)),
        median_ratio=float(np.median(ratios)),
        gof={key: float(value) for key, value in result.gof.items()},
    )


def apply_ratio_band(predicted: ArrayLike, band: CurveBand) -> tuple[np.ndarray, np.ndarray]:
    """Apply a multiplicative band to predicted values.

    Parameters
    ----------
    predicted : array-like
        Central predicted values. Must be positive.
    band : CurveBand
        Fitted band.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Lower and upper bounds, in the same shape as ``predicted``.

    Raises
    ------
    ValueError
        If a predicted value is not positive.
    """
    values = np.asarray(predicted, dtype=float)
    if np.any(values <= 0):
        raise ValueError("predicted values must be positive")
    lower_ratio, upper_ratio = band.interval()
    return values * lower_ratio, values * upper_ratio


def band_coverage(observed: ArrayLike, predicted: ArrayLike, band: CurveBand) -> float:
    """Return the fraction of observations inside the band.

    Parameters
    ----------
    observed : array-like
        Observed values.
    predicted : array-like
        Central predictions.
    band : CurveBand
        Fitted band.

    Returns
    -------
    float
        Fraction of paired observations with
        ``lower <= observed <= upper``.

    Raises
    ------
    ValueError
        If the paired arrays are invalid.
    """
    observed_array, predicted_array = _paired_arrays(observed, predicted)
    lower, upper = apply_ratio_band(predicted_array, band)
    return float(np.mean((observed_array >= lower) & (observed_array <= upper)))


def read_validation_csv(
    path: Path,
    *,
    observed_column: str,
    predicted_column: str,
    filter_column: str | None = None,
    filter_value: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read observed/predicted pairs from a CSV validation table.

    Parameters
    ----------
    path : pathlib.Path
        CSV file.
    observed_column : str
        Column holding the observed values.
    predicted_column : str
        Column holding the curve predictions.
    filter_column : str | None, optional
        Optional categorical filter column.
    filter_value : str | None, optional
        Value to retain in ``filter_column``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Observed and predicted arrays.

    Raises
    ------
    ValueError
        If a required column is missing or the filtered table is empty.
    """
    frame = pd.read_csv(path)
    for column in (observed_column, predicted_column):
        if column not in frame.columns:
            raise ValueError(f"{path} is missing column {column!r}")
    if filter_column is not None:
        if filter_column not in frame.columns:
            raise ValueError(f"{path} is missing filter column {filter_column!r}")
        if filter_value is None:
            raise ValueError("filter_value is required when filter_column is set")
        frame = frame[frame[filter_column].astype(str) == str(filter_value)]
    if frame.empty:
        raise ValueError(f"{path} has no rows after filtering")
    return (
        frame[observed_column].to_numpy(dtype=float),
        frame[predicted_column].to_numpy(dtype=float),
    )


def write_band_csv(
    path: Path,
    *,
    age: ArrayLike,
    predicted: ArrayLike,
    band: CurveBand,
) -> None:
    """Write a per-observation band table to CSV.

    Parameters
    ----------
    path : pathlib.Path
        Destination CSV path.
    age : array-like
        Age or x-axis values aligned with ``predicted``.
    predicted : array-like
        Central predictions.
    band : CurveBand
        Fitted band.

    Raises
    ------
    ValueError
        If the arrays are invalid or the band cannot be applied.
    """
    ages = np.asarray(age, dtype=float)
    values = np.asarray(predicted, dtype=float)
    if ages.ndim != 1 or values.ndim != 1 or ages.size != values.size:
        raise ValueError("age and predicted must be one-dimensional and aligned")
    lower, upper = apply_ratio_band(values, band)
    frame = pd.DataFrame(
        {
            "age": ages,
            "predicted": values,
            "lower": lower,
            "upper": upper,
        }
    )
    frame.to_csv(path, index=False)
