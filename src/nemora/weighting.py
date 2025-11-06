"""Weighting and expansion utilities for inventory tallies."""

from __future__ import annotations

import numpy as np


def hps_expansion_factor(dbh_cm: np.ndarray, baf: float) -> np.ndarray:
    """Return stand table expansion factors for HPS tallies."""
    dbh = np.asarray(dbh_cm, dtype=float)
    radius_m = dbh * 0.01 / 2.0
    basal_area = np.pi * np.square(radius_m)
    return baf / basal_area


def hps_compression_factor(dbh_cm: np.ndarray, baf: float) -> np.ndarray:
    """Inverse of the HPS expansion factor."""
    return 1.0 / hps_expansion_factor(dbh_cm, baf=baf)


def apply_weights(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Elementwise weighting helper."""
    vals = np.asarray(values, dtype=float)
    wts = np.asarray(weights, dtype=float)
    if vals.shape != wts.shape:
        raise ValueError("Values and weights must share the same shape.")
    return vals * wts
