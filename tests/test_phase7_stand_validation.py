"""Tests for plot-conditioned stand validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nemora.validation import (
    stand_metrics_from_trees,
    validate_plot_conditioned_stand,
)


def _trees(plot_id: str, *, offset: float = 0.0) -> pd.DataFrame:
    dbh = np.arange(10.0, 50.0, 2.0) + offset
    return pd.DataFrame(
        {
            "plot_id": plot_id,
            "tree_id": [f"{plot_id}-{index:03d}" for index in range(dbh.size)],
            "species_code": ["FD" if index % 3 else "HW" for index in range(dbh.size)],
            "dbh_cm": dbh,
            "height_m": 8.0 + 0.45 * dbh,
            "tally": np.ones(dbh.size),
            "status": "live",
        }
    )


def _training_trees() -> pd.DataFrame:
    return pd.concat([_trees("P1"), _trees("P2", offset=2.0)], ignore_index=True)


def _holdout_trees() -> pd.DataFrame:
    return _trees("P3", offset=1.0)


def test_stand_metrics_from_trees_uses_expansion_tally() -> None:
    metrics = stand_metrics_from_trees(
        pd.DataFrame(
            {
                "dbh_cm": [10.0, 20.0],
                "height_m": [12.0, 24.0],
                "tally": [1.0, 3.0],
                "species_code": ["FD", "FD"],
            }
        )
    )

    assert metrics.trees == 2
    assert metrics.stems_per_ha == 4.0
    assert metrics.qmd_cm == pytest.approx(np.sqrt(325.0))
    assert metrics.basal_area_m2_per_ha == pytest.approx(np.pi * 0.0325)
    assert metrics.mean_height_m == pytest.approx(21.0)


def test_validate_plot_conditioned_stand_reports_metrics_and_uncertainty() -> None:
    result = validate_plot_conditioned_stand(
        _training_trees(),
        _holdout_trees(),
        bootstrap_replicates=3,
        seed=7,
    )

    assert result.training.trees >= 1
    assert result.holdout.trees >= 1
    assert result.synthetic.trees >= 1
    assert set(result.relative_errors) == {
        "stems_per_ha",
        "basal_area_m2_per_ha",
        "qmd_cm",
        "mean_dbh_cm",
        "mean_height_m",
    }
    assert 0.0 <= result.dbh_ks_weighted <= 1.0
    assert 0.0 <= result.species_tv_distance <= 1.0
    assert result.fit_parameters["a"] > 0.0
    assert result.height_fit["slope"] > 0.0
    assert result.successful_bootstrap == 3
    assert result.uncertainty["qmd_cm"]["replicates"] == 3.0
    assert not result.comparison_frame().empty


def test_validate_plot_conditioned_stand_is_deterministic() -> None:
    first = validate_plot_conditioned_stand(
        _training_trees(),
        _holdout_trees(),
        bootstrap_replicates=0,
        seed=11,
    )
    second = validate_plot_conditioned_stand(
        _training_trees(),
        _holdout_trees(),
        bootstrap_replicates=0,
        seed=11,
    )

    pd.testing.assert_frame_equal(first.synthetic_trees, second.synthetic_trees)


def test_validate_plot_conditioned_stand_requires_tree_columns() -> None:
    with pytest.raises(KeyError, match="tally"):
        validate_plot_conditioned_stand(
            _training_trees().drop(columns=["tally"]),
            _holdout_trees(),
        )
