"""Tests for research-forest FAIB plot selection and hold-out splitting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from nemora.ingest.selection import (
    ForestPlotTarget,
    assign_holdout_split,
    resolve_tree_detail,
    select_forest_plots,
    selection_manifest,
)


def _target() -> ForestPlotTarget:
    return ForestPlotTarget(
        name="fixture forest",
        latitude=0.0,
        longitude=0.0,
        radius_km=10.0,
        bec_codes=("AAA1", "BBB1"),
        min_projected_age=60.0,
    )


def _site_header() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SITE_IDENTIFIER": ["S1", "S2", "S3", "S4"],
            "Latitude": [0.01, 0.02, 0.03, 0.50],
            "Longitude": [0.0, 0.0, 0.0, 0.0],
            "BEC_ZONE": ["A", "B", "C", "A"],
            "BEC_SBZ": ["AA", "BB", "CC", "AA"],
            "BEC_VAR": [1.0, 1.0, 1.0, 1.0],
        }
    )


def _plot_header() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SITE_IDENTIFIER": ["S1", "S2", "S3", "S4"],
            "CLSTR_ID": ["C1", "C2", "C3", "C4"],
            "VISIT_NUMBER": [1, 1, 1, 1],
            "PLOT": [1, 11, 2, 3],
        }
    )


def _sample_by_visit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SITE_IDENTIFIER": ["S1", "S2", "S3", "S4"],
            "CLSTR_ID": ["C1", "C2", "C3", "C4"],
            "VISIT_NUMBER": [1, 1, 1, 1],
            "FIRST_MSMT": ["Y", "Y", "N", "Y"],
            "MEAS_DT": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
            "PROJ_AGE_1": [70.0, 80.0, 90.0, 70.0],
            "SAMP_TYP": ["F", "F", "F", "F"],
        }
    )


def test_select_forest_plots_applies_bec_age_visit_and_radius_filters() -> None:
    selected = select_forest_plots(
        _site_header(),
        _plot_header(),
        _sample_by_visit(),
        target=_target(),
    )

    assert selected["plot_id"].tolist() == ["C1_v1_p1", "C2_v1_p11"]
    assert selected["bec"].tolist() == ["AAA1", "BBB1"]
    assert selected["projected_age"].tolist() == [70.0, 80.0]
    assert selected["distance_km"].iloc[0] < selected["distance_km"].iloc[1]
    assert selected["distance_km"].max() < 10.0


def test_select_forest_plots_rejects_no_matches() -> None:
    target = ForestPlotTarget(
        name="empty",
        latitude=0.0,
        longitude=0.0,
        radius_km=1.0,
        bec_codes=("ZZZ",),
    )

    with pytest.raises(ValueError, match="no plots matched"):
        select_forest_plots(
            _site_header(),
            _plot_header(),
            _sample_by_visit(),
            target=target,
        )


def test_select_forest_plots_requires_source_columns() -> None:
    with pytest.raises(KeyError, match="BEC_VAR"):
        select_forest_plots(
            _site_header().drop(columns=["BEC_VAR"]),
            _plot_header(),
            _sample_by_visit(),
            target=_target(),
        )


def test_assign_holdout_split_is_deterministic_and_ordered_independent() -> None:
    selected = select_forest_plots(
        _site_header(),
        _plot_header(),
        _sample_by_visit(),
        target=_target(),
    )

    first = assign_holdout_split(selected, holdout_fraction=0.5, seed=7)
    second = assign_holdout_split(selected.iloc[::-1], holdout_fraction=0.5, seed=7)

    assert set(first["split"]) == {"train", "holdout"}
    second_by_plot = second.set_index("plot_id")["split"]
    assert second_by_plot.loc[first["plot_id"]].tolist() == first["split"].tolist()


def test_assign_holdout_split_rejects_too_few_plots() -> None:
    with pytest.raises(ValueError, match="at least two"):
        assign_holdout_split(pd.DataFrame({"plot_id": ["only"]}))


def test_selection_manifest_records_counts_and_hashes_without_plot_coordinates() -> None:
    selected = select_forest_plots(
        _site_header(),
        _plot_header(),
        _sample_by_visit(),
        target=_target(),
    )
    split = assign_holdout_split(selected, holdout_fraction=0.5, seed=7)

    manifest = selection_manifest(
        split,
        target=_target(),
        holdout_fraction=0.5,
        split_seed=7,
    )

    assert manifest["counts"] == {"total": 2, "train": 1, "holdout": 1}
    assert manifest["summary"]["bec_counts"] == {"AAA1": 1, "BBB1": 1}
    assert len(manifest["summary"]["plot_ids_sha256"]) == 64
    assert "latitude" not in manifest["summary"]
    assert "longitude" not in manifest["summary"]


def _selected_for_trees() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plot_id": ["C1_v1_p1", "C2_v1_p11"],
            "cluster_id": ["C1", "C2"],
            "visit_number": [1, 1],
            "plot_number": [1, 11],
        }
    )


def _tree_detail() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CLSTR_ID": ["C1", "C1", "C1", "C2", "C3"],
            "VISIT_NUMBER": [1, 1, 1, 1, 1],
            "PLOT": [1, 1, 1, 11, 2],
            "TREE_NO": [1, 2, 3, 1, 1],
            "SPECIES": ["FD", "FD", "FD", "HW", "FD"],
            "DBH": [20.0, 30.0, None, 25.0, 22.0],
            "HEIGHT": [25.0, 28.0, 30.0, 24.0, 20.0],
            "LV_D": ["L", "L", "L", "D", "L"],
            "PHF_TREE": [2.0, 2.0, 2.0, 1.0, 1.0],
        }
    )


def test_resolve_tree_detail_filters_selected_plots_and_live_trees(
    tmp_path: Path,
) -> None:
    selected = _selected_for_trees()

    trees = resolve_tree_detail(_tree_detail(), selected)

    assert list(trees.columns) == [
        "plot_id",
        "tree_id",
        "species_code",
        "dbh_cm",
        "height_m",
        "tally",
        "status",
    ]
    assert trees["plot_id"].tolist() == ["C1_v1_p1", "C1_v1_p1"]
    assert trees["tree_id"].tolist() == ["C1_v1_p1-T00001", "C1_v1_p1-T00002"]
    assert trees["tally"].tolist() == [2.0, 2.0]
    assert trees["status"].tolist() == ["live", "live"]

    detail_path = tmp_path / "tree_detail.csv"
    _tree_detail().to_csv(detail_path, index=False)
    from_path = resolve_tree_detail(detail_path, selected, chunk_size=1)

    pd.testing.assert_frame_equal(from_path, trees, check_dtype=False)


def test_resolve_tree_detail_can_keep_dead_trees() -> None:
    trees = resolve_tree_detail(_tree_detail(), _selected_for_trees(), live_only=False)

    assert len(trees) == 3
    assert "dead" in trees["status"].tolist()


def test_resolve_tree_detail_returns_empty_canonical_frame() -> None:
    trees = resolve_tree_detail(
        _tree_detail(),
        pd.DataFrame(
            {
                "plot_id": ["missing_v1_p9"],
                "cluster_id": ["MISSING"],
                "visit_number": [1],
                "plot_number": [9],
            }
        ),
    )

    assert trees.empty
    assert list(trees.columns) == [
        "plot_id",
        "tree_id",
        "species_code",
        "dbh_cm",
        "height_m",
        "tally",
        "status",
    ]


def test_resolve_tree_detail_requires_live_status_column() -> None:
    with pytest.raises(KeyError, match="LV_D"):
        resolve_tree_detail(
            _tree_detail().drop(columns=["LV_D"]),
            _selected_for_trees(),
        )


def test_select_forest_plots_supports_a_distance_ring() -> None:
    target = ForestPlotTarget(
        name="ring",
        latitude=0.0,
        longitude=0.0,
        radius_km=10.0,
        bec_codes=("AAA1", "BBB1"),
        min_distance_km=2.0,
    )

    selected = select_forest_plots(
        _site_header(),
        _plot_header(),
        _sample_by_visit(),
        target=target,
    )

    assert selected["plot_id"].tolist() == ["C2_v1_p11"]
    assert selected["distance_km"].min() > 2.0
