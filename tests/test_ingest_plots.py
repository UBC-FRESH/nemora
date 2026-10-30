"""Tests for generic plot resolution and tally materialization."""

from __future__ import annotations

import pandas as pd
import pytest

from nemora.ingest.plots import (
    PlotInventorySpec,
    build_inventory_release,
    materialize_tally_trees,
)
from nemora.ingest.schema import InventoryProvenance, InventoryValidationError


def _provenance() -> InventoryProvenance:
    return InventoryProvenance(
        source_id="test-source",
        source_version="fixture-1",
        source_uri="https://example.invalid/inventory",
        retrieved_utc="2026-10-30T00:00:00+00:00",
        licence="Public",
        notes="Synthetic fixture for plot resolution tests.",
    )


def _plots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plot": ["P1", "P2"],
            "cluster": ["C1", "C1"],
            "baf": [12.0, 12.0],
            "lat": [49.0, 49.1],
            "lon": [-122.0, -122.1],
        }
    )


def _trees() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "plot": ["P1", "P1", "P2"],
            "diameter": [20.0, 30.0, 25.0],
            "count": [1.0, 2.0, 1.0],
            "species": ["FD", "FD", "HW"],
        }
    )


def _species() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["FD", "HW"],
            "common": ["Douglas-fir", "Western hemlock"],
            "scientific": ["Pseudotsuga menziesii", "Tsuga heterophylla"],
        }
    )


def _spec() -> PlotInventorySpec:
    return PlotInventorySpec(
        release_id="plot-fixture",
        provenance=_provenance(),
        plot_columns={
            "plot_id": "plot",
            "cluster_id": "cluster",
            "baf": "baf",
            "latitude": "lat",
            "longitude": "lon",
        },
        tree_columns={
            "plot_id": "plot",
            "dbh_cm": "diameter",
            "tally": "count",
            "species_code": "species",
        },
        species_columns={
            "species_code": "code",
            "common_name": "common",
            "scientific_name": "scientific",
        },
        tree_defaults={"status": "live"},
        crs="EPSG:3005",
    )


def test_build_inventory_release_maps_columns_and_derives_tree_count() -> None:
    release = build_inventory_release(
        _plots(),
        _trees(),
        spec=_spec(),
        species=_species(),
    )

    assert release.release_id == "plot-fixture"
    assert release.crs == "EPSG:3005"
    assert list(release.plots.columns) == [
        "plot_id",
        "cluster_id",
        "visit_number",
        "plot_number",
        "measurement_date",
        "sample_type",
        "site_identifier",
        "baf",
        "tree_count",
        "latitude",
        "longitude",
    ]
    assert release.plots["tree_count"].tolist() == [2, 1]
    assert release.trees["status"].tolist() == ["live", "live", "live"]
    assert release.trees["species_code"].tolist() == ["FD", "FD", "HW"]
    assert release.species["common_name"].tolist() == [
        "Douglas-fir",
        "Western hemlock",
    ]
    assert release.metadata["column_mapping"]["trees"]["dbh_cm"] == "diameter"


def test_build_inventory_release_rejects_missing_mapped_column() -> None:
    spec = PlotInventorySpec(
        release_id="bad-mapping",
        provenance=_provenance(),
        plot_columns={"plot_id": "plot", "baf": "missing_baf"},
    )

    with pytest.raises(InventoryValidationError, match="missing_baf"):
        build_inventory_release(_plots(), _trees(), spec=spec)


def test_build_inventory_release_rejects_orphan_tree_plot() -> None:
    trees = _trees()
    trees.loc[0, "plot"] = "P3"

    with pytest.raises(InventoryValidationError, match="absent from plots"):
        build_inventory_release(_plots(), trees, spec=_spec())


def test_materialize_tally_trees_expands_and_generates_unique_ids() -> None:
    release = build_inventory_release(
        _plots(),
        _trees(),
        spec=_spec(),
        species=_species(),
    )

    materialized = materialize_tally_trees(release)

    assert len(materialized.trees) == 4
    assert materialized.trees["tally"].tolist() == [1, 1, 1, 1]
    assert materialized.trees["tree_id"].is_unique
    assert materialized.trees["tree_id"].tolist() == [
        "TP1-0001",
        "TP1-0002",
        "TP1-0003",
        "TP2-0001",
    ]
    assert materialized.metadata["tally_grain"] == "individual_tree"
    assert materialized.metadata["materialized_tree_count"] == 4


def test_materialize_tally_trees_preserves_existing_tree_ids() -> None:
    release = build_inventory_release(
        _plots(),
        _trees(),
        spec=_spec(),
        species=_species(),
    )
    release.trees.loc[0, "tree_id"] = "FIELD-1"

    materialized = materialize_tally_trees(release)

    assert materialized.trees.loc[0, "tree_id"] == "FIELD-1"
    assert materialized.trees["tree_id"].is_unique


def test_materialize_tally_trees_rejects_fractional_tally() -> None:
    trees = _trees()
    trees.loc[0, "count"] = 1.5
    release = build_inventory_release(_plots(), trees, spec=_spec(), species=_species())

    with pytest.raises(InventoryValidationError, match="fractional"):
        materialize_tally_trees(release)
