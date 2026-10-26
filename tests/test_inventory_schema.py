"""Tests for the shared ``nemora.inventory.v1`` ingest schema."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nemora.ingest.hps import (
    export_hps_inventory,
    export_hps_outputs,
    load_plot_selections,
    run_hps_pipeline,
)
from nemora.ingest.schema import (
    SCHEMA_VERSION,
    InventoryProvenance,
    InventoryUnits,
    InventoryValidationError,
    inventory_schema,
    read_inventory_release,
    validate_inventory_release,
    write_inventory_release,
)

FIXTURES = Path(__file__).parent / "fixtures"
HPS_FIXTURES = FIXTURES / "hps"
HPS_EXPECTED = FIXTURES / "hps_export_expected"


def _provenance() -> InventoryProvenance:
    return InventoryProvenance(
        source_id="test-faib-psp",
        source_version="fixture-1",
        source_uri="https://example.invalid/faib",
        retrieved_utc="2026-10-26T00:00:00+00:00",
        licence="Public",
        notes="Synthetic fixture for schema tests.",
    )


def _hps_result():
    selections = load_plot_selections(
        HPS_FIXTURES / "plot_header.csv",
        HPS_FIXTURES / "sample_byvisit.csv",
        baf=12.0,
    )
    return run_hps_pipeline(
        HPS_FIXTURES / "tree_detail.csv",
        selections,
        bin_width=1.0,
        bin_origin=0.0,
        chunk_size=2,
    )


def _release():
    return _hps_result().to_inventory_release(
        release_id="hps-fixture",
        provenance=_provenance(),
    )


def test_hps_release_validates_and_has_schema_tables() -> None:
    release = _release()

    assert release.schema_version == SCHEMA_VERSION
    assert release.release_id == "hps-fixture"
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
    assert list(release.trees.columns) == [
        "plot_id",
        "tree_id",
        "species_code",
        "dbh_cm",
        "height_m",
        "tally",
        "status",
    ]
    assert list(release.species.columns) == [
        "species_code",
        "common_name",
        "scientific_name",
    ]
    assert release.units == InventoryUnits()
    assert release.plots["tree_count"].sum() == 4
    assert release.trees["tally"].sum() == 4
    assert release.trees["status"].unique().tolist() == ["live"]
    assert release.plots["measurement_date"].tolist() == ["1992-06-24", "1992-06-24"]

    validate_inventory_release(release)


def test_write_read_round_trip(tmp_path: Path) -> None:
    release = _release()
    target = tmp_path / "release"

    write_inventory_release(release, target)
    loaded = read_inventory_release(target)

    assert loaded.schema_version == SCHEMA_VERSION
    assert loaded.release_id == release.release_id
    assert loaded.provenance == release.provenance
    assert loaded.units == release.units
    assert loaded.metadata == release.metadata
    pd.testing.assert_frame_equal(loaded.plots, release.plots, check_dtype=False)
    pd.testing.assert_frame_equal(loaded.trees, release.trees, check_dtype=False)
    pd.testing.assert_frame_equal(loaded.species, release.species, check_dtype=False)


def test_writer_fills_optional_columns(tmp_path: Path) -> None:
    release = _release()
    release.plots = release.plots.loc[:, ["plot_id"]]
    release.trees = release.trees.loc[:, ["plot_id", "dbh_cm", "tally"]]
    release.species = release.species.loc[:, []]
    target = tmp_path / "release"

    write_inventory_release(release, target)
    loaded = read_inventory_release(target)

    assert list(loaded.plots.columns) == [
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
    assert list(loaded.trees.columns) == [
        "plot_id",
        "tree_id",
        "species_code",
        "dbh_cm",
        "height_m",
        "tally",
        "status",
    ]


def test_missing_required_key_is_rejected() -> None:
    release = _release()
    release.trees = release.trees.drop(columns=["tally"])

    with pytest.raises(InventoryValidationError, match="tally"):
        validate_inventory_release(release)


def test_unknown_unit_is_rejected() -> None:
    release = _release()
    release.units = InventoryUnits(dbh="furlong")

    with pytest.raises(InventoryValidationError, match="units.dbh"):
        validate_inventory_release(release)


def test_missing_timezone_in_provenance_is_rejected() -> None:
    release = _release()
    release.provenance = InventoryProvenance(
        source_id="test",
        source_version="1",
        retrieved_utc="2026-10-26T00:00:00",
    )

    with pytest.raises(InventoryValidationError, match="timezone"):
        validate_inventory_release(release)


def test_duplicate_plot_id_is_rejected() -> None:
    release = _release()
    duplicate = pd.concat([release.plots, release.plots.iloc[[0]]], ignore_index=True)
    release.plots = duplicate

    with pytest.raises(InventoryValidationError, match="duplicate"):
        validate_inventory_release(release)


def test_duplicate_tally_row_is_rejected() -> None:
    release = _release()
    release.trees = pd.concat([release.trees, release.trees.iloc[[0]]], ignore_index=True)

    with pytest.raises(InventoryValidationError, match="duplicate"):
        validate_inventory_release(release)


def test_orphaned_tree_plot_is_rejected() -> None:
    release = _release()
    release.trees.loc[0, "plot_id"] = "missing-plot"

    with pytest.raises(InventoryValidationError, match="absent from plots"):
        validate_inventory_release(release)


def test_missing_schema_version_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "release"
    write_inventory_release(_release(), target)
    manifest_path = target / "release.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["schema_version"]
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(InventoryValidationError, match="schema_version"):
        read_inventory_release(target)


def test_missing_provenance_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "release"
    write_inventory_release(_release(), target)
    manifest_path = target / "release.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["provenance"]
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(InventoryValidationError, match="provenance"):
        read_inventory_release(target)


def test_inventory_schema_is_portable() -> None:
    schema = inventory_schema()

    assert schema["schema_version"] == SCHEMA_VERSION
    assert schema["tables"]["trees"]["required"] == ["plot_id", "dbh_cm", "tally"]
    assert schema["tables"]["trees"]["tally_key"] == ["plot_id", "dbh_cm"]
    assert schema["units"]["dbh"] == ["cm"]
    assert schema["provenance_required"] == [
        "source_id",
        "source_version",
        "retrieved_utc",
    ]


def test_hps_legacy_export_bytes_are_unchanged(tmp_path: Path) -> None:
    result = _hps_result()
    export_hps_outputs(
        result.tallies,
        result.manifest,
        output_dir=tmp_path,
        manifest_path=tmp_path / "manifest.csv",
        quiet=True,
    )

    for expected in sorted(HPS_EXPECTED.iterdir()):
        assert (tmp_path / expected.name).read_bytes() == expected.read_bytes()


def test_hps_inventory_export_preserves_tallies_and_legacy_bytes(tmp_path: Path) -> None:
    result = _hps_result()
    legacy_dir = tmp_path / "legacy"
    inventory_dir = tmp_path / "inventory"

    export_hps_outputs(
        result.tallies,
        result.manifest,
        output_dir=legacy_dir,
        manifest_path=legacy_dir / "manifest.csv",
        quiet=True,
    )
    release = export_hps_inventory(
        result,
        inventory_dir,
        release_id="hps-fixture",
        provenance=_provenance(),
    )

    for expected in sorted(HPS_EXPECTED.iterdir()):
        assert (legacy_dir / expected.name).read_bytes() == expected.read_bytes()

    rebuilt = (
        release.trees.groupby(["plot_id", "dbh_cm"], as_index=False)["tally"]
        .sum()
        .sort_values(["plot_id", "dbh_cm"])
        .reset_index(drop=True)
    )
    expected = result.tallies_frame.sort_values(["plot_id", "dbh_cm"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt, expected, check_dtype=False)

    assert (inventory_dir / "release.json").exists()
    assert (inventory_dir / "plots.csv").exists()
    assert (inventory_dir / "trees.csv").exists()
    assert (inventory_dir / "species.csv").exists()
    assert read_inventory_release(inventory_dir).release_id == "hps-fixture"
