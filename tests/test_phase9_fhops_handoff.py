"""Tests for the Nemora-to-FHOPS tree-inventory handoff."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from nemora.export import export_fhops_tree_inventory


def _trees() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tree_id": ["T1", "T2", "T3", "T4", "T5"],
            "species_code": ["FD", "FD", "HW", "SW", "AT"],
            "dbh_cm": [18.0, 22.0, 26.0, 31.0, 14.0],
            "height_m": [15.0, 18.0, 20.0, 24.0, 12.0],
        }
    )


def test_export_writes_fhops_contract_files(tmp_path: Path) -> None:
    result = export_fhops_tree_inventory(
        _trees(),
        tmp_path,
        block_id="B01",
        inventory_id="phase9-fixture",
    )

    assert result.tree_count == 5
    assert result.inventory_path.exists()
    assert result.metadata_path.exists()

    with result.inventory_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 5
    assert rows[0]["block_id"] == "B01"
    assert rows[0]["status"] == "live"
    assert rows[0]["confidence"] == "1.0"

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "fhops.tree.inventory.v1"
    assert metadata["tree_count"] == 5
    assert metadata["inventory_sha256"] == result.inventory_sha256
    assert metadata["units"]["dbh"] == "cm"


def test_export_round_trips_through_fhops_loader(tmp_path: Path) -> None:
    pytest.importorskip("fhops")
    from fhops.tree_inventory import load_tree_inventory

    result = export_fhops_tree_inventory(
        _trees(),
        tmp_path,
        block_id="B01",
        inventory_id="phase9-roundtrip",
    )
    inventory = load_tree_inventory(result.inventory_path)

    assert inventory.metadata.inventory_id == "phase9-roundtrip"
    assert inventory.metadata.tree_count == 5
    assert len(inventory.rows) == 5
    assert {row.block_id for row in inventory.rows} == {"B01"}
    assert inventory.rows[0].dbh_cm == pytest.approx(18.0)
