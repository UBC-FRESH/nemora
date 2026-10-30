"""Tests for FHOPS scenario pinning and block reconciliation."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest
import yaml

from nemora.export import export_fhops_tree_inventory
from nemora.fhops_handoff import (
    register_fhops_scenario,
    validate_fhops_scenario_pin,
)


def _template(tmp_path: Path) -> Path:
    template = tmp_path / "template"
    (template / "data").mkdir(parents=True)
    (template / "scenario.yaml").write_text(
        yaml.safe_dump({"name": "fixture", "data": {"blocks": "data/blocks.csv"}}),
        encoding="utf-8",
    )
    with (template / "data" / "blocks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "work_required", "volume_per_ha_m3"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow({"id": "B01", "work_required": "19.26", "volume_per_ha_m3": "19.26"})
    return template


def _inventory(tmp_path: Path) -> Path:
    trees = pd.DataFrame(
        {
            "tree_id": ["T1", "T2"],
            "species_code": ["FD", "HW"],
            "dbh_cm": [20.0, 25.0],
            "height_m": [18.0, 20.0],
        }
    )
    out = tmp_path / "inventory"
    export_fhops_tree_inventory(trees, out, block_id="B01", inventory_id="fixture")
    return out


def test_register_scenario_pins_inventory_and_reconciles_block(tmp_path: Path) -> None:
    template = _template(tmp_path)
    inventory = _inventory(tmp_path)
    output = tmp_path / "scenario_pinned"

    result = register_fhops_scenario(
        template,
        output,
        inventory,
        block_id="B01",
        work_required_m3=530.543340,
    )

    assert result.work_required_m3 == 530.543340
    assert validate_fhops_scenario_pin(result.scenario_path)
    payload = yaml.safe_load(result.scenario_path.read_text(encoding="utf-8"))
    assert payload["tree_inventory"]["inventory_sha256"] == result.inventory_sha256
    assert payload["tree_inventory"]["metadata_sha256"] == result.metadata_sha256
    with (output / "data" / "blocks.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert float(rows[0]["work_required"]) == 530.543340


def test_pin_validation_fails_after_inventory_change(tmp_path: Path) -> None:
    template = _template(tmp_path)
    inventory = _inventory(tmp_path)
    output = tmp_path / "scenario_pinned"
    result = register_fhops_scenario(template, output, inventory)
    assert validate_fhops_scenario_pin(result.scenario_path)

    result.inventory_path.write_text(
        result.inventory_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    assert not validate_fhops_scenario_pin(result.scenario_path)


def test_registration_rejects_unknown_block(tmp_path: Path) -> None:
    template = _template(tmp_path)
    inventory = _inventory(tmp_path)

    with pytest.raises(ValueError, match="not declared"):
        register_fhops_scenario(template, tmp_path / "bad", inventory, block_id="B02")
