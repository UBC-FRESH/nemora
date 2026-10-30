"""Checks for the Phase 11 multi-block scenario catalog."""

from __future__ import annotations

import json
from pathlib import Path


def test_phase11_catalog_has_two_verified_scenarios() -> None:
    payload = json.loads(
        Path("notes/reports/phase11_scenario_catalog.json").read_text(encoding="utf-8")
    )

    assert payload["schema_version"] == "nemora.scenario_catalog.v1"
    assert len(payload["scenarios"]) == 2
    blocks = {item["block_id"] for item in payload["scenarios"]}
    landings = {item["landing_id"] for item in payload["scenarios"]}
    assert blocks == {"B01", "B02"}
    assert landings == {"L1", "L2"}
    assert all(item["pin_valid"] for item in payload["scenarios"])
    assert all(not item["scenario_path"].startswith("/") for item in payload["scenarios"])
