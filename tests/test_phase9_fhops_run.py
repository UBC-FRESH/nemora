"""Checks for the Phase 9 FHOPS system summary."""

from __future__ import annotations

import json
from pathlib import Path


def test_phase9_fhops_system_summary_records_run() -> None:
    payload = json.loads(
        Path("notes/reports/phase9_fhops_system_summary.json").read_text(encoding="utf-8")
    )

    assert payload["fhops_version"] == "1.2.0"
    assert payload["tree_count"] == 1978
    assert payload["unreachable_count"] == 0
    assert payload["processed_volume_m3"] > 0.0
    assert payload["truck_summary"]["dispatched_loads"] > 0
    assert "work_required" in payload["warnings"][0]
