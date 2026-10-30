"""Regression checks for the Phase 8 LiDAR coverage summary."""

from __future__ import annotations

import json
from pathlib import Path


def test_phase8_lidar_metrics_report_resolved_and_missing_coverage() -> None:
    path = Path("notes/reports/phase8_lidar_metrics.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["summary"]["plots_with_valid_canopy"] == 16
    assert payload["summary"]["plots_missing"] == 4
    assert len(payload["plots"]) == 16
    assert len(payload["missing_plots"]) == 4
    assert payload["coverage_expansion"]["resolved_plot"] == "4032469-PSP1_v1_p1"
    assert payload["coverage_expansion"]["resolved_tile"] == "093a031_3_2_1"
    assert "latitude" not in payload["plots"]["4032469-PSP1_v1_p1"]
    assert "longitude" not in payload["plots"]["4032469-PSP1_v1_p1"]
