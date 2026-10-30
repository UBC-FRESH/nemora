"""Regression checks for the Malcolm Knapp regional validation summary."""

from __future__ import annotations

import json
from pathlib import Path


def test_mkrf_validation_records_pooling_and_local_holdout() -> None:
    payload = json.loads(
        Path("notes/reports/phase9_mkrf_validation.json").read_text(encoding="utf-8")
    )

    assert payload["forest"] == "Malcolm Knapp Research Forest"
    assert payload["pooling"]["local_plots"] == 6
    assert payload["pooling"]["regional_plots"] == 43
    assert payload["pooling"]["regional_ring_km"] == [50.0, 100.0]
    assert payload["model"] == "empirical"
    assert payload["relative_errors"]["mean_dbh_cm"] < 0.0
    assert payload["relative_errors"]["mean_height_m"] < 0.0
    assert "latitude" not in payload["pooling"]
    assert "longitude" not in payload["pooling"]
