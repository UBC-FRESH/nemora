"""Regression checks for the Phase 10 Malcolm Knapp stratification."""

from __future__ import annotations

import json
from pathlib import Path


def test_phase10_stratification_records_age_rule_and_tradeoff() -> None:
    payload = json.loads(
        Path("notes/reports/phase10_mkrf_stratified.json").read_text(encoding="utf-8")
    )
    base = json.loads(Path("notes/reports/phase9_mkrf_validation.json").read_text(encoding="utf-8"))

    assert payload["pooling"]["regional_age_years"] == [80.0, 110.0]
    assert payload["pooling"]["local_plots"] == 6
    assert payload["pooling"]["regional_plots"] == 15
    assert payload["relative_errors"]["mean_dbh_cm"] > base["relative_errors"]["mean_dbh_cm"]
    assert payload["relative_errors"]["stems_per_ha"] < base["relative_errors"]["stems_per_ha"]
    assert "latitude" not in payload["pooling"]
