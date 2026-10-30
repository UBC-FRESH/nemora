"""Tests for the independent inventory-volume check."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from nemora.yield_check import (
    compare_volume_estimates,
    estimate_inventory_volume,
    estimate_tree_volume_m3,
)


def test_tree_volume_formula_matches_paraboloid_definition() -> None:
    volume = estimate_tree_volume_m3(20.0, 20.0)
    expected = (math.pi / 4.0) * (0.2**2) * (20.0 * 0.7) * 0.5
    assert volume == pytest.approx(expected)


def test_inventory_volume_sums_trees_and_compares() -> None:
    trees = pd.DataFrame({"dbh_cm": [20.0, 30.0], "height_m": [20.0, 25.0]})
    estimate = estimate_inventory_volume(trees)
    comparison = compare_volume_estimates(
        estimate.total_volume_m3,
        registered_work_required_m3=estimate.total_volume_m3,
        fhops_processed_m3=estimate.total_volume_m3 * 1.2,
    )

    assert estimate.tree_count == 2
    assert estimate.total_volume_m3 > 0.0
    assert comparison["estimate_vs_registered_relative"] == 0.0
    assert comparison["estimate_vs_processed_relative"] == pytest.approx(-1.0 / 6.0)


def test_phase11_yield_report_has_gavin_and_mkrf_checks() -> None:
    payload = json.loads(Path("notes/reports/phase11_yield_check.json").read_text(encoding="utf-8"))

    assert payload["gavin"]["estimate_m3"] > 0.0
    assert payload["mkrf"]["estimate_m3"] > 0.0
    assert "not a taper model" in payload["caveat"]
