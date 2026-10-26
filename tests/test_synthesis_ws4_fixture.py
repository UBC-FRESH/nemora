"""Tests for the deterministic synthetic landscape used by WS4."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from nemora.synthesis.ws4_fixture import (
    FIXTURE_SCHEMA_VERSION,
    TRUTH_COLUMNS,
    curve_value,
    generate_ws4_fixture,
    load_ws4_fixture,
)

FIXTURE_DIR = Path("examples/ws4_synthetic")
FIXTURE_FILES = (
    "fragments.csv",
    "fixture.json",
    "landscape.geojson",
    "recipe.xml",
    "schedule.csv",
    "truth_table.csv",
)


def _generate(target: Path, seed: int = 20261026) -> Path:
    generate_ws4_fixture(target, seed=seed)
    return target


def test_fixture_is_deterministic(tmp_path: Path) -> None:
    first = _generate(tmp_path / "first")
    second = _generate(tmp_path / "second")

    for name in FIXTURE_FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes(), name


def test_checked_in_fixture_matches_generator(tmp_path: Path) -> None:
    generated = _generate(tmp_path / "generated")

    for name in FIXTURE_FILES:
        assert (generated / name).read_bytes() == (FIXTURE_DIR / name).read_bytes(), name


def test_truth_table_reconciles_with_fragments_and_curve() -> None:
    fixture = load_ws4_fixture(FIXTURE_DIR)
    truth = fixture.truth_table
    fragments = pd.read_csv(fixture.fragments_path)

    assert fixture.metadata["schema_version"] == FIXTURE_SCHEMA_VERSION
    assert fixture.seed == 20261026
    assert list(truth.columns) == list(TRUTH_COLUMNS)
    assert len(truth) == 12
    assert truth["area_ha"].sum() == pytest.approx(36.0, abs=1e-6)
    assert truth["age"].tolist() == [40, 60, 80, 40, 60, 80, 40, 60, 80, 40, 60, 80]
    assert truth["volume_per_ha"].tolist() == [curve_value(age) for age in truth["age"]]
    expected = (truth["area_ha"] * truth["volume_per_ha"]).sum()
    assert truth["expected_volume_m3"].sum() == pytest.approx(expected, abs=1e-6)
    assert fixture.metadata["totals"]["expected_volume_m3"] == pytest.approx(expected, abs=1e-6)

    assert fragments["BLOCK"].tolist() == truth["block_id"].tolist()
    assert fragments["AREA_HA"].tolist() == pytest.approx(truth["area_ha"].tolist())
    assert fragments["F_AGE"].tolist() == truth["age"].tolist()
    assert fragments["IFM"].unique().tolist() == ["unmanaged"]


def test_missing_seed_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "fixture"
    generated = _generate(target)
    manifest_path = generated / "fixture.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    del payload["seed"]
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="seed"):
        load_ws4_fixture(target)


def test_missing_truth_table_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "fixture"
    generated = _generate(target)
    (generated / "truth_table.csv").unlink()

    with pytest.raises(FileNotFoundError, match="truth table"):
        load_ws4_fixture(target)


def test_ws4_build_and_simulate_matches_truth_table(tmp_path: Path) -> None:
    pytest.importorskip("ws4")
    from ws4.build.compiler import build_canonical_dataset
    from ws4.sim.engine import Simulator

    fixture = load_ws4_fixture(FIXTURE_DIR)
    dataset = build_canonical_dataset(
        fixture.recipe_path,
        fixture.fragments_path,
        tmp_path / "compiled",
    )
    simulator = Simulator.from_dataset(dataset)
    simulator.run(
        schedule_csv=fixture.schedule_path,
        years=2,
        out_dir=tmp_path / "sim",
    )

    values = pd.read_csv(tmp_path / "sim" / "account_values.csv")
    feature = values[(values["account"] == "feature.Volume") & (values["offset"] == 0)]
    expected = float(fixture.truth_table["expected_volume_m3"].sum())
    assert float(feature["value"].sum()) == pytest.approx(expected, abs=1e-6)
