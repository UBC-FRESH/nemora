from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from nemora.synthesis import exporters, stands, stems
from nemora.synthesis.helpers import StandDBHSampler


def _analytic_sampler(sample_size: int = 5) -> StandDBHSampler:
    entry = stands.StandBootstrapLibraryEntry(
        identifier="analytic-1",
        source="analytic",
        metadata={
            "distribution": "lognormal",
            "parameters": {"mu": 2.0, "sigma2": 0.25},
            "sample_size": sample_size,
            "mode": "analytic",
        },
        dbh_vectors={},
    )
    assignment = stands.StandBootstrapAssignment(
        stand_id="stand-0001",
        vegetation_type="fir",
        age_class="60-80",
        area=4.0,
        bootstrap_id="analytic-1",
    )
    return StandDBHSampler(assignment=assignment, entry=entry)


def test_place_trees_uniform_with_spacing() -> None:
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    rng = np.random.default_rng(123)
    points = stems.place_trees(
        polygon,
        4,
        rng=rng,
        config=stems.TreePlacementConfig(min_spacing=0.2, mode="poisson"),
    )
    assert points.shape == (4, 2)
    assert np.all((points >= 0.0) & (points <= 1.0))
    # Minimum spacing check
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            dist = np.linalg.norm(points[i] - points[j])
            assert dist >= 0.2


def test_place_trees_with_dbh_pairs_sampler_draws() -> None:
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    sampler = _analytic_sampler(sample_size=3)
    rng = np.random.default_rng(321)
    records = stems.place_trees_with_dbh(
        polygon,
        sampler,
        count=3,
        rng=rng,
        config=stems.TreePlacementConfig(min_spacing=0.05),
    )
    assert len(records) == 3
    for record in records:
        assert record["stand_id"] == sampler.assignment.stand_id
        assert record["bootstrap_id"] == sampler.bootstrap_id
        assert record["sampler_type"] == "analytic"
        x_raw = record["x"]
        y_raw = record["y"]
        dbh_raw = record["dbh"]
        assert isinstance(x_raw, int | float)
        assert isinstance(y_raw, int | float)
        assert isinstance(dbh_raw, int | float)
        x = float(x_raw)
        y = float(y_raw)
        dbh = float(dbh_raw)
        assert 0.0 <= x <= 1.0
        assert 0.0 <= y <= 1.0
        assert dbh > 0.0


def test_attach_tree_attributes_adds_basal_area_and_height() -> None:
    sampler = _analytic_sampler(sample_size=2)
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    rng = np.random.default_rng(99)
    records = stems.place_trees_with_dbh(
        polygon,
        sampler,
        count=2,
        rng=rng,
        config=stems.TreePlacementConfig(min_spacing=0.05),
    )
    enriched = stems.attach_tree_attributes(records)
    assert len(enriched) == 2
    for record in enriched:
        attrs = record["attributes"]
        assert isinstance(attrs, stems.TreeAttributes)
        assert attrs.dbh_cm > 0.0
        assert attrs.height_m > 0.0
        assert 0.0 < attrs.crown_ratio <= 1.0
        assert attrs.basal_area_m2 > 0.0
        assert attrs.biomass_tonnes >= 0.0
        assert attrs.bark_thickness_cm >= 0.0


def test_stratified_mode_spreads_points_across_grid() -> None:
    polygon = np.array([[0.0, 0.0], [1.2, 0.0], [1.2, 1.0], [0.0, 1.0]])
    points = stems.place_trees(
        polygon,
        4,
        config=stems.TreePlacementConfig(mode="stratified"),
    )
    assert points.shape == (4, 2)
    xs = np.sort(points[:, 0])
    ys = np.sort(points[:, 1])
    assert xs[0] < 0.5 < xs[-1]
    assert ys[0] < 0.5 < ys[-1]


def test_clustered_mode_respects_min_spacing() -> None:
    polygon = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    rng = np.random.default_rng(7)
    cfg = stems.TreePlacementConfig(mode="clustered", min_spacing=0.05, cluster_spread=0.1)
    points = stems.place_trees(polygon, 12, rng=rng, config=cfg)
    assert points.shape == (12, 2)
    assert np.all(points[:, 0] >= 0.0) and np.all(points[:, 0] <= 2.0)
    assert np.all(points[:, 1] >= 0.0) and np.all(points[:, 1] <= 1.0)
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            assert np.linalg.norm(points[i] - points[j]) >= 0.05


def test_placement_stats_match_expected_mean_dbh() -> None:
    sampler = _analytic_sampler(sample_size=50)
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    rng = np.random.default_rng(5)
    records = stems.place_trees_with_dbh(
        polygon,
        sampler,
        rng=rng,
        config=stems.TreePlacementConfig(min_spacing=0.02),
    )
    dbhs = np.asarray([float(cast(float, rec["dbh"])) for rec in records], dtype=float)
    assert dbhs.size == sampler.metadata.get("sample_size")
    # Lognormal mu=2.0, sigma2=0.25 → mean ≈ 8.37
    assert 7.5 <= dbhs.mean() <= 9.5
    points = np.asarray([[rec["x"], rec["y"]] for rec in records], dtype=float)
    assert np.all((points >= 0.0) & (points <= 1.0))


def test_attributes_scale_with_dbh() -> None:
    records: list[dict[str, object]] = [
        {"dbh": 10.0, "x": 0.0, "y": 0.0},
        {"dbh": 20.0, "x": 0.1, "y": 0.1},
        {"dbh": 30.0, "x": 0.2, "y": 0.2},
    ]
    enriched = stems.attach_tree_attributes(records)
    basals = [cast(stems.TreeAttributes, rec["attributes"]).basal_area_m2 for rec in enriched]
    heights = [cast(stems.TreeAttributes, rec["attributes"]).height_m for rec in enriched]
    biomass = [cast(stems.TreeAttributes, rec["attributes"]).biomass_tonnes for rec in enriched]
    assert basals == sorted(basals)
    assert heights == sorted(heights)
    assert biomass == sorted(biomass)
    for rec in enriched:
        prov = cast(dict[str, object], rec["attributes_provenance"])
        assert prov["provenance"] == stems.DEFAULT_ATTRIBUTE_PROVENANCE


def _bootstrap_sampler(sample_size: int = 6) -> StandDBHSampler:
    entry = stands.StandBootstrapLibraryEntry(
        identifier="bootstrap-1",
        source="bootstrap.json",
        metadata={"distribution": "empirical", "sample_size": sample_size, "mode": "bootstrap"},
        dbh_vectors={
            "0": [12.0, 14.0, 16.0],
            "1": [10.0, 20.0, 22.0],
        },
    )
    assignment = stands.StandBootstrapAssignment(
        stand_id="stand-0002",
        vegetation_type="pine",
        age_class="40-60",
        area=3.5,
        bootstrap_id="bootstrap-1",
    )
    return StandDBHSampler(assignment=assignment, entry=entry)


def test_clustered_mode_with_bootstrap_sampler_is_deterministic() -> None:
    sampler = _bootstrap_sampler()
    polygon = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.5], [0.0, 1.5]])
    rng = np.random.default_rng(11)
    records = stems.place_trees_with_dbh(
        polygon,
        sampler,
        rng=rng,
        config=stems.TreePlacementConfig(mode="clustered", cluster_spread=0.08, min_spacing=0.05),
    )
    assert len(records) == sampler.metadata["sample_size"]
    dbhs = np.array([rec["dbh"] for rec in records], dtype=float)
    assert np.isclose(dbhs.mean(), np.array([12.0, 14.0, 16.0, 10.0, 20.0, 22.0]).mean(), atol=1.0)
    points = np.array([[rec["x"], rec["y"]] for rec in records], dtype=float)
    assert np.all(points[:, 0] >= 0.0) and np.all(points[:, 0] <= 2.0)
    assert np.all(points[:, 1] >= 0.0) and np.all(points[:, 1] <= 1.5)
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            assert np.linalg.norm(points[i] - points[j]) >= 0.05


def test_clustered_gallery_fixture_alignment() -> None:
    fixture = Path("tests/fixtures/synthesis/clustered_gallery.json")
    payload = json.loads(fixture.read_text())
    polygon = np.asarray(payload["polygon"], dtype=float)

    rng = np.random.default_rng(payload["analytic"]["seed"])
    anal_sampler = _analytic_sampler(sample_size=10)
    anal_records = stems.place_trees_with_dbh(
        polygon,
        anal_sampler,
        rng=rng,
        config=stems.TreePlacementConfig(
            mode="clustered",
            cluster_spread=payload["analytic"]["cluster_spread"],
            min_spacing=0.05,
        ),
    )
    anal_dbh = np.array([rec["dbh"] for rec in anal_records], dtype=float)
    assert np.isclose(anal_dbh.mean(), payload["analytic"]["mean_dbh"], atol=0.75)
    assert np.isclose(anal_dbh.std(), payload["analytic"]["std_dbh"], atol=1.0)

    rng = np.random.default_rng(payload["bootstrap"]["seed"])
    bootstrap_vectors = cast(dict[str, list[float]], payload["bootstrap"]["vectors"])
    bootstrap_size = sum(len(values) for values in bootstrap_vectors.values())
    boot_entry = stands.StandBootstrapLibraryEntry(
        identifier="bootstrap-gallery",
        source="bootstrap.json",
        metadata={
            "distribution": "empirical",
            "sample_size": bootstrap_size,
            "mode": "bootstrap",
        },
        dbh_vectors=bootstrap_vectors,
    )
    boot_assignment = stands.StandBootstrapAssignment(
        stand_id="stand-0002",
        vegetation_type="pine",
        age_class="40-60",
        area=3.5,
        bootstrap_id="bootstrap-gallery",
    )
    boot_sampler = StandDBHSampler(assignment=boot_assignment, entry=boot_entry)
    boot_records = stems.place_trees_with_dbh(
        polygon,
        boot_sampler,
        rng=rng,
        config=stems.TreePlacementConfig(
            mode="clustered",
            cluster_spread=payload["bootstrap"]["cluster_spread"],
            min_spacing=0.05,
        ),
    )
    boot_dbh = np.array([rec["dbh"] for rec in boot_records], dtype=float)
    assert np.isclose(boot_dbh.mean(), payload["bootstrap"]["mean_dbh"], atol=0.5)
    assert np.isclose(boot_dbh.std(), payload["bootstrap"]["std_dbh"], atol=0.75)


def test_load_attribute_config_from_json(tmp_path: Path) -> None:
    cfg_path = tmp_path / "attr.json"
    cfg_path.write_text(
        json.dumps(
            {
                "height_a": 2.0,
                "height_b": 0.5,
                "crown_ratio": 0.6,
                "biomass_a": 0.1,
                "biomass_b": 2.0,
                "bark_thickness_a": 0.05,
                "bark_thickness_b": 1.1,
                "provenance": "test-coeffs",
            }
        ),
        encoding="utf-8",
    )
    cfg = stems.load_attribute_config(cfg_path)
    assert cfg.provenance == "test-coeffs"
    records: list[dict[str, object]] = [{"dbh": 10.0, "x": 0.0, "y": 0.0}]
    enriched = stems.attach_tree_attributes(records, config=cfg)
    prov = cast(dict[str, object], enriched[0]["attributes_provenance"])
    assert prov["provenance"] == "test-coeffs"


def test_exporters_include_provenance_in_table(tmp_path: Path) -> None:
    polygon = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    sampler = _analytic_sampler(sample_size=2)
    records = stems.place_trees_with_dbh(
        polygon,
        sampler,
        rng=np.random.default_rng(3),
        config=stems.TreePlacementConfig(min_spacing=0.1),
    )
    enriched = stems.attach_tree_attributes(records)
    df = exporters.tree_records_to_dataframe(enriched)
    assert "attributes_provenance" in df.columns


@given(
    st.sampled_from(["poisson", "stratified", "clustered"]),
    st.integers(min_value=3, max_value=12),
    st.floats(min_value=0.0, max_value=0.2),
)
@settings(deadline=None, max_examples=40)
def test_property_based_spacing_and_bounds(mode: str, count: int, min_spacing: float) -> None:
    polygon = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    rng = np.random.default_rng(42)
    cfg = stems.TreePlacementConfig(
        mode=cast(stems.TreePlacementMode, mode),
        min_spacing=min_spacing or 0.0,
        cluster_spread=0.1,
    )
    points = stems.place_trees(polygon, count, rng=rng, config=cfg)
    assert points.shape[0] == count
    assert np.all(points[:, 0] >= 0.0) and np.all(points[:, 0] <= 2.0)
    assert np.all(points[:, 1] >= 0.0) and np.all(points[:, 1] <= 1.0)
    if min_spacing > 0:
        _assert_spacing(points, min_spacing)


def _assert_spacing(points: np.ndarray, min_spacing: float) -> None:
    for i in range(points.shape[0]):
        for j in range(i + 1, points.shape[0]):
            assert np.linalg.norm(points[i] - points[j]) >= min_spacing
