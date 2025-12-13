from __future__ import annotations

import numpy as np

from nemora.synthesis import stands, stems
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
        config=stems.TreePlacementConfig(min_spacing=0.2),
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
