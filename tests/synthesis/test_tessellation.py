from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from nemora.synthesis import tessellation


def test_uniform_seed_generation_respects_aspect_ratio() -> None:
    cfg = tessellation.VoronoiSeedConfig(
        count=10,
        aspect_ratio=2.0,
        rng=np.random.default_rng(1234),
    )
    result = tessellation.generate_seed_points(cfg)
    assert result.points.shape == (10, 2)
    assert np.all(result.points[:, 0] >= 0)
    assert np.all(result.points[:, 0] <= 2.0 + 1e-9)
    assert np.all(result.points[:, 1] >= 0)
    assert np.all(result.points[:, 1] <= 1.0 + 1e-9)


def test_cluster_only_mix_is_supported() -> None:
    mix = tessellation.PointProcessMix(uniform=0.0, cluster=1.0)
    cfg = tessellation.VoronoiSeedConfig(
        count=7,
        mix=mix,
        rng=np.random.default_rng(9876),
    )
    result = tessellation.generate_seed_points(cfg)
    assert result.points.shape == (7, 2)
    assert result.process_counts["cluster"] == 7
    assert result.process_counts["uniform"] == 0


def test_editing_knobs_respected() -> None:
    cfg = tessellation.VoronoiSeedConfig(
        count=20,
        mix=tessellation.PointProcessMix(uniform=1.0),
        edit=tessellation.VoronoiEditConfig(hole_fraction=0.1, merge_fraction=0.2),
        rng=np.random.default_rng(4321),
    )
    result = tessellation.generate_seed_points(cfg)
    assert result.points.shape[0] == 20
    hole_expected = int(round(0.1 * 20))
    merge_expected = int(round(0.2 * 20))
    assert result.hole_points.shape[0] == hole_expected
    assert result.merge_pairs.shape[0] == merge_expected
    metadata = result.metadata()
    assert cast(int, metadata["initial_seed_count"]) == 20 + hole_expected + merge_expected
    edit = cast(dict[str, object], metadata["edit"])
    assert cast(int, edit["hole_count"]) == hole_expected
    assert cast(int, edit["merge_count"]) == merge_expected
    metrics = cast(dict[str, object], metadata["metrics"])
    assert metrics["polygon_count"] == 20


def test_invalid_edit_fraction_rejected() -> None:
    with pytest.raises(ValueError):
        tessellation.VoronoiEditConfig(hole_fraction=0.6, merge_fraction=0.5)


def test_voronoi_metrics_cover_area_and_vertex_stats() -> None:
    cfg = tessellation.VoronoiSeedConfig(
        count=16,
        aspect_ratio=1.5,
        mix=tessellation.PointProcessMix(uniform=1.0),
        rng=np.random.default_rng(2468),
    )
    result = tessellation.generate_seed_points(cfg)
    metrics = result.metrics
    assert metrics.polygon_count == 16
    assert metrics.area_mean > 0
    assert 0 <= metrics.area_cv < 1
    assert metrics.vertex_degree_mean > 0
    # Ensure polygons respect bounds.
    assert len(result.polygons) == 16
    for poly in result.polygons:
        if poly.size == 0:
            continue
        assert np.all(poly[:, 0] >= -1e-9)
        assert np.all(poly[:, 0] <= cfg.aspect_ratio + 1e-9)
        assert np.all(poly[:, 1] >= -1e-9)
        assert np.all(poly[:, 1] <= 1.0 + 1e-9)


def test_mask_geometry_clips_polygons() -> None:
    mask_polygon = np.array(
        [
            [0.2, 0.2],
            [0.8, 0.2],
            [0.8, 0.8],
            [0.2, 0.8],
        ],
        dtype=float,
    )
    mask = tessellation.MaskGeometry(polygons=[mask_polygon], name="test-mask")
    cfg = tessellation.VoronoiSeedConfig(
        count=8,
        aspect_ratio=1.0,
        mask=mask,
        rng=np.random.default_rng(13579),
    )
    result = tessellation.generate_seed_points(cfg)
    metadata = result.metadata()
    mask_meta = cast(dict[str, object], metadata["mask"])
    assert mask_meta["name"] == "test-mask"
    for poly in result.polygons:
        if poly.size == 0:
            continue
        assert np.all(poly[:, 0] >= 0.2 - 1e-9)
        assert np.all(poly[:, 0] <= 0.8 + 1e-9)
        assert np.all(poly[:, 1] >= 0.2 - 1e-9)
        assert np.all(poly[:, 1] <= 0.8 + 1e-9)
