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
    primary = cast(dict[str, object], mask_meta["primary"])
    assert primary["name"] == "test-mask"
    for poly in result.polygons:
        if poly.size == 0:
            continue
        assert np.all(poly[:, 0] >= 0.2 - 1e-9)
        assert np.all(poly[:, 0] <= 0.8 + 1e-9)
        assert np.all(poly[:, 1] >= 0.2 - 1e-9)
        assert np.all(poly[:, 1] <= 0.8 + 1e-9)


def test_hex_layout_generates_deterministic_points() -> None:
    cfg = tessellation.VoronoiSeedConfig(
        count=12,
        layout=tessellation.SeedLayoutConfig(mode=tessellation.SeedLayoutMode.HEX),
    )
    result_a = tessellation.generate_seed_points(cfg)
    result_b = tessellation.generate_seed_points(
        tessellation.VoronoiSeedConfig(
            count=12,
            layout=tessellation.SeedLayoutConfig(mode=tessellation.SeedLayoutMode.HEX),
        )
    )
    assert np.allclose(result_a.points, result_b.points)
    metadata = result_a.metadata()
    layout_meta = cast(dict[str, object], metadata["layout"])
    assert layout_meta["mode"] == "hex"
    assert "layout_hex" in result_a.process_counts
    assert result_a.process_counts["layout_hex"] == 12


def test_imported_layout_uses_provided_points() -> None:
    provided = np.array(
        [
            [0.1, 0.1],
            [0.2, 0.2],
            [0.3, 0.3],
            [0.4, 0.4],
        ],
        dtype=float,
    )
    cfg = tessellation.VoronoiSeedConfig(
        count=3,
        layout=tessellation.SeedLayoutConfig(
            mode=tessellation.SeedLayoutMode.IMPORTED,
            points=provided,
            source="fixture",
        ),
    )
    result = tessellation.generate_seed_points(cfg)
    assert np.allclose(result.points, provided[:3])
    metadata = result.metadata()
    layout_meta = cast(dict[str, object], metadata["layout"])
    assert layout_meta["mode"] == "imported"
    assert layout_meta["points_provided"] == provided.shape[0]
    with pytest.raises(ValueError):
        tessellation.generate_seed_points(
            tessellation.VoronoiSeedConfig(
                count=5,
                layout=tessellation.SeedLayoutConfig(
                    mode=tessellation.SeedLayoutMode.IMPORTED,
                    points=provided,
                ),
            )
        )


def test_exclude_mask_removes_polygons() -> None:
    clip_mask = tessellation.MaskGeometry(
        polygons=[
            np.array(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
                dtype=float,
            )
        ],
        name="clip",
        mode=tessellation.MaskMode.CLIP,
    )
    exclude_mask = tessellation.MaskGeometry(
        polygons=[
            np.array(
                [
                    [0.5, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.5, 1.0],
                ],
                dtype=float,
            )
        ],
        name="exclude",
        mode=tessellation.MaskMode.EXCLUDE,
    )
    layout_points = np.array([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    cfg = tessellation.VoronoiSeedConfig(
        count=2,
        layout=tessellation.SeedLayoutConfig(
            mode=tessellation.SeedLayoutMode.IMPORTED,
            points=layout_points,
        ),
        mask=clip_mask,
        mask_overlays=[exclude_mask],
    )
    result = tessellation.generate_seed_points(cfg)
    assert result.polygons[1].size == 0


def test_raster_mask_filters_polygons() -> None:
    layout_points = np.array([[0.25, 0.75], [0.75, 0.25]], dtype=float)
    raster = tessellation.RasterMask(
        values=np.array([[1.0, 1.0], [0.0, 0.0]], dtype=float),
        threshold=0.5,
        mode=tessellation.RasterMode.KEEP,
        name="slope",
    )
    cfg = tessellation.VoronoiSeedConfig(
        count=2,
        layout=tessellation.SeedLayoutConfig(
            mode=tessellation.SeedLayoutMode.IMPORTED,
            points=layout_points,
        ),
        raster_masks=[raster],
    )
    result = tessellation.generate_seed_points(cfg)
    assert result.polygons[0].size > 0
    assert result.polygons[1].size == 0


def test_geojson_layout_centroids_respected() -> None:
    square = np.array([[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]], dtype=float)
    triangle = np.array([[0.8, 0.8], [0.9, 0.6], [0.7, 0.6]], dtype=float)
    cfg = tessellation.VoronoiSeedConfig(
        count=2,
        layout=tessellation.SeedLayoutConfig(
            mode=tessellation.SeedLayoutMode.GEOJSON,
            geojson_polygons=[square, triangle],
        ),
    )
    result = tessellation.generate_seed_points(cfg)
    expected = np.array([[0.1, 0.1], [triangle[:, 0].mean(), triangle[:, 1].mean()]])
    assert np.allclose(result.points, expected)
