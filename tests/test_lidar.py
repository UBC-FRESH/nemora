"""Tests for the optional LiDAR raster observation adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from nemora.lidar import (
    CanopyMetrics,
    LidarDependencyError,
    RasterGrid,
    canopy_metrics,
    read_raster,
    sample_raster_at_points,
    summarize_canopy_grid,
)


def _write_raster(path: Path) -> tuple[tuple[float, ...], str]:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, -9999.0],
        ],
        dtype=np.float32,
    )
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs="EPSG:3005",
        transform=transform,
        nodata=-9999.0,
    ) as destination:
        destination.write(values, 1)
    return (
        (
            float(transform.a),
            float(transform.b),
            float(transform.c),
            float(transform.d),
            float(transform.e),
            float(transform.f),
        ),
        "EPSG:3005",
    )


def test_read_raster_preserves_metadata_and_masks_nodata(tmp_path: Path) -> None:
    path = tmp_path / "chm.tif"
    expected_transform, expected_crs = _write_raster(path)

    grid = read_raster(path)

    assert grid.width == 4
    assert grid.height == 4
    assert grid.crs == expected_crs
    assert grid.transform == pytest.approx(expected_transform)
    assert grid.resolution == (1.0, 1.0)
    assert grid.nodata == -9999.0
    assert np.isnan(grid.values[3, 3])
    assert grid.values[0, 0] == pytest.approx(1.0)


def test_sample_raster_at_points_handles_bounds_and_nodata(tmp_path: Path) -> None:
    path = tmp_path / "chm.tif"
    _write_raster(path)
    grid = read_raster(path)

    sampled = sample_raster_at_points(
        grid,
        np.array([0.5, 3.5, 3.5, 4.5]),
        np.array([3.5, 3.5, 0.5, 3.5]),
    )

    assert sampled[0] == pytest.approx(1.0)
    assert sampled[1] == pytest.approx(4.0)
    assert np.isnan(sampled[2])
    assert np.isnan(sampled[3])


def test_sample_raster_at_points_accepts_scalars(tmp_path: Path) -> None:
    path = tmp_path / "chm.tif"
    _write_raster(path)
    grid = read_raster(path)

    sampled = sample_raster_at_points(grid, 0.5, 3.5)

    assert sampled.shape == ()
    assert float(sampled) == pytest.approx(1.0)


def test_sample_raster_at_points_rejects_mismatched_shapes(tmp_path: Path) -> None:
    path = tmp_path / "chm.tif"
    _write_raster(path)
    grid = read_raster(path)

    with pytest.raises(ValueError, match="same shape"):
        sample_raster_at_points(grid, np.array([0.5, 1.5]), np.array([3.5]))


def test_canopy_metrics_ignores_invalid_values() -> None:
    metrics = canopy_metrics([0.0, 1.0, 2.0, 3.0, 4.0, np.nan, -1.0])

    assert metrics == CanopyMetrics(
        valid_pixels=5,
        cover_fraction=0.6,
        cover_threshold_m=2.0,
        height_mean_m=2.0,
        height_min_m=0.0,
        height_max_m=4.0,
        height_p50_m=2.0,
        height_p95_m=3.8,
    )


def test_canopy_metrics_requires_valid_values() -> None:
    with pytest.raises(ValueError, match="at least one"):
        canopy_metrics([np.nan, -1.0])


def test_summarize_canopy_grid_uses_valid_cells(tmp_path: Path) -> None:
    path = tmp_path / "chm.tif"
    _write_raster(path)
    grid = read_raster(path)

    metrics = summarize_canopy_grid(grid)

    assert metrics.valid_pixels == 15
    assert metrics.cover_fraction == pytest.approx(14 / 15)
    assert metrics.height_max_m == pytest.approx(15.0)


def test_missing_rasterio_dependency_raises_actionable_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setitem(sys.modules, "rasterio", None)

    with pytest.raises(LidarDependencyError, match="optional"):
        read_raster(tmp_path / "missing.tif")


def test_raster_grid_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape"):
        RasterGrid(
            values=np.zeros((2, 2)),
            transform=(1.0, 0.0, 0.0, 0.0, -1.0, 2.0),
            crs=None,
            nodata=None,
            resolution=(1.0, 1.0),
            width=3,
            height=2,
        )
