"""Raster sampling and canopy metrics for existing LiDAR products.

The module deliberately consumes prepared rasters in their native CRS. It does
not download tiles, classify point clouds, derive a CHM, reproject, or resample.
Install the optional ``lidar`` extra to read GeoTIFFs with rasterio.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "CanopyMetrics",
    "LidarDependencyError",
    "RasterGrid",
    "canopy_metrics",
    "read_raster",
    "sample_raster_at_points",
    "summarize_canopy_grid",
]

_RASTERIO_INSTALL_HINT = (
    "Reading rasters requires rasterio. Install Nemora with the optional "
    "`lidar` extra, for example `pip install 'nemora[lidar]'`."
)


class LidarDependencyError(ImportError):
    """Raised when an optional LiDAR dependency is not installed."""


@dataclass(frozen=True, slots=True)
class RasterGrid:
    """A single-band raster in its native CRS.

    Parameters
    ----------
    values:
        Two-dimensional array with ``height`` rows and ``width`` columns.
        Nodata and masked pixels are represented by ``NaN``.
    transform:
        Affine coefficients ``(a, b, c, d, e, f)`` in the rasterio convention:
        ``x = a * col + b * row + c`` and ``y = d * col + e * row + f``.
    crs:
        Native CRS string, or ``None`` when the source has no CRS.
    nodata:
        Source nodata value, when declared.
    resolution:
        Pixel size as ``(x_resolution, y_resolution)`` in CRS units.
    width:
        Number of columns.
    height:
        Number of rows.
    """

    values: np.ndarray
    transform: tuple[float, float, float, float, float, float]
    crs: str | None
    nodata: float | None
    resolution: tuple[float, float]
    width: int
    height: int

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 2:
            raise ValueError("RasterGrid.values must be two-dimensional")
        if len(self.transform) != 6:
            raise ValueError("RasterGrid.transform must contain six affine coefficients")
        if values.shape != (self.height, self.width):
            raise ValueError(
                "RasterGrid shape does not match height/width: "
                f"{values.shape} != ({self.height}, {self.width})"
            )
        if self.width < 1 or self.height < 1:
            raise ValueError("RasterGrid width and height must be positive")
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class CanopyMetrics:
    """Summary of canopy-height values from a valid raster footprint."""

    valid_pixels: int
    cover_fraction: float
    cover_threshold_m: float
    height_mean_m: float
    height_min_m: float
    height_max_m: float
    height_p50_m: float
    height_p95_m: float

    def to_dict(self) -> dict[str, float | int]:
        """Return the metrics as a JSON-serializable mapping."""
        return {
            "valid_pixels": self.valid_pixels,
            "cover_fraction": self.cover_fraction,
            "cover_threshold_m": self.cover_threshold_m,
            "height_mean_m": self.height_mean_m,
            "height_min_m": self.height_min_m,
            "height_max_m": self.height_max_m,
            "height_p50_m": self.height_p50_m,
            "height_p95_m": self.height_p95_m,
        }


def read_raster(path: str | Path, *, band: int = 1) -> RasterGrid:
    """Read one band of a raster as a :class:`RasterGrid`.

    Parameters
    ----------
    path:
        Path to a raster readable by rasterio.
    band:
        One-based band index.

    Raises
    ------
    LidarDependencyError
        If rasterio is not installed.
    ValueError
        If the band index is invalid or the raster has no two-dimensional band.
    """
    try:
        import rasterio  # type: ignore[import-untyped]  # optional dependency has no py.typed marker
    except ImportError as exc:  # pragma: no cover - exercised by dependency-boundary test
        raise LidarDependencyError(_RASTERIO_INSTALL_HINT) from exc

    if band < 1:
        raise ValueError("band must be a positive one-based index")
    with rasterio.open(Path(path)) as source:
        if band > source.count:
            raise ValueError(f"band {band} is out of range for a {source.count}-band raster")
        data = source.read(band, masked=True)
        values = np.asarray(data.filled(np.nan), dtype=float)
        transform = source.transform
        nodata = None if source.nodata is None else float(source.nodata)
        if nodata is not None:
            values = np.where(values == nodata, np.nan, values)
        x_resolution, y_resolution = (float(value) for value in source.res)
        crs = source.crs.to_string() if source.crs is not None else None
        return RasterGrid(
            values=values,
            transform=(
                float(transform.a),
                float(transform.b),
                float(transform.c),
                float(transform.d),
                float(transform.e),
                float(transform.f),
            ),
            crs=crs,
            nodata=nodata,
            resolution=(x_resolution, y_resolution),
            width=source.width,
            height=source.height,
        )


def sample_raster_at_points(
    grid: RasterGrid,
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
) -> np.ndarray:
    """Sample a raster at native-CRS point locations.

    Points are assigned to the pixel containing them. Points outside the grid,
    pixels with ``NaN``, and non-finite coordinates return ``NaN``.

    Parameters
    ----------
    grid:
        Raster to sample.
    x, y:
        Coordinate arrays in the raster's native CRS. They must have the same
        shape.

    Returns
    -------
    numpy.ndarray
        An array with the same shape as ``x`` and ``y``.
    """
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.shape != y_values.shape:
        raise ValueError("x and y must have the same shape")

    a, b, c, d, e, f = grid.transform
    determinant = a * e - b * d
    if abs(determinant) < 1e-15:
        raise ValueError("Raster transform is not invertible")

    x_offset = x_values - c
    y_offset = y_values - f
    columns = (e * x_offset - b * y_offset) / determinant
    rows = (-d * x_offset + a * y_offset) / determinant
    column_index = np.floor(columns)
    row_index = np.floor(rows)

    valid = (
        np.isfinite(column_index)
        & np.isfinite(row_index)
        & (column_index >= 0)
        & (column_index < grid.width)
        & (row_index >= 0)
        & (row_index < grid.height)
    )
    result = np.full(x_values.shape, np.nan, dtype=float)
    if np.any(valid):
        result[valid] = grid.values[
            row_index[valid].astype(int),
            column_index[valid].astype(int),
        ]
    result[~np.isfinite(result)] = np.nan
    return result


def canopy_metrics(
    values: np.ndarray | Sequence[float],
    *,
    cover_threshold_m: float = 2.0,
) -> CanopyMetrics:
    """Summarize canopy-height values.

    Finite, non-negative values are treated as valid canopy heights. Negative
    values and ``NaN`` are ignored. Cover is the fraction of valid pixels at or
    above ``cover_threshold_m``.

    Parameters
    ----------
    values:
        Scalar array or sequence of canopy heights in metres.
    cover_threshold_m:
        Height threshold for the cover fraction.

    Returns
    -------
    CanopyMetrics
        Valid-pixel count, cover fraction, and height summary.

    Raises
    ------
    ValueError
        If no finite, non-negative values are available.
    """
    heights = np.asarray(values, dtype=float).ravel()
    valid = np.isfinite(heights) & (heights >= 0.0)
    heights = heights[valid]
    if heights.size == 0:
        raise ValueError("canopy metrics need at least one finite, non-negative value")
    if cover_threshold_m < 0.0 or not np.isfinite(cover_threshold_m):
        raise ValueError("cover_threshold_m must be finite and non-negative")

    p50, p95 = np.percentile(heights, [50.0, 95.0])
    return CanopyMetrics(
        valid_pixels=int(heights.size),
        cover_fraction=float(np.count_nonzero(heights >= cover_threshold_m) / heights.size),
        cover_threshold_m=float(cover_threshold_m),
        height_mean_m=float(np.mean(heights)),
        height_min_m=float(np.min(heights)),
        height_max_m=float(np.max(heights)),
        height_p50_m=float(p50),
        height_p95_m=float(p95),
    )


def summarize_canopy_grid(
    grid: RasterGrid,
    *,
    cover_threshold_m: float = 2.0,
) -> CanopyMetrics:
    """Compute :func:`canopy_metrics` over all valid cells in a raster."""
    return canopy_metrics(grid.values, cover_threshold_m=cover_threshold_m)
