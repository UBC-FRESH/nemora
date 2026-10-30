"""Optional LiDAR observation and raster metric helpers.

This package does not produce LiDAR products. It samples existing rasters and
summarizes canopy structure for inventory validation and synthetic-stand
comparison. See ``notes/adr/0001-plot-lidar-boundaries.md`` for the boundary.
"""

from __future__ import annotations

from .raster import (
    CanopyMetrics,
    LidarDependencyError,
    RasterGrid,
    canopy_metrics,
    read_raster,
    sample_raster_at_points,
    summarize_canopy_grid,
)

__all__ = [
    "CanopyMetrics",
    "LidarDependencyError",
    "RasterGrid",
    "canopy_metrics",
    "read_raster",
    "sample_raster_at_points",
    "summarize_canopy_grid",
]
