# LiDAR Observation Sampling

Nemora samples existing canopy or terrain rasters. It does not produce LiDAR
products. Use FEMIC/WS4 for catalog discovery, tile materialization, point-cloud
processing, terrain products, CHM derivation, and individual-tree detection.

## Install the optional dependency

```bash
pip install "nemora[lidar]"
```

Core Nemora remains installable without rasterio. The `nemora.lidar` module
imports without it; raster-reading functions raise `LidarDependencyError` with an
install hint when rasterio is missing.

## Read a prepared raster

```python
from pathlib import Path

from nemora.lidar import read_raster

grid = read_raster(Path("data/derived/chm.tif"))
print(grid.crs, grid.resolution, grid.width, grid.height)
```

The raster is kept in its native CRS. Nodata and masked pixels are represented
by `NaN`. Nemora does not reproject or resample implicitly.

## Sample at plot or tree locations

```python
import numpy as np

from nemora.lidar import sample_raster_at_points

heights = sample_raster_at_points(
    grid,
    np.array([1200.5, 1201.5]),
    np.array([5300.5, 5301.5]),
)
```

Points outside the raster, points on nodata pixels, and non-finite coordinates
return `NaN`.

## Summarize canopy structure

```python
from nemora.lidar import canopy_metrics, summarize_canopy_grid

metrics = summarize_canopy_grid(grid, cover_threshold_m=2.0)
print(metrics.to_dict())

# The same calculation works on a sampled vector or a synthetic height surface.
plot_metrics = canopy_metrics(heights, cover_threshold_m=2.0)
```

`cover_fraction` is the fraction of valid non-negative heights at or above the
threshold. The summary also reports valid-pixel count, mean, minimum, maximum,
p50, and p95. Negative heights and `NaN` are ignored. A summary with no valid
values raises `ValueError` rather than returning a misleading zero.

## What this does not prove

A CHM is not a tree inventory. Canopy metrics can validate height structure and
cover, but they cannot recover species, DBH, or actual stem locations. A real
tree map requires a reviewed inventory or a reviewed individual-tree-detection
product. Keep that distinction in validation reports.

## Boundary with FEMIC and WS4

| Concern | Owner |
| --- | --- |
| FAIB/FIA/PSP resolution to canonical inventory | Nemora |
| Tally materialization and synthetic stands | Nemora |
| Raster sampling and canopy metrics | Nemora |
| LidarBC discovery, tile download, materialization | FEMIC/WS4 |
| Point-cloud classification, DTM/DSM/CHM production | FEMIC/WS4 |
| Individual-tree detection/segmentation | FEMIC/WS4 |
| Harvest simulation and scenario orchestration | FHOPS/WS4 |
