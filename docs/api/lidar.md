# `nemora.lidar`

`nemora.lidar` is an optional observation layer for prepared canopy or terrain
rasters. It samples rasters in their native CRS and computes canopy summaries for
inventory validation and synthetic-stand comparison.

It does **not** discover LiDAR tiles, download or materialize products, classify
point clouds, derive a CHM/DTM, or detect individual trees. Those producer
responsibilities remain in FEMIC/WS4. See
[ADR 0001](https://github.com/UBC-FRESH/nemora/blob/main/notes/adr/0001-plot-lidar-boundaries.md)
for the boundary.

Install the optional dependency:

```bash
pip install "nemora[lidar]"
```

Importing the package works without rasterio. Calling `read_raster` without the
extra raises `LidarDependencyError` with installation guidance.

## API

```{automodule} nemora.lidar
:members:
:undoc-members:
:show-inheritance:
```

```{automodule} nemora.lidar.raster
:members:
:undoc-members:
:show-inheritance:
```

```{seealso}
- [`docs/howto/lidar.md`](../howto/lidar.md) for a sampling walkthrough.
- [`docs/howto/inventory_schema.md`](../howto/inventory_schema.md) for the
  inventory release that plot samples can be joined to.
```
