# Phase 7 LiDAR coverage and materialization

Date: 2026-10-30

Phase: P7

Task: P7.3

Status: complete with partial coverage.

## Correction

The initial audit checked the local FEMIC instances and reported that no prepared
Gavin Lake DEM/CHM existed. That was too narrow. A query of the public LidarBC
catalogue through the FEMIC catalog/materialization functions found 2024 1 m DEM
and DSM tiles covering the Gavin Lake area.

The previous deferral is withdrawn. The replacement evidence is below.

## Discovery and materialization

- Source: LidarBC Open LiDAR Data Portal, OGL-BC.
- Catalogue query: `femic.lidar.catalog.discover_lidarbc_tiles` over a 25 km
  Gavin Lake AOI.
- Selected tile IDs: 16 DEM/DSM pairs covering 16 of the 20 selected plots and
  the synthetic-stand centre.
- Materialized: 32 tiles through
  `femic.lidar.materialization.materialize_lidar_selection` (16 DEM + 16 DSM).
- Native product: 1 m DEM/DSM, acquisition year 2024, native projected CRS
  (EPSG:6653 for the checked tiles).
- No reprojection or resampling was performed for the source products.

## Products

- CHM: `femic.lidar.canopy.build_canopy_products` (DSM − DEM, aligned native
  grids, negative heights clamped).
- Slope: `femic.lidar.terrain.build_terrain_products` (Horn 3×3 percent rise).
- Sampling: `nemora.lidar` at 5×5 grids with 10 m spacing around each plot
  centre; the synthetic stand used a 21×21 grid over 100 m × 100 m.

## Coverage

- 16 of 20 plots have a raster hit.
- 15 of those 16 have valid canopy and slope values; one plot falls on a nodata
  edge.
- Four plots have no LidarBC tile-index hit in the queried 25 km AOI.

Missing raster hits:

- `4032539-PSP1_v1_p1`
- `4032404-PSP1_v1_p1`
- `4032374-PSP1_v1_p1`
- `4031967-PSP1_v1_p1`

The plot `4032469-PSP1_v1_p1` has a raster hit but no valid sampled pixels.

## Results

Across the 15 valid plot samples:

- Canopy cover (≥2 m): mean **0.664**.
- Mean canopy height: mean **12.05 m**.
- Canopy p95 height: mean **25.95 m**.
- Slope: mean **17.83%**.

Synthetic-stand centre (100 m × 100 m grid):

- Canopy cover (≥2 m): **0.879**.
- Mean canopy height: **9.82 m**.
- Canopy p95 height: **17.29 m**.
- Slope: mean **17.23%**; p95 **31.75%**.

The synthetic stand has higher cover but lower mean/p95 height than the sampled
plot contexts. This is context evidence, not a stand-level accuracy claim: the
plot samples are small grids around points, and the synthetic stand is a
statistical reconstruction.

## Provenance

- Full per-plot and per-tile metrics: `notes/reports/phase7_lidar_metrics.json`.
- No plot coordinates are written to the metrics JSON.
- The materialized tiles and derived CHMs are local scratch artifacts, not
  committed.
- The source product is OGL-BC LidarBC 2024 1 m DEM/DSM.

## Limits

- Four selected plots have no LidarBC index coverage in the queried AOI.
- One covered plot has no valid sampled pixels.
- CHM is a DSM−DEM product, not a field canopy measurement.
- Plot context uses 40 m × 40 m sample grids, not full plot polygons.
- No individual-tree detection, species inference, or operational claim is
  made.

## Reproduction

```python
from femic.lidar.catalog import discover_lidarbc_tiles, write_lidar_selection_manifest
from femic.lidar.materialization import materialize_lidar_selection
from femic.lidar.canopy import build_canopy_products
from femic.lidar.terrain import build_terrain_products
```

Use the public LidarBC catalogue with the Gavin Lake AOI in EPSG:3005, filter to
the selected DEM/DSM tile pairs, materialize them into a scratch root, derive
CHM/slope for each aligned pair, then sample with `nemora.lidar` as described
above. The metrics JSON is the checked-in evidence.
