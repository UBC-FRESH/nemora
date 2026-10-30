# Phase 8 LidarBC coverage expansion

Date: 2026-10-30

Phase: P8

Task: P8.2

Status: complete with four residual coverage gaps.

## What changed

P7 left 15/20 plots with valid canopy/slope samples, four plots with no
tile-index hit, and one covered plot with no valid sampled pixels.

P8.2 re-queried the public LidarBC catalogue at 1 km, 3 km, 10 km, and 25 km
around each affected plot.

- `4032469-PSP1_v1_p1` was resolved using the alternate tile
  `093a031_3_2_1`, which was already materialized in P7. The P7 sample had used
  an adjacent tile with nodata at the plot.
- The four other plots have no tile whose indexed footprint contains the point,
  even in a 25 km query. They remain outside available LidarBC coverage:
  - `4032539-PSP1_v1_p1`
  - `4032404-PSP1_v1_p1`
  - `4032374-PSP1_v1_p1`
  - `4031967-PSP1_v1_p1`

No new tile download was required for the resolved plot. No synthetic or
unrelated raster was substituted for the remaining four.

## Result

- Plots with a valid canopy/slope sample: **16/20**.
- Residual missing plots: **4/20**.
- Mean plot canopy cover (≥2 m): **0.683**.
- Mean canopy height: **12.59 m**.
- Mean canopy p95 height: **26.07 m**.
- Mean slope: **17.91%**.

Per-plot values and the resolved tile are in
`notes/reports/phase8_lidar_metrics.json`.

## Limits

- The four missing plots are a coverage limitation, not a model failure.
- The sample footprint is a 5×5 grid at 10 m spacing, not a full plot polygon.
- CHM is DSM−DEM, not a field canopy measurement.
- No reprojection, resampling, ITD, species inference, or operational claim.

## Reproduction

Query the public LidarBC catalogue around each plot in EPSG:3005, filter to
tiles whose footprint contains the plot, materialize DEM/DSM pairs through
FEMIC, derive CHM/slope, and sample with `nemora.lidar`. The P7 materialized
tiles already contain the alternate `093a031_3_2_1` product used here.
