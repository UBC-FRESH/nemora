# Phase 7 LiDAR coverage audit

Date: 2026-10-30

Phase: P7

Task: P7.3

Status: deferred. No prepared Gavin Lake DEM/CHM product is available in the
accessible public inputs.

## What was checked

- The Gavin Lake FEMIC instance contains legacy spatial/stand-model inputs but no
  prepared DEM or CHM raster.
- The Malcolm Knapp FEMIC instance contains legacy model/configuration inputs
  but no prepared DEM or CHM raster.
- The nearest prepared AFRF canopy product is the Knifecreek pilot
  `pilot_093a001_2_4_3/chm.tif` (EPSG:6653, 1 m). Its WGS84 bounds are
  approximately `-121.8506, 52.0371` to `-121.8245, 52.0504`, about 47 km south
  of Gavin Lake. It does not cover the selected Gavin Lake plots.
- The FHOPS tree-scale demo contains a synthetic Gavin DEM. It is not a
  LidarBC/FEMIC product and was not used as validation evidence.

## Decision

Do not download or reprocess raw LiDAR in P7.3. Do not substitute:

- the Knifecreek CHM, which is a different forest and footprint;
- the synthetic demo DEM, which has no LidarBC provenance;
- a reprojected/resampled raster, which would hide the coverage gap.

P7.3 remains deferred until a prepared Gavin Lake DEM/CHM product and its
provenance are supplied. The existing `nemora.lidar` contract tests
(`tests/test_lidar.py`) still cover raster sampling, nodata, bounds, and canopy
metrics with synthetic fixtures.

## What the final report can say

The Phase 7 validation report can report plot and stand results with the LiDAR
section explicitly marked unavailable. It must not claim terrain or canopy
validation for Gavin Lake.

## Unblocking options

1. Supply a prepared Gavin Lake DEM/CHM bundle with CRS, resolution, licence, and
   source-product provenance.
2. Approve a separate data-preparation task outside Nemora to build such a
   bundle through FEMIC/LidarBC.
3. Run a method-only demonstration on the Knifecreek CHM, clearly labelled as a
   different forest and not used in the Gavin validation.
