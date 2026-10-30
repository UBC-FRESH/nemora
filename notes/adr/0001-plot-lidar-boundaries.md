# ADR 0001: Plot and LiDAR module boundaries

Status: accepted

Date: 2026-10-30

Phase: P6

## Context

Nemora already owns the shared `nemora.inventory.v1` schema, plot selection and
DBH tally aggregation, distribution fitting, sampling, and synthetic stand
generation. The next validation step needs two more things:

1. a reusable way to resolve source-specific plot tables into the canonical
   inventory release and to materialize tally classes into individual trees;
2. a way to compare synthesized or observed stands with existing canopy and
   terrain rasters.

LiDAR production already exists in FEMIC and is consumed by WS4/FHOPS. That
code includes LidarBC catalog discovery, tile selection and download,
materialization manifests, point-cloud rasterization, terrain products, CHM
derivation, and individual-tree detection. Moving those concerns into Nemora
would add heavy dependencies and mix a producer with an analysis library.

## Decision

Nemora owns the generic inventory and observation side of the boundary:

- source-to-`nemora.inventory.v1` plot resolution;
- tally-class materialization into individual tree rows;
- distribution fitting, sampling, and synthetic stand generation;
- sampling existing rasters at plot or tree locations;
- canopy and terrain metric summaries over plots, stands, or synthetic trees;
- optional adapters that turn external LiDAR products into Nemora observations;
- uncertainty and validation reports that compare observed and synthetic
  summaries.

FEMIC and WS4 keep the LiDAR production and workflow side:

- source catalog discovery and tile selection;
- download, checksum, and materialization manifests;
- point-cloud classification and rasterization;
- DTM/DSM/CHM and terrain-product derivation;
- individual-tree detection/segmentation;
- scenario registration and harvest-workflow orchestration.

FHOPS remains a consumer. It may use Nemora outputs, but Nemora does not import
FHOPS and FHOPS does not become a Nemora dependency.

## Dependency boundary

- Core Nemora must remain importable without geospatial or LiDAR libraries.
- `nemora.lidar` is an optional module requiring the `lidar` extra
  (`rasterio`).
- Missing optional dependencies raise `LidarDependencyError` with an install
  instruction.
- Raster inputs are consumed in their native CRS. Nemora samples and summarizes;
  it does not reproject or resample implicitly.
- Polygon support and vector IO remain outside the first phase of
  `nemora.lidar`; callers can pass point locations or mask rasters themselves.

## Consequences

- A single canonical inventory contract still feeds fitting, sampling,
  synthesis, and validation.
- The same canopy metrics can be computed for a real CHM and for a synthetic
  landscape without moving the producer into Nemora.
- LiDAR production can evolve independently of Nemora.
- A future vector adapter can be added without changing the raster contract.
- Nemora will not be able to claim it produces a "real" LiDAR product or a real
  tree map from a CHM. Documentation and reports must preserve that limit.

## Rejected alternatives

- Move all FEMIC LiDAR code into Nemora. Rejected: it would couple heavy
  production dependencies and workflow state to a distribution/simulation
  package.
- Keep plot resolution in WS4/FHOPS. Rejected: source-to-schema normalization is
  a Nemora ingest concern and is needed by multiple consumers.
- Put raster sampling in WS4/FHOPS first. Rejected: the sampling contract is
  generic and belongs beside the synthetic stands and inventory schema it
  validates.
