# Phase 7 validation report: Gavin Lake

Date: 2026-10-30

Phase: P7 Research-forest plot and LiDAR validation

Status: first-pass validation complete; LiDAR context deferred for missing prepared data.

## Scope

This report covers:

- selection of public FAIB PSP plots for Gavin Lake;
- resolution of plot and live tree detail;
- a training/hold-out stand validation with a seeded synthetic stand;
- an audited gap in prepared Gavin Lake DEM/CHM coverage.

It does not claim a real tree map, a field-calibrated stand, or an operational
harvest prescription.

## Inputs and provenance

- Source: BC FAIB PSP public compilation.
- Licence: OGL-BC.
- Source URI: `ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/psp/`.
- Input tables: `faib_header.csv`, `faib_plot_header.csv`,
  `faib_sample_byvisit.csv`, `faib_tree_detail.csv`.
- Selection manifests:
  - `notes/reports/phase7_plot_selection_gavin.json`
  - `notes/reports/phase7_plot_selection_mkrf.json`
- Stand validation result: `notes/reports/phase7_stand_validation_gavin.json`.
- Selection and split seeds: 20261030.
- Stand seed: 20261030.

The selection manifests record input sizes/checksums, filters, counts, and a
sorted plot-ID digest. They deliberately omit plot coordinates.

## Plot selection

| Target | Radius | Plots | Train | Hold-out | BEC | Projected age |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Gavin Lake | 25 km | 20 | 15 | 5 | 18 `SBSdw1`, 2 `ICHmk3` | 74–152 |
| Malcolm Knapp | 50 km | 6 | 4 | 2 | 6 `CWHdm` | 78–98 |

Gavin Lake is the first validation target. Malcolm Knapp is an exploratory
fallback because six plots are too few for a meaningful independent hold-out
comparison.

## Tree detail

The Gavin Lake selection resolves to 2,870 live trees across all 20 plots.

- Trees per plot: 74–331; median 105.5.
- Species: 12 codes; dominant `FD`, `AT`, `SW`, `PL`, and `BL`.
- DBH: 2.0–56.1 cm.
- Measured height: 1.9–41.2 m.
- Expansion tally: 40,920.48.

The training/hold-out releases validate under `nemora.inventory.v1`. They are
written to an untracked `build/phase7/` directory because they contain plot
coordinates.

## Stand validation

Training: 15 plots / 2,086 trees. Hold-out: 5 plots / 784 trees.

The synthetic stand uses:

- a grouped Weibull DBH fit weighted by `PHF_TREE`;
- a log-log height–DBH fit on training measured trees;
- training species proportions;
- one hectare-equivalent at the training mean stems/ha;
- 1,978 generated trees.

| Metric | Training | Hold-out | Synthetic | Relative error |
| --- | ---: | ---: | ---: | ---: |
| Stems/ha | 1,977.84 | 2,250.57 | 1,978.00 | -12.11% |
| Basal area (m²/ha) | 48.15 | 40.72 | 52.43 | +28.76% |
| QMD (cm) | 20.30 | 16.01 | 18.37 | +14.73% |
| Mean DBH (cm) | 17.85 | 14.36 | 13.54 | -5.74% |
| Mean height (m) | 17.80 | 14.58 | 14.24 | -2.34% |

DBH distribution and composition:

- Weighted DBH KS distance: **0.201**.
- Species total-variation distance: **0.217**.

The first pass is a negative result. Basal area and QMD are high relative to the
hold-out plots; mean DBH and mean height are slightly low. One pooled Weibull
does not capture the species- and plot-level structure.

## Uncertainty

Plot-resampling bootstrap, 100 successful replicates:

| Synthetic metric | 5th percentile | Median | 95th percentile |
| --- | ---: | ---: | ---: |
| Stems/ha | 1,422.75 | 2,030.50 | 2,492.15 |
| Basal area (m²/ha) | 23.52 | 52.06 | 74.21 |
| QMD (cm) | 11.26 | 18.07 | 25.43 |
| Mean DBH (cm) | 8.81 | 13.22 | 19.97 |
| Mean height (m) | 10.27 | 13.72 | 19.39 |

The intervals are wide. The point estimates should not be read as precise stand
estimates.

## LiDAR and terrain context

A public LidarBC catalogue query through the FEMIC discovery/materialization
functions found 2024 1 m DEM and DSM coverage for Gavin Lake. Sixteen DEM/DSM
pairs were materialized and used to derive CHM and slope products. No source
reprojection or resampling was performed.

Coverage and results:

- 16 of 20 selected plots have a raster hit; 15 have valid canopy and slope
  values.
- Four selected plots have no LidarBC tile-index hit in the 25 km AOI.
- One covered plot has no valid sampled pixels.

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

This is context evidence, not a stand-level accuracy claim. The plot footprints
are 40 m × 40 m sample grids, and the synthetic stand is a statistical
reconstruction. See `notes/reports/phase7_lidar_coverage.md` and
`notes/reports/phase7_lidar_metrics.json`.

## Observed, fitted, assumed, synthetic

- **Observed:** FAIB plot locations, projected ages, live tree DBH/height,
  expansion tallies, species codes.
- **Fitted:** grouped Weibull DBH parameters and log-log height relation.
- **Assumed:** BEC/age/radius selection rule; independent species sampling;
  one-hectare-equivalent density; no spatial competition or defects.
- **Synthetic:** the generated tree list and its stand metrics.
- **Raster context:** LidarBC 2024 1 m DEM/DSM-derived CHM and slope; partial plot coverage.

## Limits

- The synthetic stand is a statistical reconstruction, not a surveyed stand or a
  real tree map.
- The selection is a documented rule, not a probability sample.
- The hold-out set has only five plots.
- The height model ignores species and measured-height coverage differences.
- LiDAR/terrain context is partial: four plots lack raster coverage and one has no valid sampled pixels.
- No operational, financial, or regulatory conclusion is supported.

## Reproduce

```bash
python scripts/select_research_forest_plots.py \
  --forest gavin \
  --site-header data/external/faib/raw/faib_header.csv \
  --plot-header data/external/faib/raw/faib_plot_header.csv \
  --sample-byvisit data/external/faib/raw/faib_sample_byvisit.csv \
  --output notes/reports/phase7_plot_selection_gavin.json

python scripts/validate_phase7_stand.py \
  --forest gavin \
  --faib-root data/external/faib/raw \
  --release-root build/phase7 \
  --report notes/reports/phase7_stand_validation_gavin.json \
  --bootstrap-replicates 100
```

## Recommended refinement

Before treating the synthetic stand as calibrated:

1. fit species-stratified DBH distributions or a mixture model;
2. fit height with species and a defensible height-coverage rule;
3. evaluate plot-level random effects and density constraints;
4. prepare a Gavin Lake LidarBC DEM/CHM bundle through an approved data task;
5. rerun the hold-out comparison and update this report.
