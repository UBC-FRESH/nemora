# Phase 7 plan: research-forest plot and LiDAR validation

Parent issue: #20

Branch: `feature/p7-research-forest-validation`

Status: active

## Goal

Produce a reproducible validation package for one or both research forests:

- public FAIB PSP plots resolved into `nemora.inventory.v1`;
- a plot-conditioned mature synthetic stand per forest;
- held-out comparisons for DBH, height, basal area, stems/ha, and species mix;
- prepared DEM/CHM context through `nemora.lidar`;
- a limitations-first report that does not claim a real tree map.

## Defaults and decision points

- Start with Gavin Lake when the local FAIB PSP sample is adequate for the target
  BEC and mature-age range.
- If the local sample is too thin, use Malcolm Knapp with an explicit regional
  pooling rule. Record the rule before looking at validation results.
- Use one mature stand per forest for the first validation.
- Keep all inputs public or synthetic. No private plot locations or unpublished
  attributes.

## Workflow

### P7.1: select plots

1. Define the forest, BEC, mature-age threshold, and proximity filter.
2. Load public FAIB plot, visit, and tree tables.
3. Resolve canonical fields with `nemora.ingest.plots`.
4. Record source URI, licence, retrieval date, filters, counts, and checksums.
5. Split plots deterministically into training and hold-out sets.
6. Write the selection manifest and note before fitting.

### P7.2: stand validation

1. Build training and hold-out `nemora.inventory.v1` releases.
2. Fit/bootstrap DBH and height structure from training plots only.
3. Generate a seeded mature synthetic stand.
4. Compare held-out DBH, height, basal area, stems/ha, and species mix.
5. Propagate plot-resampling and seed uncertainty.

### P7.3: LiDAR context

1. Use prepared DEM/CHM products already produced by FEMIC/LidarBC.
2. Record product name, source, licence, CRS, resolution, and footprint.
3. Sample plot locations and synthetic trees through `nemora.lidar`.
4. Summarize terrain and canopy cover/height without reprojection or resampling.
5. Report missing coverage and do not substitute a product silently.

### P7.4: report

The report must separate:

- observed FAIB plot measurements;
- fitted distributions and assumptions;
- synthetic stand values;
- raster-derived context;
- uncertainty and negative results;
- what the validation does not establish.

## Reproducibility

- Record the split seed, stand seed, software versions, input checksums, and
  commands.
- Keep manifests and small public/synthetic fixtures in the repository.
- Keep large public extracts outside tracked files and record their retrieval
  commands.
- Do not commit credentials, private locations, or machine-specific paths.

## Verification

- `python -m pytest -q`
- `ruff format --check src tests`
- `ruff check src tests`
- `mypy src`
- `sphinx-build -b html docs _build/html -W`
- `pre-commit run --all-files`

## Open choices and recommendations

- **Forest order:** Gavin Lake first; fall back to Malcolm Knapp only if the
  plotted sample cannot support a mature-stand split. Reason: Gavin Lake is the
  documented first target and keeps the local BEC story simple.
- **Validation depth:** one mature stand per forest for the first pass. Reason:
  the goal is a defensible validation loop, not a landscape inventory.
- **LiDAR scope:** prepared rasters only. Reason: raw production remains in
  FEMIC/WS4 and is outside this phase.
