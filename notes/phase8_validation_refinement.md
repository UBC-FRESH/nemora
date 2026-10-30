# Phase 8 plan: validation refinement and LiDAR coverage expansion

Parent issue: #26

Branch: `feature/p8-validation-refinement`

Status: active

## Goal

Improve the Gavin Lake validation in two directions:

1. replace or constrain the pooled Weibull stand model based on held-out
   performance;
2. close the four missing LidarBC plot-coverage gaps and resolve the one
   nodata plot.

The output is a before/after validation report, not a calibrated operational
model.

## P8.1: model refinement

Candidates, in order:

1. current pooled Weibull baseline (P7);
2. species-stratified Weibull fits with explicit small-sample fallback;
3. mixture DBH model;
4. plot-level/random-effect alternative where the sample supports it.

Height refinement:

- fit height–DBH with species terms;
- guard against measured-height coverage bias;
- report residual uncertainty.

Selection uses the same held-out plots and metrics: DBH KS, BA, QMD, mean DBH,
mean height, and species TV distance. A candidate that does not improve is
reported as a negative result.

## P8.2: LidarBC coverage

- Re-query the public LidarBC catalogue for the four missing plots.
- Materialize only the required DEM/DSM pairs through FEMIC.
- Derive CHM/slope and sample the plots.
- Confirm or explain the nodata plot.
- Record tile IDs, acquisition year, native CRS, resolution, licence, and
  coverage.

No raw point-cloud processing or reprojection is performed in Nemora.

## P8.3: rerun and report

- Rerun stand validation with the selected model.
- Rerun plot and synthetic-stand canopy/slope summaries.
- Report before/after metrics and uncertainty.
- Write `notes/reports/phase8_validation.md` and the close-out report.

## Verification

- `python -m pytest -q`
- `ruff format --check src tests scripts`
- `ruff check src tests scripts`
- `mypy src`
- `sphinx-build -b html docs _build/html -W`
- `pre-commit run --all-files`

## Defaults

- Gavin Lake remains the primary target; Malcolm Knapp remains exploratory.
- Select by held-out performance, not visual fit.
- Use public LidarBC products through FEMIC; keep raw processing outside
  Nemora.
- Report negative results and wide uncertainty explicitly.
