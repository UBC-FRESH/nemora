# sgsR Sampling Bridge

`nemora.sampling.sgsr` wraps the GPL-3 `sgsR` R package through a small R
script. It stratifies a single structural metric layer with quantiles, allocates
samples, and returns the allocation, sample coordinates, and structural strata.

## Port or bridge?

**Bridge.** `sgsR` is GPL (>= 3), while Nemora is MIT. Copying or porting sgsR
code into Nemora would create a licence and maintenance problem. The bridge
calls the installed R package instead.

The cost is a real R runtime dependency:

- R and the `sgsR`, `terra`, and `sf` R packages must be installed;
- users may need to set `R_LIBS_USER` when `sgsR` is in a non-default library;
- the bridge follows sgsR's public function signatures, so an upstream API
  change can require a wrapper update; and
- integration tests are skipped when sgsR is unavailable.

## Install

```bash
# Debian/Ubuntu system libraries used by terra/sf
sudo apt-get install -y r-base libgdal-dev libgeos-dev libproj-dev libudunits2-dev

# R packages
Rscript -e 'install.packages(c("sgsR"), repos="https://cloud.r-project.org")'
```

If you installed sgsR into a custom library, set `R_LIBS_USER` before running
Nemora:

```bash
export R_LIBS_USER=/path/to/R/library
```

## Worked example

The repository includes a deterministic 20 x 20 structural metric grid:

```bash
nemora sampling-sgsr examples/sgsr_sampling/structure.asc \
  --output tmp/sgsr_example \
  --n-samples 24 \
  --seed 42 \
  --n-strata 3
```

Equivalent Python:

```python
from pathlib import Path

from nemora.sampling.sgsr import run_sgsr

result = run_sgsr(
    Path("examples/sgsr_sampling/structure.asc"),
    Path("tmp/sgsr_example"),
    n_samples=24,
    seed=42,
    n_strata=3,
)
print(result.allocation)
print(result.strata)
print(result.samples.head())
```

The bridge writes:

- `allocation.csv` — per-stratum `strata` and `total` sample counts;
- `samples.csv` — `x`, `y`, `strata`, `type`, and `rule`;
- `strata.csv` — per-stratum cell count and metric min/max/mean;
- `strata.tif` — the structural strata raster; and
- `metadata.tsv` — R/sgsR/terra/sf versions, licence, seed, and counts.

For the worked example the allocation is:

```text
strata,total
1,8
2,8
3,8
```

The three strata contain 134, 133, and 133 cells. Their metric ranges are
approximately `7.0-11.486`, `11.487-13.767`, and `13.783-18.0`.

## Reproducibility

`seed` is required. Set the same metric, stratum count, allocation, method, and
seed to reproduce the same allocation and sample coordinates. The result
metadata records the seed, sgsR version, and GPL-3 licence.

The integration test runs the bridge twice and compares both the allocation and
the sample table.

## Errors and limits

- A missing raster, constant raster, or raster with no valid cells raises a
  clear `SgsRError`.
- A missing `Rscript` or `sgsR` installation raises `SgsRError` with install
  guidance.
- `n_samples < 1`, `n_strata < 2`, an unknown allocation, an unknown method, or
  a negative `mindist` raises `ValueError`.
- The bridge does not add a new sampling theory, a LiDAR metric pipeline, a
  field campaign, or a FEMIC/ws4 model input. It is an adapter around the
  published sgsR workflow.
- sgsR's structural strata and allocation are only as meaningful as the input
  metric layer. The structural metric must be provided and validated by the
  user.

## Tests

```bash
pytest tests/test_sgsr.py -q
```

The real bridge tests are skipped unless `Rscript` can load `sgsR`. The
argument, missing-input, missing-Rscript, and seed-required tests always run.

## References

- Goodbody, T.R.H., Coops, N.C., Queinnec, M., et al. (2023). *sgsR: a
  structurally guided sampling toolbox for LiDAR-based forest inventories.*
  Forestry: An International Journal of Forest Research.
  https://doi.org/10.1093/forestry/cpac055
- Queinnec, M., White, J.C., & Coops, N.C. (2021). Comparing airborne and
  spaceborne photon-counting LiDAR canopy structural estimates across different
  boreal forest types. *Remote Sensing of Environment*, 262, 112510.
- sgsR package: https://cran.r-project.org/package=sgsR (GPL >= 3).
