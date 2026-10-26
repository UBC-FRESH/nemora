# sgsR worked example

`structure.asc` is a deterministic 20 x 20 synthetic structural metric grid.
It is small enough for a documentation/CI smoke test and is not real LiDAR.

Run the bridge:

```bash
export R_LIBS_USER=/path/to/R/library  # only if sgsR is in a custom library

nemora sampling-sgsr examples/sgsr_sampling/structure.asc \
  --output tmp/sgsr_example \
  --n-samples 24 \
  --seed 42 \
  --n-strata 3
```

The expected proportional allocation is eight samples in each of three strata.
The bridge writes `allocation.csv`, `samples.csv`, `strata.csv`, `strata.tif`,
and `metadata.tsv`.

Requires R and the GPL-3 `sgsR` package plus `terra` and `sf`. See
`docs/howto/sgsr_sampling.md` for the bridge decision and limits.
