# WS4 synthetic landscape fixture

This directory is a deterministic synthetic landscape exported by
`nemora.synthesis.ws4_fixture`. It is an integration fixture for the
Nemora → WS4 seam, not a calibrated synthetic forest and not a local yield
estimate.

## What is here

| File | Purpose |
| --- | --- |
| `recipe.xml` | Minimal ForestModelXML recipe: natural and managed volume curves, one clearcut treatment |
| `fragments.csv` | WS4 fragment inventory: block, area, age, AU, management state |
| `schedule.csv` | Empty schedule for the baseline simulation |
| `truth_table.csv` | Per-fragment area, age, expected volume, and placeholder structure |
| `fixture.json` | Seed, generator parameters, totals, and realism limits |
| `landscape.geojson` | The Voronoi fragments with their stand attributes |
| `README.md` | This note |

## Known properties

- Seed: `20261026`.
- Fragments: 12 Voronoi polygons.
- Total area: `36.0 ha`.
- Ages: `40`, `60`, and `80` years, four fragments each.
- Natural volume curve: `(0, 0)`, `(40, 120)`, `(80, 220)`, linearly
  interpolated.
- Expected volume: `6623.03115 m3` at the first simulation year.
- Placeholder structure: fir, lognormal DBH by age, 200 trees per hectare
  target, no mortality, no species mix, no treatment schedule.

The per-fragment values are in `truth_table.csv`. The total is the sum of
`expected_volume_m3`.

## Regenerate

From the Nemora repository root:

```bash
python scripts/generate_ws4_fixture.py \
  --output examples/ws4_synthetic \
  --seed 20261026 \
  --force
```

The generator requires the seed. The same seed and parameters reproduce the
fixture byte for byte. `--force` overwrites the generated files in place and
leaves this README alone.

## Run through WS4

With WS4 installed:

```bash
ws4 build \
  --xml examples/ws4_synthetic/recipe.xml \
  --fragments examples/ws4_synthetic/fragments.csv \
  --out tmp/ws4_synthetic/compiled

ws4 simulate \
  --dataset tmp/ws4_synthetic/compiled \
  --schedule examples/ws4_synthetic/schedule.csv \
  --years 2 \
  --out tmp/ws4_synthetic/sim
```

At offset `0`, `account_values.csv` should contain:

```text
0,feature.Volume,6623.031150000001
```

The local smoke test (`tests/test_synthesis_ws4_fixture.py`) checks this within
`1e-6 m3` and is skipped when WS4 is not installed.

## Limits

- One species, three ages, no mortality, no treatments, and no spatial
  adjacency. It does not represent a real landscape.
- Tree allometry is a placeholder from `nemora.synthesis.stems`; it is not
  calibrated for taper, biomass, or local yield.
- The volume curve is the synthetic WS4 example curve, not a local yield
  estimate.
- The fixture tests build/simulate wiring, determinism, and the truth-table
  contract. It is not evidence about a real forest.
