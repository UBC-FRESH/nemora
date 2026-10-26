# Synthetic WS4 Fixture

This how-to generates and checks a deterministic synthetic landscape for the
Nemora → WS4 build/simulate seam. It is a software integration fixture, not a
calibrated forest and not a yield estimate.

## What it produces

The fixture combines the existing synthesis modules:

- `nemora.synthesis.tessellation` for a seeded Voronoi landscape;
- `nemora.synthesis.stands` for fragment attributes; and
- `nemora.synthesis.stems` for a placeholder tree list and structure summary.

The checked-in fixture lives in `examples/ws4_synthetic/`:

```text
examples/ws4_synthetic/
├── README.md
├── recipe.xml
├── fragments.csv
├── schedule.csv
├── truth_table.csv
├── fixture.json
└── landscape.geojson
```

Known properties:

- seed `20261026`;
- 12 fragments, 36.0 ha total;
- ages 40, 60, and 80 (four fragments each);
- natural volume curve `(0, 0)`, `(40, 120)`, `(80, 220)`;
- expected first-year volume `6623.03115 m3`;
- 200 trees per hectare target, placeholder fir structure.

`truth_table.csv` is the published per-fragment truth: area, age, volume per
hectare, expected volume, tree count, density, mean DBH, mean height, and basal
area. `fixture.json` records the seed, generator parameters, totals, and limits.

## Generate

The seed is required so the fixture cannot silently become a different
landscape:

```bash
python scripts/generate_ws4_fixture.py \
  --output examples/ws4_synthetic \
  --seed 20261026 \
  --force
```

The generator also accepts `--count`, `--aspect-ratio`, `--area-ha`, and
`--trees-per-ha`. Changing any of these is a different fixture; update the
truth table and the docs together.

## Build and simulate

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

The baseline schedule is empty. At offset `0`, `account_values.csv` should
contain:

```text
0,feature.Volume,6623.031150000001
```

The expected value is the sum of `area_ha * volume_per_ha` in
`truth_table.csv`.

## Test

```bash
pytest tests/test_synthesis_ws4_fixture.py -q
```

The tests check:

- the generator is deterministic from the seed;
- the checked-in fixture matches a fresh generation byte for byte;
- the truth table reconciles with `fragments.csv` and the curve;
- removing the seed or truth table fails validation; and
- WS4 build/simulate reproduces the total volume within `1e-6 m3` when WS4 is
  installed.

The WS4 test uses `pytest.importorskip("ws4")`, so a Nemora-only CI run skips
the external build step rather than failing for a missing optional tool.

## Realism limits

- The landscape is one species, three ages, and no mortality or treatment
  schedule.
- The tree allometry is a placeholder from `nemora.synthesis.stems`; it is not
  calibrated for taper, biomass, or local yield.
- The volume curve is the synthetic WS4 example curve, not a local yield
  estimate.
- The fixture is not a real landscape and is not evidence about a real forest.

Use it to catch broken build wiring, seed drift, truth-table drift, and volume
accounting regressions. Use a real inventory and a validated yield model for a
management decision.
