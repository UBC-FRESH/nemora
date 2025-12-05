# Synthesis Bootstrap Integration (Planning)

Nemora’s upcoming `synthesis` module will consume bootstrap samples produced by
`nemora.sampling.bootstrap_inventory`. This page sketches how `BootstrapResult` feeds stem/stand
generators so downstream modules can align on a common contract. The helper utilities now live in
`nemora.synthesis.helpers` so downstream consumers do not need to duplicate schema wrangling.

## Voronoi seed configuration (Phase 1 kickoff)

Phase 1 starts with reproducible Voronoi seed sets that mirror the CJFR/rlandscape control knobs.
Use `tessellation.VoronoiSeedConfig` to describe the point-process mixture, aspect ratio, and the
hole/merge editing fractions (`p_H`, `p_M`). The generator now returns a
`tessellation.VoronoiSeedResult` so downstream code (or docs/tests) can persist the control
parameters alongside the coordinates.

```python
import numpy as np
from pathlib import Path

from nemora.synthesis import tessellation, exporters

cfg = tessellation.VoronoiSeedConfig(
    count=200,
    aspect_ratio=2.0,
    mix=tessellation.PointProcessMix(uniform=0.4, cluster=0.4, inhibition=0.2),
    edit=tessellation.VoronoiEditConfig(hole_fraction=0.05, merge_fraction=0.1),
    rng=np.random.default_rng(20251205),
)
result = tessellation.generate_seed_points(cfg)
exporters.export_metadata_json(result.metadata(), Path("artifacts/seeds.json"))
```

`result.points` always contains `cfg.count` coordinates (post-editing). The metadata captures the
initial seed totals plus the hole/merge selections so Voronoi builders/CLI plumbing can reproduce
the same mixture later on.

### Editing knobs

`hole_fraction` and `merge_fraction` apply to the final target polygon count. The seed generator
internally produces `count + n_hole + n_merge` points, deletes the requested hole fraction, and
collapses random merge pairs into shared midpoints. Fractions must sum to < 1 (mirroring the CJFR
constraints) to guarantee a feasible configuration.

### CLI export

You can also export seed recipes directly from the CLI without writing Python scaffolding:

```bash
nemora synthesis-generate-seeds \
    --count 150 \
    --aspect-ratio 1.5 \
    --mix-uniform 0.5 \
    --mix-cluster 0.3 \
    --mix-inhibition 0.2 \
    --hole-fraction 0.05 \
    --merge-fraction 0.1 \
    --seed 20251205 \
    --output artifacts/seed_recipe.json
```

The resulting JSON contains the full configuration metadata (point-process mix, cluster/SSI/lattice
parameters, edit fractions) and, by default, the raw coordinate arrays. Add `--metadata-only` when
you only need the knobs (e.g., docs/tests that re-run the generator on demand). Each export also
captures the CJFR-style metrics (`n`, polygon-area `CV`, `μ_d`, `σ_d`) so downstream planning docs
can quote the same statistics without recomputing the Voronoi diagram. When a convex GeoJSON mask is
available, add `--mask-geojson path/to/polygon.geojson` (plus optional `--mask-name`) to clip the
Voronoi polygons/metrics to physiographic boundaries.

### Deterministic layouts

`VoronoiSeedConfig` now accepts a `SeedLayoutConfig`, enabling deterministic seed placement without
relying on the stochastic point-process mix. Set `layout=SeedLayoutConfig(mode="hex")` for a hex
packed grid or `layout=SeedLayoutConfig(mode="imported", points=array)` when upstream workflows
provide explicit `(x, y)` coordinates. Hex layouts derive spacing from the requested `count` and
`aspect_ratio`, ensuring repeatable coverage across doc/tests/CLI exports.

The CLI exposes the same controls:

```bash
# Hex-packed arrangement (ignores mix knobs)
nemora synthesis-generate-seeds --count 80 --layout hex --metadata-only --output seeds_hex.json

# Imported coordinates from CSV (x,y headers) or JSON points
nemora synthesis-generate-seeds \
    --count 50 \
    --layout imported \
    --layout-points fixtures/seed_points.csv \
    --output fixtures/imported_layout.json
```

Imported layouts expect coordinates in the unit box (x ∈ [0, aspect_ratio], y ∈ [0, 1]). CSV inputs
must expose `x` and `y` headers; JSON inputs can be a raw list of `[x, y]` pairs or an object with a
`points` list. Metadata emitted by `export_seed_recipe` reports the chosen layout mode plus the
number of coordinates provided so downstream docs/tests can cite the provenance.

## Expected input shape

```python
from nemora.sampling import BootstrapResult, bootstrap_inventory
from nemora.synthesis.helpers import bootstrap_to_dataframe

result: BootstrapResult = bootstrap_inventory(..., return_result=True)
frame = bootstrap_to_dataframe(result)
frame.attrs["nemora_bootstrap"]  # metadata dict (distribution, parameters, bins, tallies, etc.)
```

Synthesis can read either the stacked array (`result.stacked()`) or the richer DataFrame (with
attached metadata). Each bootstrap sample preserves:

- `distribution`, `parameters`: provenance of the fitted distribution.
- `bins`, `tallies`: original stand-table inputs (useful for diagnostics).
- `resample`, `bin`, `draw`: per-stem data powering stem generation.

Stand/stem generators should persist the metadata (e.g., attach `distribution`/`parameters` to the
output manifests) so simulation workflows can trace provenance.

## Helper module (`nemora.synthesis.helpers`)

Nemora exposes helper utilities that convert bootstrap results into synthesis-ready payloads:

```python
from nemora.synthesis.helpers import bootstrap_payload

payload = bootstrap_payload(result)
frame = payload.frame          # pandas.DataFrame with resample/bin/draw columns
stacked = payload.stacked      # numpy.ndarray view of all sampled (bin, draw) pairs
metadata = payload.metadata    # dict: distribution, parameters, bins, tallies, etc.
```

Upcoming synthesis APIs (`generate_stems_from_bootstrap`, `build_stand_attributes`) accept the
`BootstrapPayload` so they can group by `resample` and persist provenance alongside generated stems.

## CLI inspection

Use the Typer CLI to run a quick bootstrap and inspect the metadata without writing custom scripts:

```bash
nemora sampling-describe-bootstrap tests/fixtures/hps_psp_stand_table.csv \
    --distribution weibull \
    --resamples 3 \
    --sample-size 10 \
    --seed 2025 \
    --show-samples
```

The command auto-fits the requested distribution (unless you pass explicit `--param name=value`
assignments), bootstraps the stand table, prints the metadata tables, and optionally shows a preview
of sampled `(resample, bin, draw)` rows. Add `--json` when downstream tooling should ingest the
output programmatically.

## Next steps

- Flesh out synthesis stubs (`generate_stems_from_bootstrap` etc.) to consume the helper.
- Extend simulation planning notes so inventory simulators can ingest the same DataFrame.
- Wire automated docs/examples once synthesis code lands.

## Roadmap alignment

### Phase 1 — Landscape geometry scaffolding

- Translate the CJFR control metrics (`n`, `CV`, `μ_d`, `σ_d`) into CLI/API inputs.
- Map desired metrics to `tessellation.VoronoiSeedConfig` and persist the chosen parameters with the exported metadata JSON.

### Phase 2 — Stand & stem generation

- Use `stands.StandAttributeTemplate` to ingest vegetation/age tables (FLG-style Weibull parameters) and drive DBH generation via `nemora.sampling`.
- Ensure downstream exporters attach the bootstrap metadata so simulations inherit the provenance trail.

### Phase 3 — Export + CLI workflows

- Emit both GeoJSON and CSV/Parquet assets via `nemora.synthesis.exporters`.
- Provide a `nemora synthesis generate-landscape` CLI entry point that accepts YAML/JSON recipes describing seed processes, attribute templates, and exporter settings.

For now, keep this contract in mind when scripting bootstrap-driven workflows so future synthesis
components integrate cleanly.
