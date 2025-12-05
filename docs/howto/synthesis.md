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

Use `--layout geojson` when you already have polygon features that should drive seed placement;
repeat `--layout-geojson path/to/polygons.geojson` to register the feature collection(s). The
generator uses polygon centroids, guaranteeing deterministic coordinates without converting the
files to CSV intermediates.

### Physiographic modifiers

Vector masks now accept multiple overlays, each tagged as `clip` or `exclude`. Repeat
`--mask-geojson path/to/mask.geojson --mask-mode clip --mask-name riparian` to constrain the
landscape, then supply `--mask-geojson path/to/waterbodies.geojson --mask-mode exclude` to carve
voids that remove specific polygons entirely. When multiple masks are provided, the CLI pairs
entries with the optional `--mask-mode` / `--mask-name` lists by position.

Raster constraints complement the vector overlays for quick slope/elevation gating. Provide NumPy
arrays (``.npy``/``.npz``) or CSV grids that span the seed bounding box, then describe the logic with
`--mask-raster path.npy --mask-raster-threshold 0.4 --mask-raster-mode keep`. The tessellation
pipeline samples each polygon’s seed coordinate against the raster value and discards polygons that
fall outside the configured threshold (keep mode) or inside an exclusion zone. Metadata emitted by
`export_seed_recipe` now lists both vector overlays and raster constraints so downstream exporters
can reproduce the same filters.

### Worked example — vector + raster overlays

The snippet below walks through the CLI flow for deterministic layouts that honour vector and raster
modifiers. First, create simple GeoJSON boundary/exclusion shapes:

```bash
mkdir -p artifacts/masks
cat <<'GEOJSON' > artifacts/masks/boundary.geojson
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"name": "planning-area"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[0, 0], [1.8, 0], [1.8, 1], [0, 1], [0, 0]]]
      }
    }
  ]
}
GEOJSON

cat <<'GEOJSON' > artifacts/masks/waterbodies.geojson
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {"name": "lake"},
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[0.9, 0.1], [1.7, 0.1], [1.7, 0.6], [0.9, 0.6], [0.9, 0.1]]]
      }
    }
  ]
}
GEOJSON
```

Next, produce a lightweight raster that keeps only elevations above 0.45:

```bash
python - <<'PY'
import numpy as np
arr = np.linspace(0.2, 0.8, num=36, dtype=float).reshape(6, 6)
np.save("artifacts/masks/elevation.npy", arr)
PY
```

Finally, run the CLI with GeoJSON-driven layouts, vector overlays, and the raster filter:

```bash
nemora synthesis-generate-seeds \
  --layout geojson \
  --layout-geojson artifacts/masks/boundary.geojson \
  --mask-geojson artifacts/masks/boundary.geojson \
  --mask-mode clip \
  --mask-name planning-area \
  --mask-geojson artifacts/masks/waterbodies.geojson \
  --mask-mode exclude \
  --mask-name water \
  --mask-raster artifacts/masks/elevation.npy \
  --mask-raster-threshold 0.45 \
  --mask-raster-mode keep \
  --metadata-only \
  --output artifacts/masks/seed_recipe.json
```

Use `jq '.metadata' artifacts/masks/seed_recipe.json` to double-check which overlays fired. Feed the
same metadata into `nemora.synthesis.exporters.export_geojson` (see below) whenever you want to
visualise the resulting polygons in GIS software.

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

## Stand attribute scaffolding

Use `nemora.synthesis.stands.build_templates` (or `load_templates_from_json`) to convert vegetation
summaries into reusable templates, then `sample_stand_attributes` to fill a target area with sampled
patch descriptors:

```python
from pathlib import Path
from nemora.synthesis import stands

templates = stands.load_templates_from_json(Path("data/veg_templates.json"))
samples = stands.sample_stand_attributes(templates, total_area=50.0, rng=np.random.default_rng(0))
for sample in samples:
    print(sample.vegetation_type, sample.age_class, f"{sample.area:.1f} ha")
```

Each sample records the vegetation type, chosen age class, and allocated area so later phases can
attach DBH distributions or bootstrap payloads per stand. The helper accepts optional weights (for
probability surfaces) and respects the same `np.random.Generator` hooks used elsewhere in the
synthesis module.

### Template JSON format

Attribute templates are plain JSON files. Each record declares the vegetation type, an optional
`area_weight`, and the available age classes (with optional weights/extras). A minimal example:

```json
[
  {
    "vegetation_type": "CedarHemlock",
    "area_weight": 0.55,
    "age_classes": [
      {"label": "30-60", "weight": 0.4, "site_index": 22},
      {"label": "60-90", "weight": 0.6, "site_index": 24}
    ],
    "extras": {"target_basal_area": 28.5}
  },
  {
    "vegetation_type": "DouglasFir",
    "area_weight": 0.45,
    "age_classes": [
      {"label": "20-40", "weight": 0.3},
      {"label": "40-80", "weight": 0.7}
    ]
  }
]
```

`stands.load_templates_from_json` validates the schema, normalises weights (uniform when omitted),
and preserves any custom `extras` mapping so later phases can propagate site-based modifiers.

### Python walkthrough — sampling attributes + exporting GeoJSON

```python
from pathlib import Path
import numpy as np

from nemora.synthesis import exporters, stands, tessellation

# 1. Sample 25 ha of attributes.
templates = stands.load_templates_from_json(Path("data/veg_templates.json"))
samples = stands.sample_stand_attributes(
    templates,
    total_area=25.0,
    rng=np.random.default_rng(0),
)

# 2. Generate deterministic hex seeds that match the sample count.
seed_cfg = tessellation.VoronoiSeedConfig(
    count=len(samples),
    layout=tessellation.SeedLayoutConfig(mode=tessellation.SeedLayoutMode.HEX),
)
seed_result = tessellation.generate_seed_points(seed_cfg)

# 3. Pair polygons with the sampled attributes and emit GeoJSON + recipe metadata.
features = []
for poly, sample in zip(seed_result.polygons, samples):
    features.append(
        {
            "type": "Feature",
            "properties": {
                "veg_type": sample.vegetation_type,
                "age_class": sample.age_class,
                "area_ha": sample.area,
            },
            "geometry": {"type": "Polygon", "coordinates": [poly.tolist()]},
        }
    )

exporters.export_geojson(features, Path("artifacts/stands.geojson"))
exporters.export_seed_recipe(seed_result, Path("artifacts/seed_recipe.json"))
```

Drop the resulting GeoJSON into QGIS/ArcGIS to visualise the tessellation while preserving the seed
configuration + CJFR metrics for regression tests.

### CLI helper — sampling without Python scaffolding

Use the Typer subcommand when you just need a manifest of sampled stands:

```bash
nemora synthesis-sample-attributes \
    --templates data/veg_templates.json \
    --total-area 40 \
    --seed 2025 \
    --output artifacts/stands_sampled.json
```

The command loads the template JSON, samples enough patches to cover the requested area, and writes a
JSON list such as:

```json
[
  {"vegetation_type": "CedarHemlock", "age_class": "60-90", "area": 4.22},
  {"vegetation_type": "DouglasFir", "age_class": "20-40", "area": 3.01}
]
```

Pass the manifest downstream to synthesis/exporter scripts or stash it alongside the Voronoi seed
recipe so regression tests share the same attribute plan.

### CLI helper — attach attributes to polygons

1. Export a seed recipe that includes polygons:

```bash
nemora synthesis-generate-seeds \
    --count 40 \
    --include-polygons \
    --metadata-only \
    --output artifacts/seeds_with_polygons.json
```

2. Sample attributes as shown above:

```bash
nemora synthesis-sample-attributes \
    --templates data/veg_templates.json \
    --total-area 40 \
    --seed 123 \
    --output artifacts/stands_sampled.json
```

3. Assign the samples to polygons and emit GeoJSON (`export_stand_geojson_from_polygons` powers this CLI):

```bash
nemora synthesis-assign-stands \
    --seed-recipe artifacts/seeds_with_polygons.json \
    --attributes artifacts/stands_sampled.json \
    --output artifacts/stands.geojson
```

Use `--strict-count` when you expect the number of non-empty polygons to match the sampled stands
exactly; otherwise, the command truncates whichever list is longer and prints a warning. The output
GeoJSON stores both the sampled template area and the actual polygon area so downstream workflows
can reconcile differences.

Behind the scenes the CLI calls `nemora.synthesis.exporters.export_stand_geojson_from_polygons`,
which you can also import directly if you want to stitch polygons + samples inside a notebook or
custom workflow.

## Link stand manifests to bootstrap DBH payloads

Once you have sampled stand attributes and exported bootstrap DBH payloads (via
`nemora sampling-export-bootstrap-dbh`), use `nemora synthesis-link-bootstraps` to connect each stand
to a concrete payload. The command consumes a *plan* file that maps vegetation/age-class pairs to
bootstrap JSON artifacts:

```json
{
  "rules": [
    {"name": "cedar-old", "vegetation_type": "CedarHemlock", "age_class": "60-90",
     "bootstrap": "bootstrap/cedar_60_90.json"},
    {"name": "cedar", "vegetation_type": "CedarHemlock", "bootstrap": "bootstrap/cedar.json"},
    {"name": "fir", "vegetation_type": "DouglasFir", "bootstrap": "bootstrap/dfir.json"}
  ],
  "default_bootstrap": "bootstrap/default.json"
}
```

Paths are resolved relative to the plan file, so you can keep the JSON alongside its referenced
payloads (the files are just the JSON produced by `sampling-export-bootstrap-dbh`). Run the linker:

```bash
nemora synthesis-link-bootstraps \
    --attributes artifacts/stands_sampled.json \
    --plan artifacts/bootstrap_plan.json \
    --id-prefix stand \
    --output artifacts/stand_bootstrap_manifest.json
```

The resulting manifest captures the attribute source, plan, bootstraps, and generated stand IDs:

```json
{
  "attributes_source": "artifacts/stands_sampled.json",
  "plan_source": "artifacts/bootstrap_plan.json",
  "bootstraps": {
    "cedar-old": {
      "source": "bootstrap/cedar_60_90.json",
      "metadata": {"distribution": "weibull", "resamples": 5, "...": "..."},
      "dbh_vectors": {"0": [22.4, 23.1], "1": [21.0]}
    }
  },
  "assignments": [
    {
      "stand_id": "stand-0001",
      "vegetation_type": "CedarHemlock",
      "age_class": "60-90",
      "area": 4.2,
      "bootstrap_id": "cedar-old"
    }
  ]
}
```

Share `bootstraps` across multiple stands by pointing multiple rules at the same JSON artifact. The
manifest will only embed each payload once, and downstream workflows can look up payload metadata by
the `bootstrap_id` field stored on every assignment. Future synthesis steps will merge this manifest
with the polygon GeoJSON so each stand polygon has both attribute defaults and DBH vectors ready for
tree-list generation.

### Embed bootstrap metadata in the stand GeoJSON

Pass the manifest into `synthesis-assign-stands` to propagate the generated stand IDs and bootstrap
references into the GeoJSON output:

```bash
nemora synthesis-assign-stands \
    --seed-recipe artifacts/seeds_with_polygons.json \
    --attributes artifacts/stands_sampled.json \
    --bootstrap-manifest artifacts/stand_bootstrap_manifest.json \
    --output artifacts/stands_with_bootstrap.geojson
```

Each feature now includes the sampled vegetation/age attributes plus:

- `stand_id`: deterministic identifier from the manifest (`stand-0001`, etc.).
- `bootstrap_id`: key referencing the embedded payload.
- `bootstrap_metadata`: subset of the payload metadata (distribution, parameters, resamples, sample
  size, and source path).

Downstream tools can read the GeoJSON directly to discover which DBH payload to use for each polygon,
removing the need to join manifests manually.

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
