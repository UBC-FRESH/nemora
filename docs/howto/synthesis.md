# Synthesis Bootstrap Integration (Planning)

Nemora’s upcoming `synthesis` module will consume bootstrap samples produced by
`nemora.sampling.bootstrap_inventory`. This page sketches how `BootstrapResult` feeds stem/stand
generators so downstream modules can align on a common contract. The helper utilities now live in
`nemora.synthesis.helpers` so downstream consumers do not need to duplicate schema wrangling.

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
