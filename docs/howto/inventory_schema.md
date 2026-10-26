# Shared Inventory Schema

`nemora.inventory.v1` is the shared tidy representation for inventory releases.
It gives FAIB, FIA, HPS, and future adapters one place to record plot identity,
tree or DBH-class tallies, species, canonical units, and provenance.

This is an ingest boundary, not a new inventory source. Adapters convert an
existing release into the schema; downstream fitting and sampling consume the
canonical tables.

## Release directory

A release is a directory:

```text
my-release/
├── release.json
├── plots.csv
├── trees.csv
└── species.csv
```

`release.json` carries the schema version, release id, units, provenance, CRS,
metadata, and table column lists. The CSV tables are always written with the
full canonical columns, even when an input only supplies a subset.

Use the reader and writer rather than hand-editing the manifest:

```python
from pathlib import Path

from nemora.ingest.schema import (
    InventoryProvenance,
    read_inventory_release,
    validate_inventory_release,
    write_inventory_release,
)

release = read_inventory_release(Path("data/inventory/my-release"))
validate_inventory_release(release)
write_inventory_release(release, Path("data/inventory/my-release-copy"))
```

## Tables

### `plots.csv`

One row per plot visit. Required: `plot_id`. Optional columns include
`cluster_id`, `visit_number`, `plot_number`, `measurement_date`,
`sample_type`, `site_identifier`, `baf`, `tree_count`, `latitude`, and
`longitude`.

`plot_id` must be unique. Coordinates require `release.crs`. `baf` is in
`m2_per_ha`; `tree_count` is a count.

### `trees.csv`

One row per tree or per DBH tally class. Required: `plot_id`, `dbh_cm`,
`tally`. Optional: `tree_id`, `species_code`, `height_m`, `status`.

- `dbh_cm` and `height_m` are in centimetres and metres.
- `tally` is a count or expansion weight in count units.
- `tree_id` is nullable for aggregated DBH classes.
- Every `plot_id` must exist in `plots.csv`.
- If `species_code` is present, a species lookup is required.

### `species.csv`

One row per species code. Required: `species_code`; optional `common_name` and
`scientific_name`.

### Units

The canonical units are fixed:

| Field | Unit |
| --- | --- |
| `dbh` | `cm` |
| `height` | `m` |
| `tally` | `count` |
| `baf` | `m2_per_ha` |

An adapter converts source units before constructing the release. The validator
rejects an unknown unit rather than guessing.

### Provenance

`release.json` requires:

- `source_id`;
- `source_version`; and
- `retrieved_utc` with an explicit timezone.

Optional provenance fields are `source_uri`, `licence`, `checksum`, and
`notes`. The schema version itself is required; a manifest without
`schema_version` will not load.

## Migrated path: HPS

HPS is the first migrated path because its manifest and DBH tallies are already
tidy. The adapter is additive:

```python
from pathlib import Path

from nemora.ingest.hps import (
    export_hps_inventory,
    export_hps_outputs,
    load_plot_selections,
    run_hps_pipeline,
)

selections = load_plot_selections(
    Path("faib_plot_header.csv"),
    Path("faib_sample_byvisit.csv"),
    baf=12.0,
)
result = run_hps_pipeline(Path("faib_tree_detail.csv"), selections)

# Historical per-plot CSV + manifest bytes, unchanged.
export_hps_outputs(
    result.tallies,
    result.manifest,
    output_dir=Path("out/hps_baf12"),
    manifest_path=Path("out/hps_baf12/manifest.csv"),
)

# Canonical release in addition to the legacy files.
export_hps_inventory(
    result,
    Path("out/hps_baf12_inventory"),
    release_id="hps-baf12",
)
```

HPS tallies are DBH-class aggregates, so the canonical `trees` table has a null
`tree_id`, a null `species_code`, and `status="live"`. The `plots` table carries
the manifest fields and `tree_count`. The legacy export is verified byte-for-byte
against `tests/fixtures/hps_export_expected/`.

From the CLI:

```bash
nemora ingest-faib-hps data/external/psp/raw \
  --output out/hps_baf12 \
  --inventory-dir out/hps_baf12_inventory \
  --no-fetch
```

## Validation failures

The validator rejects:

- a missing required table column, such as `tally` or `dbh_cm`;
- an unknown unit;
- a missing or unknown `schema_version`;
- missing or timezone-free provenance;
- duplicate plot ids;
- trees referencing an unknown plot; and
- species codes without a lookup.

The failure tests live in `tests/test_inventory_schema.py`.

## Migration order

| Order | Path | Status | Why |
| --- | --- | --- | --- |
| 1 | HPS | migrated | Manifest + DBH tallies already map cleanly; legacy bytes preserved |
| 2 | FAIB stand table | pending | BAF and expansion weights are explicit; needs plot provenance |
| 3 | FIA stand table | pending | Condition proportions and plot CNs need a stable plot key |
| 4 | Tree-level readers | pending | Species, status, and height coverage vary by source |

Do not migrate all paths at once. Each adapter must keep its legacy output
available for one release while adding the canonical release alongside it.

## Nightly check

The nightly ingest workflow runs the schema regression suite in addition to the
live FAIB/FIA download checks. A schema failure therefore produces the same
nightly issue path as an upstream source failure.
