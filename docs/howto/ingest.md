# Ingest Module (Draft)

This page introduces the scaffolding for the forthcoming `nemora.ingest` module.
It covers the core abstractions (`DatasetSource`, `TransformPipeline`) that new
connectors will extend to transform raw forest inventory releases (BC FAIB, FIA,
etc.) into the tidy stand tables consumed by `nemora.distfit`, `nemora.sampling`,
and other modules.

## DatasetSource

`DatasetSource` captures enough metadata for the toolkit to locate/download raw
files. Provide a `fetcher` callable when remote retrieval is required:

```python
from pathlib import Path

from nemora.ingest import DatasetSource


def fetch_bc_faib(source: DatasetSource) -> list[Path]:
    output_dir = Path("data/external") / source.name
    output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: integrate with the FAIB portal API (https://bcgov-env.shinyapps.io/FAIB_GROUND_SAMPLE/)
    # to download PSP/CMI/NFI/YSM extracts. For now, drop a placeholder.
    (output_dir / "README.txt").write_text("FAIB data placeholder\n", encoding="utf-8")
    return [output_dir]


BC_FAIB_SOURCE = DatasetSource(
    name="bc-faib",
    description="BC FAIB ground sample plots (PSP, CMI, NFI, YSM)",
    uri="https://bcgov-env.shinyapps.io/FAIB_GROUND_SAMPLE/",
    metadata={
        "notes": (
            "Public FAIB portal; subsample by BAF/prism size as needed. "
            "Bulk downloads also available via FTP under "
            "ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/psp/"
            " and the companion web interface at "
            "https://bcgov-env.shinyapps.io/FAIB_GROUND_SAMPLE/."
        )
    },
    fetcher=fetch_bc_faib,
)
```

When `BC_FAIB_SOURCE.fetch()` is invoked it delegates to `fetch_bc_faib`. Future
connectors will implement authenticated fetchers and cache management.

## TransformPipeline

`TransformPipeline` holds an ordered list of callables that accept/return
`pandas.DataFrame` objects:

```python
import pandas as pd

from nemora.ingest import TransformPipeline


def convert_units(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(dbh_cm=frame["dbh_mm"] / 10.0)


def compute_stand_table(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.assign(stand_table=frame["tally"] * frame["expansion_factor"])


pipeline = TransformPipeline(
    name="bc-faib-hps",
    metadata={"description": "Convert FAIB tallies to Nemora stand table format"},
)
pipeline.add_step(convert_units)
pipeline.add_step(compute_stand_table)
```

### Data dictionaries

FAIB publishes companion Excel data dictionaries alongside each compilation.
For example, the PSP release exposes `PSP_data_dictionary_20250514.xlsx` under
the FTP path above. Include these files in ingest documentation so analysts can
interpret column names (`faib_plot_header.csv`, `faib_tree_detail.csv`, etc.).
The non-PSP directory mirrors the structure (see
`ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/non_psp/`
and `non_PSP_data_dictionary_20250514.xlsx`). These spreadsheets map column
codes to descriptions; keep a local copy alongside any downloads so analysts can
interpret FAIB variable names when building pipelines.

.. note::

   The FAIB team confirmed the portal data is fully public and can be
   redistributed. For bulk processing the FTP endpoints above are faster and
   expose the complete PSP, CMI, NFI, and YSM compilations (hundreds of megabytes
   per table). Nemora stores fetched CSVs under `data/external/faib/`, which is
   already `.gitignore`-d; treat that directory as a local cache and avoid
   committing the raw extracts.

   During rapid iteration you can limit downloads to specific files by passing
   `filenames=["faib_plot_header.csv"]` to :func:`nemora.ingest.faib.download_faib_csvs`
   so that small metadata tables can be fetched without transferring the
   multi-hundred-megabyte tree detail extracts.
```

Running `pipeline.run(raw_frame)` applies the configured steps sequentially—
ideal for cleaning CSV extracts, building stand tables, and harmonising column
names. Pipelines will be orchestrated by future CLI commands.

See `nemora.ingest.faib` for utilities (`load_psp_dictionary`,
`load_non_psp_dictionary`, `aggregate_stand_table`) that download schemas and
collapse tree detail tables into Nemora-ready stand-table summaries.

.. todo:: Flesh out end-to-end ingestion workflows (including CLI usage and
          caching guidelines) once dataset connectors are implemented.

## CLI helper

Nemora exposes an early CLI stub for PSP stand tables:

```bash
nemora ingest-faib tests/fixtures/faib --baf 12 --output stand_table.csv

# Fetch PSP extracts and write output
nemora ingest-faib data/external/faib --baf 12 --fetch --dataset psp --output stand_table.csv
# Force a fresh download (overwrite cached files) before building the stand table
nemora ingest-faib data/external/faib --baf 12 --fetch --overwrite --output stand_table.csv
# Preview suggested BAF values and exit without generating a table
nemora ingest-faib data/external/faib --auto-bafs --fetch --dataset psp

# Fetch extracts, auto-select BAFs, and generate a manifest + stand tables
nemora faib-manifest data/external/faib/manifest_psp --auto-bafs --auto-count 3
# Reuse an existing download, skip fetch, and limit each stand table to 200 rows
nemora faib-manifest examples/faib_manifest --source tests/fixtures/faib --no-fetch --baf 12 --max-rows 200

# Generate trimmed fixtures + manifest (used in tests)
python scripts/generate_faib_manifest.py examples/faib_manifest --dataset psp
# Auto-select representative BAF values before generating the manifest
python scripts/generate_faib_manifest.py examples/faib_manifest --dataset psp --auto
# Limit stand tables to the first 200 rows when exporting the manifest samples
python scripts/generate_faib_manifest.py examples/faib_manifest --dataset psp --auto --max-rows 200

# Aggregate an FIA stand table (prototype) using local CSV extracts
python - <<'PY'
from nemora.ingest.fia import build_stand_table_from_csvs

table = build_stand_table_from_csvs(
    "data/external/fia/raw",
    plot_cn=47825253010497,
)
print(table.head())
PY

# Aggregate FIA stand tables via CLI (trimmed fixtures example)
nemora ingest-fia tests/fixtures/fia --tree-file tree_small.csv --cond-file cond_small.csv \
  --plot-file plot_small.csv --plot-cn 47825261010497 --plot-cn 47825253010497 --output fia_sample.csv
```

The command expects pre-downloaded FAIB CSV extracts; future versions will
bundle fetch/caching logic.

## Repository sample

The repository contains a trimmed PSP example generated with
`scripts/generate_faib_manifest.py` under `examples/faib_manifest/`.
The manifest (`faib_manifest.csv`) lists each stand-table CSV (e.g.,
`stand_table_baf12.csv`) alongside the BAF, row count, and a `truncated` flag so
tests and documentation can reference a lightweight sample of the full FAIB
release. Re-run the script with `--max-rows` to regenerate the samples from a
larger local cache without bloating the repository.

The CLI and script both call :func:`nemora.ingest.faib.generate_faib_manifest`,
which orchestrates downloads, BAF selection, stand-table aggregation, and
manifest creation. The helper returns the manifest path, generated table paths,
and any files downloaded so automated workflows can inspect the output.

## FIA prototype

Nemora includes early helpers for USDA FIA CSV extracts
(:mod:`nemora.ingest.fia`). The :func:`nemora.ingest.fia.build_stand_table_from_csvs`
function joins ``TREE``/``COND``/``PLOT`` tables, filters live
trees/conditions, converts DBH to centimetres, and aggregates stand tables
weighted by ``TPA_UNADJ`` and condition proportions. These utilities are the
first step toward a full FIA ingest pipeline; use them to validate schema joins
on downloaded samples while additional ETL automation is being planned.

The CLI supports automatic downloads via ``--fetch-state``; Nemora maps state
codes to the public FIA Datamart URLs (for example ``nemora ingest-fia
data/fia --fetch-state hi`` will retrieve ``HI_TREE.csv``, ``HI_PLOT.csv``, and
``HI_COND.csv`` before aggregating). Downloads are optional—pass custom
``--tree-file``/``--cond-file``/``--plot-file`` arguments when working with
pre-existing extracts or trimmed fixtures.

**Licensing note:** FIA data are public domain but attribution is appreciated;
refer to the USDA legal notice at <https://www.fia.fs.usda.gov/contact/legal.php>.
When redistributing trimmed fixtures (e.g., under ``tests/fixtures/fia``) include
the citation and acquisition date so downstream users understand the provenance.
