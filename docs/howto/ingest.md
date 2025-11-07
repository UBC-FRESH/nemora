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
and `non_PSP_data_dictionary_20250514.xlsx`).
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
```

The command expects pre-downloaded FAIB CSV extracts; future versions will
bundle fetch/caching logic.
