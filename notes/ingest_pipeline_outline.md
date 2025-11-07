# FAIB Ingest Pipeline Outline

Date: 2025-11-07
Status: In progress — stand-table aggregation (`build_stand_table_from_csvs`) and CLI stub implemented; fetch/caching still pending.

## Objectives

- Mirror the published FAIB data dictionaries so downstream users can trace column provenance.
- Subsample PSP and non-PSP tables by basal area factor (BAF) to create stand-table inputs suitable for `nemora.distfit`.
- Provide CLI entry points to fetch, cache, and transform raw CSV extracts.

## Proposed Pipeline

1. **Fetch**
   - Use `DatasetSource(fetcher=...)` to download `faib_plot_header`, `faib_sample_byvisit`, `faib_tree_detail`, etc.
   - Cache zipped artifacts under `data/external/faib/`.

2. **Transform**
   - Join headers and sample metadata by `(CLSTR_ID, VISIT_NUMBER, PLOT)`.
   - Filter PSP visits (`SAMP_TYP` codes) and compute per-tree expansion factors.
   - Bin DBH to centimetre midpoints, then aggregate tallies by BAF.
   - Produce manifest files summarising dataset metadata (region, plot count, BAF).

3. **Output**
   - Write per-plot CSVs under `examples/faib_psp_baf{N}/`.
   - Emit a tidy stand-table parquet for fast analytics.

## Tests

- Unit tests covering join logic, DBH binning, and BAF subsampling using small CSV fixtures.
- Integration test (optional skip) that downloads a tiny slice via FTP to ensure schema alignment.

## Documentation

- Expand `docs/howto/ingest.md` once the pipeline is implemented.
- Add CLI usage examples (`nemora ingest faib --baf 12`).
