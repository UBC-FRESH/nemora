# FAIB Ingest Pipeline Outline

Date: 2025-11-07
Status: In progress — stand-table aggregation, FTP fetch helper (with overwrite-safe caching), CLI manifest automation, and sample manifests committed; large-sample orchestration + downstream harmonisation still pending.

## Objectives

- Mirror the published FAIB data dictionaries so downstream users can trace column provenance.
- Subsample PSP and non-PSP tables by basal area factor (BAF) to create stand-table inputs suitable for `nemora.distfit`.
- Provide CLI entry points to fetch, cache, and transform raw CSV extracts.

## Proposed Pipeline

1. **Fetch**
   - Use `DatasetSource(fetcher=...)` to download `faib_plot_header`, `faib_sample_byvisit`, `faib_tree_detail`, etc.
   - Cache CSV extracts under `data/external/faib/` (gitignored) and support overwrite-safe refreshes via `.part` temp files.
   - ✅ `generate_faib_manifest` now wraps `download_faib_csvs`, enabling automated fetch + manifest creation via CLI/script.
   - ✅ FIA prototype helper aggregates `TREE`/`COND`/`PLOT` tables into stand tables (plot CN filter, DBH conversion).

2. **Transform**
   - Join headers and sample metadata by `(CLSTR_ID, VISIT_NUMBER, PLOT)`.
   - Filter PSP visits (`SAMP_TYP` codes) and compute per-tree expansion factors.
   - Bin DBH to centimetre midpoints, then aggregate tallies by BAF.
   - Produce manifest files summarising dataset metadata (region, plot count, BAF, truncation flags).

3. **Output**
   - Write per-plot CSVs under `examples/faib_psp_baf{N}/`.
   - Emit a tidy stand-table parquet for fast analytics (TODO; current pipeline writes CSV + manifest).

## Tests

- Unit tests covering join logic, DBH binning, and BAF subsampling using small CSV fixtures.
- Integration test (optional skip) that downloads a tiny slice via FTP to ensure schema alignment (pending).
- CLI smoke test added for `nemora faib-manifest`; extend with end-to-end download once CI policy confirmed.
- Add regression harness for FIA aggregation once trimmed fixtures are authored (TODO).

## Documentation

- Expand `docs/howto/ingest.md` (updated with FAIB portal/FTP notes, `--overwrite`, and `--max-rows`; add full pipeline walkthrough once ETL lands).
- Add CLI usage examples (`nemora ingest-faib --fetch --overwrite`) showing cache management (done).
