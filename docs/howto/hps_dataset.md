# BC PSP HPS Data

This note outlines how to assemble publicly available horizontal point sampling
datasets from the BC Forest Analysis and Inventory Branch (FAIB) compilations.
The goal is to obtain a clean, reproducible subset that mirrors the BAF 12 HPS
workflow used in the Vegetation Resource Inventory (VRI).

## Source

- FTP: `ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/`
  - `psp/` – Provincial Vegetation Resource Inventory permanent sample plots.
  - `non_psp/` – Related compilations for non‑PSP programmes.
- Metadata: `PSP_data_dictionary_20250514.xlsx`, `non_PSP_data_dictionary_20250514.xlsx`
  (download and store checksums alongside scripts).

## Relevant Tables

| File | Purpose | Key Fields |
| ---- | ------- | ---------- |
| `faib_plot_header.csv` | Plot descriptors (one per plot/visit) | `CLSTR_ID`, `VISIT_NUMBER`, `PLOT`, `SITE_IDENTIFIER` |
| `faib_sample_byvisit.csv` | Plot visit metadata | `CLSTR_ID`, `VISIT_NUMBER`, `FIRST_MSMT`, `MEAS_DT`, `SAMP_TYP` |
| `faib_tree_detail.csv` | Per-tree measurements (large; chunked download) | `CLSTR_ID`, `VISIT_NUMBER`, `PLOT`, `DBH`, `LV_D`, `TREE_NO`, `SP0` |

Additional summary tables (`faib_compiled_*`) provide aggregated basal area and
heights but are optional for the initial HPS tally pipeline.

## Extraction Recipe

1. **Mirror metadata**: save the data dictionaries and record SHA256 hashes in
   `data/external/psp/CHECKSUMS`.
2. **Filter plots**: load `faib_plot_header.csv` and retain rows that correspond
   to the desired PSP visit(s). The compilations do not store the BAF explicitly,
   so the workflow records the assumed value (BAF 12) alongside each plot.
3. **Join visit context**: merge `faib_sample_byvisit` on
   `(CLSTR_ID, VISIT_NUMBER)` to identify active measurement cycles (e.g.,
   `FIRST_MSMT == "Y"` for baseline PSP visits).
4. **Build tallies**: stream `faib_tree_detail.csv` with `pandas.read_csv(..., chunksize=...)`
   selecting the columns above; filter to plots discovered in step 2, keep live
   trees (`STATUS_CD == "L"`), and bin DBH to centimetre midpoints. Output per plot:
   - `dbh_cm` bin centre,
   - `tally` counts,
   - `baf` (12),
   - optional species/stratum attributes for future use.
   Store under `data/examples/hps_baf12/<plot_id>.csv`.
5. **Document lineage**: create `data/examples/hps_baf12/README.md` summarising
   the selection criteria, transformation script, and citation requirements.

## Command-line helper

Use `scripts/prepare_hps_dataset.py` to automate the recipe above. The script
downloads (or reuses cached) PSP CSVs, filters to first-measurement BAF 12 plots,
and writes per-plot tallies plus a manifest, following the data preparation steps
documented in the EarthArXiv preprint by Paradis (2025).

```bash
python scripts/prepare_hps_dataset.py \
  --output-dir data/examples/hps_baf12 \
  --cache-dir data/external/psp/raw \
  --baf 12 \
  --max-plots 25
```

Key options:

- `--include-all-visits`: keep every measurement instead of first-measurement
  plots.
- `--sample-type F`: restrict to specific `SAMP_TYP` codes if required.
- `--status L --status I`: define which tree status codes count as “live”.
- `--dry-run`: inspect how many plots would be produced without writing files.

### Sample bundle

The repository ships a small bundle generated with:

```bash
PYTHONPATH=src python scripts/prepare_hps_dataset.py \
  --output-dir examples/hps_baf12 \
  --manifest examples/hps_baf12_manifest.csv \
  --cache-dir data/external/psp/raw \
  --baf 12 \
  --max-plots 5
```

Outputs:

- Tallies: `examples/hps_baf12/*.csv`
- Manifest: `examples/hps_baf12_manifest.csv`
- Raw downloads cached (gitignored) under `data/external/psp/raw`.

## Automation Status

- [x] Scripted pipeline (`scripts/prepare_hps_dataset.py`) with caching and binning controls.
- [x] Pytest fixtures covering selection + tally logic (`tests/fixtures/hps`).
- [x] PSP sample bundle committed under `examples/hps_baf12` with manifest and provenance notes.
- [x] Regression guard for the reference Weibull fit (`tests/test_hps_parity.py`).
- [x] Censored meta-plot fixture + regression (`tests/fixtures/hps/meta_censored.csv`,
      `tests/test_censored_workflow.py`).
