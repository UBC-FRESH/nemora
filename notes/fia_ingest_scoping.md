# FIA Ingest Scoping Notes

Date: 2025-11-07
Status: Draft — identify data sources, access requirements, and initial ETL targets.

## Objectives

- Determine publicly accessible FIA raw datasets suitable for Nemora ingest prototypes.
- Document download mechanisms (API vs bulk files) and licensing constraints.
- Identify minimal column subsets needed to reproduce Nemora stand-table inputs.
- Outline validation checks and fixtures to accompany an initial FIA connector.

## Data sources to evaluate

1. **USDA FIA Datamart** (https://apps.fs.usda.gov/fia/datamart/)
   - Downloadable CSV packages (plot, tree, condition tables) by state/region.
   - Requires agreeing to terms of use; no account needed.
   - Verify FTP/HTTPS endpoints and sizing for automation.
2. **FIADB API (beta)** (https://www.fia.fs.fed.us/tools-data/spatial/fiadb-downloads.php)
   - JSON/REST endpoints for filtered extracts.
   - Investigate rate limits and authentication.
3. **DataLad mirrors**
   - Check for existing mirrors (e.g., within OpenForestObservatory) to leverage provenance tools.

## Required tables/fields (initial hypothesis)

| Table | Key columns | Notes |
| --- | --- | --- |
| PLOT | `CN`, `LAT`, `LON`, `ELEV` | Spatial metadata for stand grouping |
| COND | `CN`, `PLOT`, `COND_STATUS_CD`, `SITECLCD` | Filter active conditions |
| TREE | `TREE_CN`, `PLOT`, `DBH`, `TPA_UNADJ`, `STATUSCD`, `SPCD` | DBH and expansion factors |
| SUBPLOT | `PLOT`, `SUBPLOT`, `SUBPLOT_STATUS_CD` | Optional; confirm necessity |

Tasks:
- Validate actual column names and joins after downloading a sample state (e.g., Oregon or Washington).
- Confirm expansion factor handling (`TPA_UNADJ` vs derived values) for HPS-equivalent tallies.

## ETL considerations

- Provide configurable filters (state, inventory year, condition status).
- Normalise DBH to centimetres (FIA stores inches).
- Derive basal area factors from plot design metadata if available; otherwise document assumptions.
- Aggregate to Nemora stand table via grouped DBH bins + expansion sums.

## Action items

- [x] Download a small FIA sample (single state) and inspect schemas; store notes under `data/external/fia/raw/`.
- [ ] Prototype a `DatasetSource` entry and fetcher stub mirroring FAIB approach.
- [x] Trim the HI sample into fixtures (`tests/fixtures/fia/`) with README/licensing notes and update tests to rely on them.
- [x] Add regression coverage exercising multi-condition weighting and non-live tree filtering.
- [x] Coordinate with documentation to ensure licensing/attribution requirements captured.
- [x] Draft a CLI workflow (`nemora ingest-fia`) after fixtures and regression harness are available.

## 2025-11-07 observations (HI sample)

- Pulled `HI_TREE.csv` (~9.1 MB), `HI_PLOT.csv` (~0.5 MB), and `HI_COND.csv` (~0.7 MB) into `data/external/fia/raw/`.
- Key join columns confirmed:
  - `PLOT.CN` ⇄ `TREE.PLT_CN` / `COND.PLT_CN`
  - `TREE`: `SUBP`, `TREE`, `CONDID`, `TPA_UNADJ`, `DIA` (inches), `STATUSCD`, `SPCD`.
  - `COND`: `CONDPROP_UNADJ`, `COND_STATUS_CD`, ownership (`OWNCD`), condition size (`STDSZCD`), site class fields.
  - `PLOT`: `LAT`, `LON`, `ELEV`, `INVYR`, `MEASYEAR`, design metadata (`DESIGNCD`, `REMPER`).
- DBH recorded in inches (`DIA`); conversion to cm required for Nemora stand tables.
- Expansion weights available via `TPA_UNADJ` (trees per acre); need conversion to per-ha or direct tally weights.
- Condition proportions (`CONDPROP_UNADJ`, `MICRPROP_UNADJ`, `SUBPPROP_UNADJ`, `MACRPROP_UNADJ`) will gate weighting when multiple conditions share a plot.
- Plot status codes (`PLOT_STATUS_CD`) and condition status codes (`COND_STATUS_CD`) determine live forest vs nonforest filtering.
- Next steps: derive sample aggregation script joining plot/cond/tree using the fixtures, evaluate basal area factors (if available), and capture column descriptions from FIA documentation.
- CLI fetch helper (`nemora ingest-fia --fetch-state HI`) wraps `download_fia_tables` and defaults filenames to the downloaded ``STATE_TABLE.csv`` convention.
- Consider adding an optional integration test (skipped by default) that exercises live downloads for a small state (e.g., HI) to monitor upstream schema changes.
