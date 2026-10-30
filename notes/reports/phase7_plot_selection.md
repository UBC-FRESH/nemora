# Phase 7 plot selection

Date: 2026-10-30

Phase: P7

Task: P7.1

Status: selection and tree-detail resolution complete; stand validation continues in P7.2.

## Decision

Use **Gavin Lake** for the first validation stand.

- Gavin Lake has 20 selected mature plots within 25 km: 15 training and 5
  hold-out plots.
- Malcolm Knapp has only 6 selected mature plots within 50 km: 4 training and 2
  hold-out plots.
- The Malcolm Knapp sample is too thin for a meaningful independent hold-out
  comparison. Keep it as an exploratory fallback, not as a second validated
  stand in this pass.

## Selection rules

| Rule | Gavin Lake | Malcolm Knapp |
| --- | --- | --- |
| Centre | 52.462616, -121.763919 | 49.306630, -122.551375 |
| Radius | 25 km | 50 km |
| BEC | `ICHmk3`, `SBSdw1` | `CWHdm` |
| Minimum projected age | 60 years | 60 years |
| Sample type | `F` | `F` |
| First visit only | yes | yes |
| Split seed | 20261030 | 20261030 |
| Hold-out fraction | 0.25 | 0.25 |

The age filter uses `PROJ_AGE_1`; it is a source attribute, not a field-verified
stand age. The split is a SHA-256 key over `seed:plot_id`, so it is stable across
machines and row order.

## Results

Gavin Lake:

- 20 plots total; 15 train, 5 hold-out.
- BEC: 18 `SBSdw1`, 2 `ICHmk3`.
- Projected age: 74-152 years; median 124.
- Distance from target centre: 2.29-23.92 km.
- Sorted plot-ID SHA-256: `aa60d48f31bb27bafeb697a4ba3caa731c0b774906c1964f6ef1628ca7cb60f3`.

Malcolm Knapp:

- 6 plots total; 4 train, 2 hold-out.
- BEC: 6 `CWHdm`.
- Projected age: 78-98 years; median 93.
- Distance from target centre: 8.97-48.47 km.
- Sorted plot-ID SHA-256: `1cc08661ecc368f86f682f7026b546ce33bbe56f3df2889728626e4e0d535096`.

## Tree detail

The public tree-detail table resolves to 2,870 live tree rows across all 20 Gavin
Lake plots. Every selected plot has at least one live tree record.

- Trees per plot: 74-331; median 105.5.
- Species: 12 codes; dominant codes are `FD` (1,346), `AT` (477), `SW` (437),
  `PL` (258), and `BL` (172).
- DBH: 2.0-56.1 cm.
- Measured height: 1.9-41.2 m.
- Expansion tally (`PHF_TREE` sum): 40,920.48.

A local pilot release validated under `nemora.inventory.v1` with 20 plots, 2,870
tree rows, and 12 species. Tree rows are not committed; regenerate them from the
public inputs with `resolve_tree_detail`.

## Provenance

- Source: BC FAIB PSP public compilation.
- Source URI: `ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/psp/`.
- Licence: OGL-BC.
- Input tables: `faib_header.csv`, `faib_plot_header.csv`,
  `faib_sample_byvisit.csv`.
- Per-file sizes and SHA-256 digests are recorded in the two manifests:
  `phase7_plot_selection_gavin.json` and `phase7_plot_selection_mkrf.json`.

## Reproduction

The manifests deliberately omit plot coordinates. Regenerate them from the public
inputs with:

```bash
python scripts/select_research_forest_plots.py \
  --forest gavin \
  --site-header data/external/faib/raw/faib_header.csv \
  --plot-header data/external/faib/raw/faib_plot_header.csv \
  --sample-byvisit data/external/faib/raw/faib_sample_byvisit.csv \
  --output notes/reports/phase7_plot_selection_gavin.json
```

Use `--forest mkrf` for the fallback target.

## Limits

- The selected plots are ground samples, not a tree map.
- The selection is a documented rule, not a probability sample.
- The expansion tally is not a literal stem count; P7.2 must use it explicitly.
- Canopy/terrain context is not evaluated yet; P7.3 does that.
