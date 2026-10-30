# Research-forest plot selection

Phase 7 starts by selecting public FAIB PSP plots for one research-forest target.
The selection is explicit and reproducible: target centre, radius, BEC codes,
projected-age floor, sample type, and first-visit filter are all recorded.

## Targets

The checked-in script defines two targets:

| Key | Centre | Radius | BEC codes |
| --- | --- | --- | --- |
| `gavin` | 52.462616, -121.763919 | 25 km | `ICHmk3`, `SBSdw1` |
| `mkrf` | 49.306630, -122.551375 | 50 km | `CWHdm` |

The default path starts with Gavin Lake when its local sample can support a
train/hold-out split. Malcolm Knapp is the fallback, with the wider radius
recorded explicitly.

## Run the selection

```bash
python scripts/select_research_forest_plots.py \
  --forest gavin \
  --site-header data/external/faib/raw/faib_header.csv \
  --plot-header data/external/faib/raw/faib_plot_header.csv \
  --sample-byvisit data/external/faib/raw/faib_sample_byvisit.csv \
  --output notes/reports/phase7_plot_selection.json
```

The script requires the public FAIB PSP tables. It does not download them and
does not write plot coordinates to the output. The manifest contains:

- the target and filters;
- total, train, and hold-out counts;
- BEC counts, age range, and distance range;
- a SHA-256 digest of the sorted plot IDs;
- source URI, licence, retrieval timestamp, and input-file checksums.

The split uses a SHA-256 key over the split seed and plot ID, so it is stable
across machines and independent of row order.

## Python API

```python
from nemora.ingest.selection import (
    ForestPlotTarget,
    assign_holdout_split,
    select_forest_plots,
    selection_manifest,
)

target = ForestPlotTarget(
    name="example",
    latitude=52.462616,
    longitude=-121.763919,
    radius_km=25.0,
    bec_codes=("ICHmk3", "SBSdw1"),
)

selected = select_forest_plots(site_header, plot_header, sample_by_visit, target=target)
split = assign_holdout_split(selected, holdout_fraction=0.25, seed=20261030)
manifest = selection_manifest(
    split,
    target=target,
    holdout_fraction=0.25,
    split_seed=20261030,
)
```

## Limits

- A selected plot is a ground sample, not a tree map.
- The projected age is a source attribute, not a field-verified stand age.
- BEC and distance filters are selection rules, not a probability sampling
  design.
- The manifest deliberately omits plot coordinates; regenerate selection from
  the public inputs when coordinates are needed locally.
