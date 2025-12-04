# FAIB Manifest Parquet Workflow

Nemora can emit FAIB manifest summaries as both CSV and Parquet. Parquet provides
columnar storage and faster downstream analytics—recommended for notebook or
Spark pipelines.

## CLI examples

- Fetch PSP extracts, auto-select BAFs, and generate manifests/stats:

  `nemora faib-manifest data/external/faib/manifest_psp --auto-bafs --auto-count 3 --parquet`

- Reuse cached downloads, limit rows, and emit Parquet alongside CSV:

  `nemora faib-manifest examples/faib_manifest --source tests/fixtures/faib --no-fetch --baf 12 --max-rows 200 --parquet`

## Loading the Parquet manifest

```python
import pandas as pd

manifest = pd.read_parquet("examples/faib_manifest/faib_manifest.parquet")
print(manifest.head())
```

The Parquet file mirrors the CSV schema (`dataset`, `baf`, `rows`, `path`,
`truncated`). Keep both formats until Parquet becomes the default so downstream
tools can migrate at their own pace.

## Feed manifest entries into sampling workflows

Once a manifest exists you can select an individual stand table, fit a
distribution, and draw samples while tuning the numeric integration settings:

```python
from pathlib import Path

import numpy as np
import pandas as pd

from nemora.core import InventorySpec
from nemora.distfit import fit_inventory
from nemora.sampling import SamplingConfig, sample_distribution

manifest = pd.read_parquet("examples/faib_manifest/faib_manifest.parquet")
stand_csv = Path(manifest.loc[0, "path"])  # CSV path captured in the manifest
stand_table = pd.read_csv(stand_csv)

bins = stand_table["dbh_cm"].to_numpy()
tallies = stand_table["tally"].to_numpy(dtype=float)

inventory = InventorySpec(
    name=stand_csv.stem,
    sampling="hps",
    bins=bins,
    tallies=tallies,
    metadata={"grouped": True},
)

fit = fit_inventory(inventory, ["weibull"], configs={})[0]
config = SamplingConfig(
    grid_points=4096,
    support_multiplier=12.0,
    integration_method="quad",
    quad_abs_tol=1e-9,
    quad_rel_tol=1e-8,
    cache_numeric_cdf=True,
)

draws = sample_distribution(
    fit.distribution,
    fit.parameters,
    size=500,
    random_state=123,
    config=config,
)
print(draws[:5])
```

This script loads the Parquet manifest, pinpoints the original stand-table CSV,
fits a Weibull distribution, and samples DBH draws using a high-resolution CDF
grid. Adjust `SamplingConfig` to benchmark how different grid densities or
integration methods affect accuracy/performance, and swap in
`nemora.sampling.bootstrap_inventory` when you need the richer metadata tracked
by `BootstrapResult`.
