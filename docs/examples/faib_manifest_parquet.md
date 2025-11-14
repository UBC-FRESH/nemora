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
