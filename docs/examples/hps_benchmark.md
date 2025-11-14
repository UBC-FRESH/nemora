# HPS Pipeline Benchmarking

`nemora ingest-benchmark` measures how long the FAIB→HPS pipeline takes for a
given set of plots without writing outputs. Use this to sanity-check performance
before running large batch jobs or after modifying the pipeline.

## Running the benchmark

```bash
# Reuse local PSP extracts and run three iterations (default)
nemora ingest-benchmark data/external/faib --no-fetch

# Download PSP files to a cache directory and run five iterations
nemora ingest-benchmark data/external/faib --fetch --cache-dir data/external/psp/raw --iterations 5
```

Example output:

```
Iteration 1/3: 1.842s
Iteration 2/3: 1.807s
Iteration 3/3: 1.815s

┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┓
┃ Runs         ┃ Average (s)  ┃ Fastest (s)  ┃ Slowest (s)  ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━┩
│ 3            │ 1.821        │ 1.807        │ 1.842        │
└──────────────┴──────────────┴──────────────┴──────────────┘
Tree total: 12,408 (plots=3, live_status=L)
```

## Interpreting results

- **Average/Fastest/Slowest** help spot variability (e.g., cold cache vs warm cache).
- **Tree total / plot count** confirm the benchmark used the expected subset.
- Record typical timings in your project notes; if nightly ingest monitoring reports
significant deviations, rerun this benchmark to diagnose regression vs. upstream changes.
