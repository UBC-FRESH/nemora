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

## Capturing JSONL telemetry

Use `--report-path` to append JSON lines that mirror the console summary. These logs power the
nightly ingest monitoring workflow and are handy when reviewing performance-sensitive pull requests.

```bash
nemora ingest-benchmark data/external/faib --no-fetch --iterations 3 \
  --report-path logs/ingest_benchmark.jsonl
tail -n 1 logs/ingest_benchmark.jsonl
{"timestamp":"2025-11-08T08:12:02.345Z","iterations":3,"average_seconds":1.82,"tree_total":12408,...}
```

Check the nightly GitHub Actions artifact (`ingest-benchmark-report`) if you need the latest trends
without rerunning the benchmark locally.

## Nightly automation

The `Nightly Ingest Integration` workflow runs the benchmark every night, parses the JSONL output
into Markdown/text summaries (`reports/ingest_benchmark_summary.md` / `.txt`), and enforces
`INGEST_BENCHMARK_AVG_THRESHOLD=3.0` seconds for the average runtime. When the threshold is
exceeded the job fails and the auto-created issue includes the summary table, making it easy to spot
regressions without digging through raw logs. Download the artifact if you need the full JSONL +
summary history for deeper analysis.
