# Nemora Modular Reorganisation Plan

Date: 2025-11-06
Status: Draft – living work plan for the forthcoming structural reorganisation.

## Guiding Principles

- Preserve backwards compatibility where feasible; provide clear migration utilities when breaking changes are unavoidable.
- Modularise by **domain** rather than implementation detail: ingestion/munging, distribution fitting, sampling, synthetic forest generation, and inventory simulation.
- Ensure every module exposes well-documented public APIs and optional CLI entry points.
- Keep cross-module dependencies explicit (import from and depend on `nemora.core` types/interfaces rather than circular imports).

## Proposed Top-Level Layout

```
src/nemora/
    core/            # Shared types, validation, utilities
    distributions/   # Central distribution registry + metadata shared across modules
    ingest/          # Data munging / ETL flows for raw inventory datasets
    distfit/         # Distribution inference & grouped estimators (build to alpha first)
    sampling/        # PDF/CDF inversion, bootstrap, Monte Carlo helpers
    synthforest/     # Synthetic landscape + stem generation engines
    simulations/     # Inventory collection simulators (plots, remote sensing, etc.)
    cli/             # Subcommand entry points (Python API remains primary interface)
```

## High-Level Workstreams

1. **Core scaffolding**
   - [x] Introduce `nemora.core` with shared types (e.g., `InventorySpec`), seed utilities, logging/config plumbing.
   - [x] Define module-level conventions (public API, config schema, entry points).

2. **Central distributions registry (`nemora.distributions`)**
   - [x] Extract canonical distribution metadata/registration logic from current code.
   - [x] Expose shared APIs so ingest, distfit, sampling, and synthforest can consume consistent definitions.
   - [x] Document extension points for user-supplied distributions.

3. **Distribution fitting (`nemora.distfit`) – Alpha delivered**
   - [x] Move existing fitting, grouped estimators, and mixture utilities into dedicated subpackage.
   - [x] Wire both Python API and CLI subcommands to the new namespace without breaking current usage.
   - [x] Expand unit tests/CI around grouped EM, fixtures, CLI regression, and ensure coverage reports run under new module name.
   - [x] Publish alpha documentation (module overview + API reference) and update README.

4. **Ingestion / ETL (`nemora.ingest`) – Phase 2 kickoff**
   - [x] Audit current scripts for reusable ETL logic (manifest generator, CLI wiring).
   - [x] Design `DatasetSource`, `RecordBatch`, `TransformPipeline` abstractions aligned with `nemora.core`.
   - [x] Implement key connectors (BC FAIB, FIA, etc.), add CLI helpers, and unit/integration tests against sample raw datasets. *(FAIB + FIA connectors, CLIs, and nightly integration coverage are live; see ingest outline for remaining benchmarking tasks.)*
   - [x] Verify FTP access to FAIB PSP/non-PSP datasets and capture download instructions/DataLad strategy (caching helpers + env-gated integration test).
   - [x] Parse FAIB PSP/non-PSP data dictionaries (XLSX) and surface schema metadata for ingest docs/tests.
   - [x] Flesh out FAIB ingest pipeline per `notes/ingest_pipeline_outline.md` (fetch, transform, output), including Parquet manifest adoption guidance + ingest benchmarking metrics surfaced in docs/nightly runs. *(Follow-up: automate ingest-benchmark trend capture.)*

5. **Sampling engine (`nemora.sampling`)**
   - [x] Catalogue existing sampling utilities (mixtures, truncated normals, etc.) and migrate next.
   - [x] Provide PDF → CDF inversion (analytic + numeric), bootstrap/Monte Carlo generators, and integrate with the central distribution registry.
   - [ ] Benchmark accuracy and ensure compatibility with `distfit` outputs. *(See `notes/sampling_module_plan.md` for detailed roadmap, including BootstrapResult adoption plan.)*

6. **Synthetic forest generation (`nemora.synthforest`)**
   - [ ] Define landscape/stem data models, leveraging `distributions` + `sampling`.
   - [ ] Implement stand attribute sampling, stem population generation, and optional high-resolution detail.
   - [ ] Deliver visualization/export tools and robust unit tests.

7. **Inventory simulation (`nemora.simulations`)**
   - [ ] Create interfaces for field and remote-sensing inventory simulations wired to `synthforest`.
   - [ ] Provide CLI workflows and integrate with sampling for uncertainty runs.
   - [ ] Build validation harness comparing simulated outputs to known ground truths.

8. **CLI + API coherence**
   - [ ] Maintain both Python API and Typer CLI across modules; no blanket deprecation of scripts, but encourage CLI subcommands for reproducibility.
   - [ ] Ensure CLI auto-discovers module entry points via extras where appropriate.

9. **Documentation & communication**
   - [ ] Produce module overview pages (how-to + API reference) as modules mature.
   - [ ] Update README/changelog with reorganised scope and module descriptions.
   - [ ] Draft notes highlighting the project’s early stage but rapid iteration plan (distfit alpha quickly, other modules phased later).
   - [x] Flesh out ingest API narrative (overview of DatasetSource/TransformPipeline, CLI cross-links, dataset helper summaries).

10. **Testing & CI strategy**
    - [ ] Expand unit/integration tests per module; keep coverage gating distfit alpha milestone.
    - [ ] Maintain nightly/CI runs; add module-specific coverage tracking as new components land.
    - [x] Nightly FAIB/FIA ingest integration workflow exercising live downloads with environment-gated pytest selection.
        - [x] Configure `.github/workflows/nightly-ingest.yml` to run on a nightly cron and manual dispatch.
        - [x] Steps: set up Python, install project deps, export `NEMORA_RUN_FAIB_INTEGRATION=1` / `NEMORA_RUN_FIA_INTEGRATION=1`, execute `pytest` against `tests/test_ingest_faib.py::test_build_faib_dataset_source_integration` and `tests/test_ingest_fia.py::test_download_fia_tables_integration`.
        - [x] Enable automatic retries/log capture for transient network errors; failures surface via the Actions UI with auto-created issues for triage.
        - [x] Document local re-run instructions in `CONTRIBUTING.md` and link the workflow from the roadmap (Phase 2 ingest testing milestone).
        - [x] Monitoring policy: rely on GitHub email notifications generated by the auto-created failure issues (`nightly-ingest-failure` label); ensure maintainers watch the repo and enable workflow email alerts.

11. **Release milestones**
    - Alpha: `nemora.distributions` + `nemora.distfit` stabilised, docs/tests updated.
    - Beta: ingest + sampling modules added with CLI/API coverage.
    - v0.1.0: synthforest, simulations, and associated docs/tests in place.

## Dependencies & Sequencing Notes

- Project age (~2 days) means no backward-compatibility burden; move quickly while keeping tests green.
- Build `core` + central `distributions` first to avoid circular imports and ensure shared metadata.
- `distfit` alpha complete; results feed directly into upcoming sampling/ingest modules.
- `ingest` depends on `core`/`distributions` types; plan abstractions now that distfit is settled.
- `sampling` depends on the central registry and distfit outputs; schedule right after ingest scaffolding.
- `synthforest` and `simulations` layer on top of distfit + sampling and can follow after alpha milestones.
- Maintain strong unit testing and CI gating at each step.

## Open Questions

- Packaging for large synthetic artefacts (consider DataLad datasets/extras).
- Extras management for heavy dependencies (geospatial, simulation packages).
- Documentation structure for cross-module tutorials (single handbook vs per-module guides).

## Next Steps

1. [x] Consolidate distribution metadata (bounds/defaults/extras) into `nemora.distributions`, add registry helper docs/tests, and align top-level roadmap tasks with the new coverage.
2. [x] Extend the sampling roadmap: wire `BootstrapResult` outputs into synthforest/simulations and document CLI usage for the new integration controls (bootstrap payload helper + docs captured in `docs/howto/synthforest.md`).
   - [x] Document how ingest-generated Parquet manifests feed sampling flows, including numeric integration tuning (`docs/examples/faib_manifest_parquet.md`, `docs/howto/sampling.md`).
3. [x] Capture ingest benchmarking metrics (from `nemora ingest-benchmark` and nightly runs), decide how to surface trends, and document the workflow in README/CONTRIBUTING for ongoing monitoring. *(Nightly workflow summaries + threshold enforcement now publish tables to the job summary and failure issues; README/CONTRIBUTING explain how to rerun locally and interpret the data.)*
   - [x] Promote FAIB manifest Parquet adoption from optional to default so downstream notebooks/tests can rely on the columnar artifact (`nemora faib-manifest` writes CSV+Parquet unless `--no-parquet` is passed).
4. [x] Build registry inspection tooling:
   - [x] Expose a Python helper (`list_registry_metadata`) that returns each distribution’s bounds/defaults/extras for downstream modules/tests.
   - [x] Add a `nemora registry --describe/--show-metadata/--json` CLI path that prints the helper output so contributors can audit plugin registrations.
   - [x] Extend `tests/test_registry.py` (and CLI smoke tests) to cover the helper/CLI plus plugin edge cases; ensure failures remain descriptive.
   - [x] Update docs/README to explain how to inspect the registry (linking to the helper, CLI command, and YAML/entry-point sections).
5. [x] Automate ingest benchmark telemetry collection:
   - [x] Wire the `ingest-benchmark --report-path` JSONL output into the nightly workflow (persist as artifact + attach issue snippet on failure).
   - [x] Summarise rolling metrics in docs/CHANGE_LOG so trend shifts are visible, and define alert thresholds for future automation. *(Summary Markdown/text artifacts land under `reports/`, `INGEST_BENCHMARK_AVG_THRESHOLD` enforces a 3.0s ceiling, and README/CONTRIBUTING/docs capture the behavior.)*
6. [x] Expose synthforest bootstrap consumption through the CLI/tests once helpers stabilize (e.g., `nemora sampling describe-bootstrap`), then document the workflow in README + notes.
   - [x] Prototype a Typer subcommand that loads a stand table (CSV/Parquet), auto-fits or parses parameters, and renders the metadata via `nemora.synthforest.helpers`.
   - [x] Add pytest coverage for the CLI + helper plumbing (unit + CLI smoke test).
   - [x] Update README + docs/howto/synthforest.md with CLI usage, JSON examples, and troubleshooting notes.
7. [x] Publish Sphinx docs via GitHub Pages (follow the FHOPS pattern for parity).
   - [x] Mirror the `UBC-FRESH/fhops` workflow (`.github/workflows/ci.yml`) so CI now installs doc deps, runs `sphinx-build -b html docs _build/html -W`, stages `_build/html` under `tmp/pages/.nojekyll`, and uploads it with `actions/upload-pages-artifact`.
   - [x] Add a `deploy-docs` job gated on `main`, configured with the `github-pages` environment plus `pages: write` / `id-token: write` permissions, so docs auto-deploy after CI succeeds (mirrors FHOPS `deploy-pages` job).
   - [x] Keep room for telemetry/benchmark artifacts alongside the HTML output by staging everything under `tmp/pages/` before publishing; extend later if ingest telemetry needs its own mini-site.
