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
   - [ ] Design `DatasetSource`, `RecordBatch`, `TransformPipeline` abstractions aligned with `nemora.core`.
   - [ ] Implement key connectors (BC FAIB, FIA, etc.), add CLI helpers, and unit/integration tests against sample raw datasets. *(FAIB fetch + manifest CLI landed; FIA prototype helper committed; next: fixtures + CLI automation.)*
   - [x] Verify FTP access to FAIB PSP/non-PSP datasets and capture download instructions/DataLad strategy (caching helpers + env-gated integration test).
   - [x] Parse FAIB PSP/non-PSP data dictionaries (XLSX) and surface schema metadata for ingest docs/tests.
   - [ ] Flesh out FAIB ingest pipeline per `notes/ingest_pipeline_outline.md` (fetch, transform, output).

5. **Sampling engine (`nemora.sampling`)**
   - [x] Catalogue existing sampling utilities (mixtures, truncated normals, etc.) and migrate next.
   - [x] Provide PDF → CDF inversion (analytic + numeric), bootstrap/Monte Carlo generators, and integrate with the central distribution registry.
   - [ ] Benchmark accuracy and ensure compatibility with `distfit` outputs. *(See `notes/sampling_module_plan.md` for detailed roadmap.)*

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

10. **Testing & CI strategy**
   - [ ] Expand unit/integration tests per module; keep coverage gating distfit alpha milestone.
   - [ ] Maintain nightly/CI runs; add module-specific coverage tracking as new components land.

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

1. Finalise `DatasetSource` / `TransformPipeline` interface proposal and circulate for feedback.
2. Prototype sampling helpers (PDF→CDF inversion, bootstrap configuration) and capture design decisions.
3. Draft ingest fixture plan (BC FAIB/FIA samples), including automated tests for PSP/non-PSP FTP downloads and data dictionary parsing.
