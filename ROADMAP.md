# Nemora Readiness Roadmap

This roadmap tracks Nemora’s expansion from a distribution-fitting prototype into a modular
meta-package. It is intentionally aggressive—distfit is sprinting to alpha—while other modules
will land in sequenced phases. The plan is updated alongside daily work; if something feels out of
date, check `notes/nemora_modular_reorg_plan.md` for the latest detail.

## Phase 0 — Foundations ✅ (complete)
- Repository scaffolding, licensing, CI harness.
- Initial documentation + contribution standards.
- Core refactor: renamed project to Nemora, bootstrapped `nemora.core`, centralised distribution
  registry, migrated fitting stack into `nemora.distfit`, added compatibility shims.

## Phase 1 — Distribution Fitting Alpha 🚧 (in progress, target: next 1–2 days)
- [x] Finalise `nemora.distfit` alpha API (grouped EM, mixtures, goodness-of-fit).
- [x] Expand distfit unit tests (fixtures, CLI regressions, coverage gating).
- [x] Publish module overview + API reference; update README/CLI help.
- [x] Verify notebooks/examples reference the new namespace.
- [x] Cut changelog entry announcing distfit alpha and note breaking import changes.

## Phase 2 — Core Module Expansion (sequenced after distfit alpha)
- `nemora.distributions`
  - [ ] Document extension points, add user-facing registry helpers.
  - [ ] Move remaining distribution metadata (bounds, defaults) from ad-hoc code.
- `nemora.sampling`
  - [x] Implement PDF→CDF inversion (analytic + numeric).
  - [x] Provide bootstrap / Monte Carlo sampling utilities & tests.
  - [x] Integrate mixture helpers with distfit outputs.
- `nemora.ingest`
  - [x] Design abstraction for raw inventory sources (`DatasetSource`, `TransformPipeline`).
  - [x] Port existing scripts (HPS dataset prep) into pipelines.
  - [x] Add CLI helpers for fetching / transforming reference datasets.
  - [x] Publish ingest how-to updates covering FAIB/FIA workflows.
  - [x] Add regression coverage for FAIB manifest + pipeline orchestration.
- `nemora.synthforest`
  - [ ] Define landscape/stem data models.
  - [ ] Implement stand attribute sampling, stem population generation.
  - [ ] Provide export & visualisation helpers (GeoJSON, rasters).
- `nemora.simulations`
  - [ ] Create interfaces for plot-based and remote-sensing simulations.
  - [ ] Integrate with synthforest outputs and sampling utilities.
  - [ ] Model measurement noise / detection bias; deliver CLI workflows.

## Phase 3 — Quality & Release Readiness
- [ ] Harden testing (property-based tests, synthetic fixtures, regression suites).
- [ ] Add benchmarking harness for long-running fits / synthetic generation.
- [ ] Flesh out Sphinx documentation (per-module API, cross-cutting how-tos).
- [ ] Configure Read the Docs + link to GitHub releases.
- [ ] Establish release process (semver, changelog cadence).

## Phase 4 — Community & Deployment
- [ ] Prep v0.1.0 release notes / announcement.
- [ ] Publish package to PyPI; wire automation for tagged releases.
- [ ] Finalise CRAN submission plan for `nemorar`.
- [ ] Draft contributor guide, code of conduct, issue templates.
- [ ] Outreach (blog posts, mailing lists, working group updates).

## Detailed Next Steps Notes
1. **Phase 2 Kickoff — Ingest module design**
   - [x] Finalise `DatasetSource` / `TransformPipeline` abstractions in `notes/nemora_modular_reorg_plan.md`.
   - [x] Stub `nemora/ingest/__init__.py` and outline initial pipeline modules.
   - [x] Prepare BC FAIB fixture manifests for integration tests (`examples/faib_manifest`, CLI).
   - [x] Automate FAIB fetch → manifest → stand-table ETL flow (cache management, CLI integration test).
   - [x] Schedule FIA dataset scoping session once FAIB automation is stable (`notes/fia_ingest_scoping.md` drafted; HI sample downloaded to `data/external/fia/raw` for schema review).
   - [x] Trim FIA HI sample into test fixtures (`tests/fixtures/fia/`) and wire ingest tests to use them.
   - [x] Build FIA CLI/ETL workflow (state + filters) once fixtures/regression harness are in place.
   - [x] Expose FIA download helper via `DatasetSource` fetcher abstraction and integrate into ingest module planning.
   - [x] Document FIA CLI usage + caching guidance in `docs/howto/ingest.md`.
   - [x] Promote FAIB pipeline into `TransformPipeline` implementation and add CLI entry point.
2. **Sampling module prototypes**
   - [ ] Draft numeric/analytic PDF→CDF inversion helpers in notebooks.
   - [ ] Specify bootstrap/Monte Carlo API surface to align with distfit outputs.
   - [ ] Identify regression tests required for mixture integration.
3. **Documentation TODOs**
   - [ ] Update how-to guides as new modules land (`ingest`, `sampling`, etc.).
   - [ ] Add module API pages (placeholders present).
   - [ ] Highlight CLI + Python API parity intent.

## Backlog & Ideas
- [ ] Investigate GPU acceleration for large tally batches.
- [ ] Explore Bayesian fitting backends (PyMC/NumPyro).
- [ ] Interactive visualisation tools for PDF comparisons.
- [ ] Integration with FHOPS web dashboards.
- [ ] Consider optional DataLad datasets for synthetic artefacts.
