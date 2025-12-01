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
  - [x] Document extension points, add user-facing registry helpers.
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
1. **Distribution registry hardening**
   - [x] Inventory bounds/defaults/extras scattered across distfit, sampling, and ingest helpers; relocate the metadata into `src/nemora/distributions` and surface it through helper APIs.
   - [ ] Add a CLI/`nemora registry` subcommand (and Python helper) that prints the registered bounds/defaults/extras so contributors can inspect metadata without diving into code.
   - [ ] Expand regression tests covering the new helper/CLI output (including plugin registration coverage) and ensure failure modes stay readable.
   - [ ] Publish an updated registry reference (docs + README) connecting the Python helpers, CLI inspection, YAML configs, and entry-point plugins.
2. **Sampling module adoption**
   - [ ] Wire `BootstrapResult` into upcoming synthforest/simulation plans and document how downstream modules consume its metadata.
   - [x] Add numerical accuracy tests that exercise the trapezoid/Simpson/quad integration modes against SciPy references.
   - [ ] Extend docs/examples so sampling workflows demonstrate ingest-produced Parquet manifests and the configurable numeric integration settings.
3. **Ingest monitoring & benchmarking**
   - [ ] Capture `nemora ingest-benchmark` runtime stats (CLI + nightly workflow) and surface the trend in docs or CHANGE_LOG for visibility.
   - [ ] Document the nightly FAIB/FIA workflow rerun + notification process in `CONTRIBUTING.md` so contributors can verify the job locally.
   - [ ] Evaluate whether manifest Parquet adoption should become the default artifact once benchmarks confirm no downstream regressions.

## Backlog & Ideas
- [ ] Investigate GPU acceleration for large tally batches.
- [ ] Explore Bayesian fitting backends (PyMC/NumPyro).
- [ ] Interactive visualisation tools for PDF comparisons.
- [ ] Integration with FHOPS web dashboards.
- [ ] Consider optional DataLad datasets for synthetic artefacts.
