# Nemora Readiness Roadmap

This roadmap tracks Nemora’s expansion from a distribution-fitting prototype into a modular
meta-package. It is intentionally aggressive—fit is sprinting to alpha—while other modules
will land in sequenced phases. The plan is updated alongside daily work; if something feels out of
date, check `notes/nemora_modular_reorg_plan.md` for the latest detail.

## Phase 0 — Foundations ✅ (complete)
- Repository scaffolding, licensing, CI harness.
- Initial documentation + contribution standards.
- Core refactor: renamed project to Nemora, bootstrapped `nemora.core`, centralised distribution
  registry, migrated fitting stack into `nemora.fit`, added compatibility shims.

## Phase 1 — Distribution Fitting Alpha 🚧 (in progress, target: next 1–2 days)
- [x] Finalise `nemora.fit` alpha API (grouped EM, mixtures, goodness-of-fit).
- [x] Expand fit unit tests (fixtures, CLI regressions, coverage gating).
- [x] Publish module overview + API reference; update README/CLI help.
- [x] Verify notebooks/examples reference the new namespace.
- [x] Cut changelog entry announcing fit alpha and note breaking import changes.

## Phase 2 — Core Module Expansion (sequenced after fit alpha)
- `nemora.distributions`
  - [x] Document extension points, add user-facing registry helpers.
  - [x] Move remaining distribution metadata (bounds, defaults) from ad-hoc code.
- `nemora.sampling`
  - [x] Implement PDF→CDF inversion (analytic + numeric).
  - [x] Provide bootstrap / Monte Carlo sampling utilities & tests.
  - [x] Integrate mixture helpers with fit outputs.
- `nemora.ingest`
  - [x] Design abstraction for raw inventory sources (`DatasetSource`, `TransformPipeline`).
  - [x] Port existing scripts (HPS dataset prep) into pipelines.
  - [x] Add CLI helpers for fetching / transforming reference datasets.
  - [x] Publish ingest how-to updates covering FAIB/FIA workflows.
  - [x] Add regression coverage for FAIB manifest + pipeline orchestration.
- `nemora.synthesis`
  - [x] Phase 0 research + scaffolding (CJFR/rlandscape + FLG requirements, module/tests/docs landing zones).
  - [x] Phase 1 kickoff: deterministic seed generator covering all four point processes plus CJFR hole/merge editing with exporter-friendly metadata.
  - [x] Phase 1 continuation: Voronoi clipping + target metric reporting (n, CV, μ_d, σ_d) wired into exporters/CLI, including convex GeoJSON masks, multi-mask overlays, and raster keep/exclude modifiers.
  - [x] Phase 1 deterministic layouts: hex-packed grids, imported point sets, and GeoJSON centroid placement exposed via `SeedLayoutConfig` and the CLI (`--layout hex|imported|geojson`).
  - [x] Implement stand attribute sampling (template helper + CLI manifest export; DBH integration still pending).
  - [x] Provide export & visualisation helpers (GeoJSON stand exporter and CLI; raster helpers future work).
- `nemora.simulation`
  - [ ] Create interfaces for plot-based and remote-sensing simulations.
  - [ ] Integrate with synthesis outputs and sampling utilities.
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
   - [x] Inventory bounds/defaults/extras scattered across fit, sampling, and ingest helpers; relocate the metadata into `src/nemora/distributions` and surface it through helper APIs.
   - [x] Add a CLI/`nemora registry` subcommand (and Python helper) that prints the registered bounds/defaults/extras so contributors can inspect metadata without diving into code.
   - [x] Expand regression tests covering the new helper/CLI output (including plugin registration coverage) and ensure failure modes stay readable.
   - [x] Publish an updated registry reference (docs + README) connecting the Python helpers, CLI inspection, YAML configs, and entry-point plugins.
2. **Sampling module adoption**
   - [x] Wire `BootstrapResult` into upcoming synthesis/simulation plans and document how downstream modules consume its metadata (bootstrap payload helper + docs).
   - [x] Add numerical accuracy tests that exercise the trapezoid/Simpson/quad integration modes against SciPy references.
   - [x] Extend docs/examples so sampling workflows demonstrate ingest-produced Parquet manifests and the configurable numeric integration settings.
   - [x] Prototype synthesis/simulation adapters that consume the new helper (CLI wiring + regression tests).
3. **Ingest monitoring & benchmarking**
   - [x] Capture `nemora ingest-benchmark` runtime stats (CLI + nightly workflow) and surface the trend in docs or CHANGE_LOG for visibility.
   - [x] Document the nightly FAIB/FIA workflow rerun + notification process in `CONTRIBUTING.md` (plus the new benchmark summary/threshold automation) so contributors can verify the job locally.
   - [x] Evaluate whether manifest Parquet adoption should become the default artifact once benchmarks confirm no downstream regressions (CLI now writes CSV + Parquet by default with `--no-parquet` opt-out).
4. **Module naming alignment**
   - [x] Rename the distribution-fitting stack under `nemora.fit` while preserving compatibility shims/CLI aliases so downstream docs/tests stay stable during the transition.
   - [x] Rename `nemora.synthforest` to `nemora.synthesis`, folding forest/stand/tree-level generators under the clearer namespace and updating planning docs/notes accordingly.
   - [x] Announce the breaking rename in README/CHANGE_LOG once the shims/tests/CLI wiring are proven on the `rename-modules-plan` branch (ready to merge into `main`).
5. **Synthesis Phase 1 — tessellation prototype**
   - [x] Upgrade `tessellation.generate_seed_points` so it produces a `VoronoiSeedResult` with process-mix counts plus CJFR hole/merge bookkeeping, wired into docs/tests.
   - [x] Persist the metadata via exporters + CLI plumbing so seed recipes become first-class artifacts.
  - [x] Extend Phase 1 with Voronoi clipping + target metric reporting (`n`, `CV`, `μ_d`, `σ_d`) benchmarked against CJFR/rlandscape fixtures and add deterministic layout controls (hex grids + imported/geojson points) to both the API and CLI.
  - [x] Layer in physiographic modifiers (raster-informed masks, multi-mask combos) once the deterministic layout plumbing stabilises.
  - [x] Bootstrap stand attribute sampling scaffolding (template loader + stochastic sampler); integrate with ingest-fed manifests next.
6. **Synthesis Phase 2 — stand bootstrap linking**
   - [x] Ship the stand→bootstrap manifest helper + CLI so sampled attributes can reference DBH payloads exported via `sampling-export-bootstrap-dbh` (plan parser, manifest writer, docs/tests).
   - [x] Thread the manifest into polygon exporters/GeoJSON so each stand feature carries a `stand_id`, `bootstrap_id`, and metadata preview for downstream tree generation.
   - [x] Outline the analytic (parameter-driven) pathway alongside bootstrap payloads and document the end-to-end workflow (seed recipe → stand templates → bootstrap manifest → tree synthesis) in `docs/howto/synthesis.md`.

## Backlog & Ideas
- [ ] Investigate GPU acceleration for large tally batches.
- [ ] Explore Bayesian fitting backends (PyMC/NumPyro).
- [ ] Interactive visualisation tools for PDF comparisons.
- [ ] Integration with FHOPS web dashboards.
- [ ] Consider optional DataLad datasets for synthetic artefacts.
