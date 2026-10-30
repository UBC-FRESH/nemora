# Nemora Readiness Roadmap

This roadmap tracks Nemora’s expansion from a distribution-fitting prototype into a modular
meta-package. It is intentionally aggressive—fit is sprinting to alpha—while other modules
will land in sequenced phases. The plan is updated alongside daily work; if something feels out of
date, check `notes/nemora_modular_reorg_plan.md` for the latest detail.

## Issue Tracker Map

| Phase | Parent issue | Branch | Status |
| --- | --- | --- | --- |
| P6 | #15 | `feature/p6-plot-lidar-interfaces` | complete |
| P7 | #20 | `feature/p7-research-forest-validation` | complete |
| P8 | #26 | `feature/p8-validation-refinement` | complete |
| P9 | #31 | `feature/p9-cross-forest-handoff` | complete |
| P10 | #36 | `feature/p10-calibration-handoff` | complete |
| P11 | #41 | `feature/p11-multi-block-readiness` | active |

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

## Phase 5 — Uncertainty, Sampling, and Synthetic Validation

Status: complete (P5.1-P5.4 complete).
Branch: `feature/p5.uncertainty-sampling`; split PRs:
`feature/p5.1-curve-uncertainty`, `feature/p5.2-ws4-fixture`,
`feature/p5.3-ingest-schema`, and `feature/p5.4-sgsr-wrapper`.

- [x] P5.1 Curve uncertainty and a 90% band for one family (`#6`).
  - [x] Add `nemora.uncertainty` with ratio fitting, multiplicative bands,
    coverage checks, and a validation-table reader.
  - [x] Fit TSA46 `CWH_HW` height ratios; holdout coverage 87.6%.
- [x] P5.2 Synthetic landscape fixture with known properties (`#8`).
  - [x] Generate a seeded Voronoi landscape and placeholder stems.
  - [x] Export WS4 recipe, fragments, schedule, truth table, and metadata.
  - [x] Run WS4 build/simulate and reproduce the expected volume within
    `1e-6 m3`.
  - [x] Add deterministic, truth-table, missing-seed, and WS4 smoke tests.
  - [x] Document the realism limits and regeneration command.
- [x] P5.3 Shared inventory ingest schema and one migrated path (`#9`).
  - [x] Publish `nemora.inventory.v1` in `nemora.ingest.schema`.
  - [x] Add a validator and release reader/writer.
  - [x] Migrate HPS through the schema-backed adapter while keeping the legacy
    per-plot CSV and manifest bytes unchanged.
  - [x] Add byte-level HPS fixtures, schema failure tests, and the nightly check.
  - [x] Document the contract, HPS migration, and remaining path order.
- [x] P5.4 sgsR structurally guided sampling wrapper (`#10`).
  - [x] Record the R-bridge decision: sgsR is GPL-3 and Nemora is MIT.
  - [x] Add `nemora.sampling.sgsr`, the bundled R bridge, and the
    `nemora sampling-sgsr` CLI.
  - [x] Return allocation, samples, structural strata, and provenance metadata.
  - [x] Add reproducibility, bad-input, constant-metric, and missing-R tests.
  - [x] Add the deterministic structural-metric example and docs.

Acceptance: the band is emitted with a recorded coverage check; the synthetic
fixture is deterministic and runs through ws4 build/simulate; the ingest schema
rejects a missing key and reproduces the existing output; the sampling wrapper
is reproducible; each child has a test that can fail and a docs note.

## Phase 6 — Plot resolution and LiDAR observation interfaces

Parent issue: #15
Branch: `feature/p6-plot-lidar-interfaces`
Status: complete

Goal: make Nemora the home for generic plot resolution/materialization and
LiDAR observation sampling while keeping raw LiDAR production in FEMIC/WS4.
See `notes/adr/0001-plot-lidar-boundaries.md` and
`notes/phase6_plot_lidar_interfaces.md`.

- [x] P6.1 Record the plot/LiDAR boundary and API plan (#16)
  - [x] Audit the existing Nemora, WS4, and FEMIC surfaces.
  - [x] Write the ADR and phase plan.
  - [x] Add the Phase 6 roadmap and issue links.
- [x] P6.2 Add plot resolution and tally materialization helpers (#17)
  - [x] Add `nemora.ingest.plots` with mapped-table resolution.
  - [x] Add opt-in tally materialization with deterministic tree IDs.
  - [x] Add tests and inventory-schema documentation.
- [x] P6.3 Add optional LiDAR raster and canopy metrics adapter (#18)
  - [x] Add `nemora.lidar` raster sampling and canopy metrics.
  - [x] Add synthetic raster tests and optional-dependency guidance.
  - [x] Add API/how-to documentation and the `lidar` extra.

## Phase 7 — Research-forest plot and LiDAR validation

Parent issue: #20
Branch: `feature/p7-research-forest-validation`
Status: complete

Goal: produce a reviewed, reproducible validation package for one or both research
forests using public FAIB plots and prepared FEMIC/LidarBC rasters, without claiming a
real tree map. See `notes/phase7_research_forest_validation.md`.

- [x] P7.1 Select and resolve research-forest FAIB plots (#21)
  - [x] Define forest/BEC/age/proximity filters and a held-out split.
  - [x] Resolve public plot, visit, and tree tables.
  - [x] Record provenance, licence, checksums, and a selection manifest.
  - [x] Select Gavin Lake as the first target: 20 plots, 15 train / 5 hold-out.
- [x] P7.2 Build and validate plot-conditioned mature-stand inventories (#22)
  - [x] Build training and hold-out `nemora.inventory.v1` releases.
  - [x] Generate one seeded synthetic stand per target forest.
  - [x] Report DBH, height, basal area, stems/ha, and species-mix errors.
- [x] P7.3 Validate canopy and terrain metrics from prepared rasters (#23) — complete with partial coverage
  - [x] Identify public LidarBC DEM/DSM products and provenance through FEMIC discovery.
  - [x] Materialize 16 DEM/DSM pairs and sample plots plus the synthetic stand through `nemora.lidar`.
  - [x] Report native-resolution canopy and slope metrics with partial plot coverage.
- [x] P7.4 Write the validation report and limitations (#24)
  - [x] Assemble metrics, uncertainty, and provenance tables.
  - [x] Write the report with explicit observed/fitted/synthetic labels.
  - [x] State what the validation does not establish.

## Phase 8 — Validation refinement and LiDAR coverage expansion

Parent issue: #26
Branch: `feature/p8-validation-refinement`
Status: complete

Goal: improve the Gavin Lake stand model and close the LidarBC coverage gaps
without weakening provenance or making operational claims. See
`notes/phase8_validation_refinement.md`.

- [x] P8.1 Compare refined stand models on held-out Gavin Lake plots (#27)
  - [x] Implement species-stratified DBH fitting.
  - [x] Evaluate a plot-level empirical alternative; mixture remains future work.
  - [x] Height refinement: selected empirical candidate preserves observed DBH/height pairs; species-specific parametric height remains future work.
  - [x] Select the empirical candidate by held-out performance.
- [x] P8.2 Expand LidarBC coverage and resolve the remaining raster gaps (#28)
  - [x] Re-query the catalogue for the four missing plots at 1/3/10/25 km.
  - [x] Resolve the nodata plot with an already-materialized alternate tile; no new download required.
  - [x] Explain the four residual coverage gaps and resolve the nodata plot.
- [x] P8.3 Rerun validation, update the report, and close out (#29)
  - [x] Rerun the stand validation with the selected empirical model and use the expanded LiDAR coverage.
  - [x] Report before/after metrics and uncertainty.
  - [x] Write `notes/reports/phase8_validation.md` and the phase close-out report.

## Phase 9 — Cross-forest validation and FHOPS handoff

Parent issue: #31
Branch: `feature/p9-cross-forest-handoff`
Status: complete

Goal: validate Malcolm Knapp with an explicit regional pool and hand the selected
Nemora stand to FHOPS through a tested tree-inventory export. See
`notes/phase9_cross_forest_handoff.md`.

- [x] P9.1 Validate Malcolm Knapp with an explicit regional pool (#32)
  - [x] Select local and 43-plot regional CWHdm pool.
  - [x] Record 50–100 km ring, filters, and local hold-out.
  - [x] Run empirical transfer validation; report negative DBH/height transfer.
- [x] P9.2 Export the selected stand to FHOPS with round-trip tests (#33)
  - [x] Confirm the FHOPS tree inventory fields and units.
  - [x] Implement `nemora.export.export_fhops_tree_inventory`.
  - [x] Add a round-trip validation test through the FHOPS loader.
- [x] P9.3 Run the FHOPS tree-scale handoff and write the close-out report (#34)
  - [x] Ran FHOPS 1.2.0 on 1,978 exported trees: 530.54 m³, 0 unreachable.
  - [x] Assembled the Gavin/Malcolm Knapp comparison.
  - [x] Wrote `notes/reports/phase9_validation.md` and the phase close-out report.

## Phase 10 — Calibration and handoff hardening

Parent issue: #36
Branch: `feature/p10-calibration-handoff`
Status: complete

Goal: improve Malcolm Knapp transfer and harden the FHOPS scenario handoff
without weakening provenance. See `notes/phase10_calibration_handoff.md`.

- [x] P10.1 Stratify the CWHdm pool and rerun Malcolm Knapp (#37)
  - [x] Inspected age/height distributions; age-first rule selected.
  - [x] Recorded the 80–110 age rule before scoring.
  - [x] Reran transfer; DBH/height/KS improved, density/BA worsened.
- [x] P10.2 Harden FHOPS scenario registration and block reconciliation (#38)
  - [x] Automate scenario pin/checksum checks.
  - [x] Reconciled block `work_required` to 530.543 m³; warning removed.
  - [x] Added `tests/test_phase10_fhops_scenario.py`.
- [x] P10.3 Rerun cross-forest system comparison and close out (#39)
  - [x] Ran calibrated and reference scenarios.
  - [x] Added calibrated-vs-reference rate sensitivity.
  - [x] Wrote `notes/reports/phase10_validation.md` and the phase close-out report.

## Phase 11 — Multi-block validation and release readiness

Parent issue: #41
Branch: `feature/p11-multi-block-readiness`
Status: active

Goal: build a reproducible multi-block/multi-landing scenario set, add an
independent yield check, run sensitivity, and prepare release/readiness docs.
See `notes/phase11_multi_block_readiness.md`.

- [x] P11.1 Build the multi-block/multi-landing scenario set (#42)
  - [x] Registered Gavin B01/L1 and Malcolm Knapp B02/L2 scenarios.
  - [x] Wrote `notes/reports/phase11_scenario_catalog.json`.
  - [x] Added `tests/test_phase11_scenario_catalog.py`.
- [ ] P11.2 Add independent yield validation and handoff hardening (#43)
  - [ ] Choose and document an independent volume estimator.
  - [ ] Compare estimate, work_required, and FHOPS processed volume.
  - [ ] Add a regression test.
- [ ] P11.3 Run sensitivity set and write the multi-block report (#44)
  - [ ] Run calibrated/reference rates and truck-count sensitivity.
  - [ ] Write the multi-block report.
  - [ ] Record warnings and limits.
- [ ] P11.4 Prepare release/readiness docs and close out (#45)
  - [ ] Update release/readiness docs.
  - [ ] Write the close-out report.
  - [ ] Queue the phase PR.

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
   - [x] Wire the manifest into DBH sampler helpers so future tree generators can consume bootstrap + analytic payloads via a single API.
7. **Phase 2 typing cleanup**
   - [x] Resolve pandas typing mismatches in FIA/FAIB ingest + HPS dataprep and remove the temporary mypy overrides.
   - [x] Align the clustered bootstrap gallery fixture with the fixture-provided vectors to keep synthesis regression tests deterministic.
8. **P5.3 shared inventory schema**
   - [x] Publish `nemora.inventory.v1`, the validator, and the release reader/writer.
   - [x] Migrate HPS and preserve the legacy per-plot CSV/manifest bytes.
   - [x] Add byte-level fixtures, schema failure tests, nightly coverage, and the migration note.
   - [ ] Migrate the FAIB stand table next, then FIA, then source-specific tree readers.
9. **P5.2 synthetic WS4 fixture**
   - [x] Generate a seeded Voronoi landscape with stand and stem attributes.
   - [x] Export the WS4 recipe, fragments, empty schedule, truth table, and metadata.
   - [x] Reproduce the expected volume through WS4 build/simulate.
   - [x] Lock the fixture with deterministic, truth-table, missing-seed, and WS4 smoke tests.
   - [x] Document the realism limits and regeneration command.
10. **P5.4 sgsR sampling bridge**
   - [x] Record the R-bridge choice and the GPL-3/MIT boundary.
   - [x] Add the Python wrapper, bundled R script, and `sampling-sgsr` CLI.
   - [x] Return allocation, sample coordinates, structural strata, and versions.
   - [x] Add the deterministic metric example and real/skipped integration tests.
   - [x] Document installation, reproducibility, and maintenance cost.
11. **P6 plot/LiDAR boundary**
   - [x] Record the ADR and phase plan.
   - [x] Add mapped plot release resolution and tally materialization.
   - [x] Add optional raster sampling and canopy metrics.
   - [x] Keep LiDAR catalog/materialization/terrain/ITD production in FEMIC/WS4.
12. **P7 research-forest validation**
   - [ ] Select and resolve public FAIB plots with a deterministic hold-out split.
     - [x] Selected Gavin Lake: 20 mature plots, 15 train / 5 hold-out.
     - [x] Resolve tree detail and build the canonical pilot release.
   - [x] Build and validate plot-conditioned mature-stand inventories.
   - [x] Materialize public LidarBC DEM/DSM and validate CHM/slope context through `nemora.lidar`; report missing plot coverage.
   - [x] Publish the limitations-first validation report.

## Backlog & Ideas
- [ ] Investigate GPU acceleration for large tally batches.
- [ ] Explore Bayesian fitting backends (PyMC/NumPyro).
- [ ] Interactive visualisation tools for PDF comparisons.
- [ ] Integration with FHOPS web dashboards.
- [ ] Consider optional DataLad datasets for synthetic artefacts.
