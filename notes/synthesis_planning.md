# `nemora.synthesis` module planning notes


## Rlandscape package

We can base our methods on the methods described in `reference-papers/2012-a-voronoi-tessellation-based-approach-to-generate-hypothetical-forest-landscapes.pdf`.

See R package:
https://rpubs.com/gpassolt/rlandscape

The R package has not been updated in 13 years, so we can safely assume it is "dead code". We should basically scrape the CJFR paper and R package code for details, and just reimplement all of this in Python code (crediting the original package authors for their contributions and code and such, but then possibly expanding on this).

## FLG package

Also have a look at the FLG project documentation under `reference-papers/flg`. Basically I wrote this way back. The Rlandscape package probably has a better implementation, but maybe there is something in the FLG documentation or whatnot worth salvaging (my guess is no, but work carefull scraping through and documenting)

## Data dependencies & tooling

Many synthesis workflows lean on DataLad-provisioned reference datasets (FAIB/FIA manifests, raster
fixtures, etc.). When setting up a new workstation:

1. Install `git-annex` first (package managers work best: NeuroDebian/apt on Ubuntu, `brew install git-annex` on macOS, or `datalad-installer git-annex` on any platform). DataLad will not function without it.
2. Install DataLad with the full extras so Python deps land correctly: `pip install datalad[full]`
   (use `--user` or `python3 -m pip` if needed). This only handles Python requirements.
3. Optional but recommended: use `pip install datalad-installer` and rerun `datalad-installer git-annex`
   when upgrading so non-Python binaries stay current across platforms.

Keep `git-annex`/`datalad` on your `PATH` before attempting to fetch synthesis inputs; the ingest and
sampling notes assume the tooling is ready to go.

## Multi-phase implementation plan

### Phase 0 — Research & design scaffolding
- [x] **CJFR + Rlandscape requirements dump**
  Catalogue requirements from the CJFR paper (`reference-papers/2012-a-voronoi-tessellation-based-approach-to-generate-hypothetical-forest-landscapes.pdf`) and the legacy R source (rpubs link) into this document (algorithms, inputs, stochastic controls, outputs).
  - [x] Summarise core algorithm steps (seed point generation, Voronoi clipping, stand attribute assignment) with explicit page references for each mechanism.
  - [x] Extract data requirements (input rasters, stand tables, configuration parameters), calling out which inputs are optional vs. mandatory.
  - [x] Capture stochastic controls (seed handling, distribution selection, reproducibility guarantees) and cite how Rlandscape handles them today.
- [x] **FLG documentation review**
  Review the FLG documentation (`reference-papers/flg/*`) to capture complementary insights (stand attribute templates, historical assumptions) and flag what, if anything, we will reuse.
  - [x] List reusable artefacts (attribute schemas, calibration datasets) and explicitly log any FLG features we intend to drop or modernise.
  - [x] Identify calibration/validation data mentioned in FLG docs that can seed regression tests or gallery notebooks.
- [x] **Module/test/doc scaffolding**
  Define the Python module skeleton in `src/nemora/synthesis/` (submodules for tessellation, canopy assignment, stand population, exporters) plus matching `tests/` scaffolding and doc stubs.
  - [x] Create placeholder modules (`tessellation.py`, `stands.py`, `exporters.py`) with TODO docstrings outlining responsibilities + dependencies.
  - [x] Add `tests/synthesis/` scaffolding (fixtures directory, smoke-test placeholders) plus sample seed data for future Voronoi tests.
  - [x] Expand `docs/howto/synthesis.md` + `docs/reference/synthesis.md` with section headings aligned to Phases 1‑3 so we know where upcoming content lands.
- [x] **Roadmap alignment**
  When the Phase 0 checklist is complete, update ROADMAP Phase 2 (`nemora.synthesis`) and `notes/nemora_modular_reorg_plan.md` to mark the research groundwork done and unblock Phase 1 implementation.

### Phase 1 — Landscape geometry & metadata
- [x] Implement Voronoi-based tiling mirroring Rlandscape behaviour:
  - [x] Deliver deterministic `VoronoiSeedResult` generation that mixes the four point processes and applies the CJFR hole/merge editing knobs, returning metadata for exporters/tests.
  - [x] Expose exporter/CLI plumbing so seed recipes (config + metadata) can be captured as JSON artifacts.
  - [x] Deterministic seed/control of initial plot centers (random, hex-packed, or imported points).
    - `SeedLayoutConfig` + `SeedLayoutMode` drive reproducible grids; CLI exposes `--layout hex`
      and `--layout imported --layout-points path/to/points.csv|json`.
    - Imported layouts require coordinates inside the unit box so Voronoi clipping remains stable;
      metadata records the layout type/source for downstream recipes/tests.
    - Added `--layout geojson` support so polygon centroids can seed deterministic layouts directly
      from GeoJSON feature collections.
  - [x] Boundary clipping + min/max polygon size constraints (CJFR metrics tracked via Voronoi polygons); convex GeoJSON mask clipping plumbs through CLI/exporters (physiographic rasters remain TODO).
  - [x] Optional slope/elevation raster support for physiographic modifiers.
    - Multiple vector overlays supported via repeatable `--mask-geojson`/`--mask-mode` pairs
      (clip/exclude), plus raster-based keep/exclude grids with threshold controls.
- [x] Attach stand-level attributes (species mix, age class, site index, crown closure) using probability surfaces or user-specified distributions; expose hooks for ingesting real inventory summaries.
  - [x] Initial scaffolding: `sample_stand_attributes` consumes templates (JSON or iterable records)
    and fills a requested area with vegetation/age-class assignments for downstream DBH sampling.
  - [x] CLI bridge: `nemora synthesis-sample-attributes` loads template JSON and emits stand manifests
    so docs/tests can sample attributes without bespoke scripts.
  - [x] Tessellation integration: `nemora synthesis-generate-seeds --include-polygons` plus
    `nemora synthesis-assign-stands` attach sampled attributes to Voronoi polygons and export GeoJSON.
- [x] Write regression tests comparing polygon statistics vs. Rlandscape reference runs (see `tests/fixtures/synthesis/reference_metrics.json` + tessellation regression test).

### Phase 2 — Stand tree-list synthesis
- [x] Integrate the sampling stack so each synthesized stand can draw DBH vectors via bootstrap or analytic sampling:
  - [x] Provide a stand→bootstrap manifest helper + CLI so sampled attributes can reference DBH payloads exported by `sampling-export-bootstrap-dbh` (plan JSON + Typer command + regression tests).
  - [x] Thread the manifest into stand exporters/CLI so GeoJSON features carry `stand_id`, `bootstrap_id`, and a metadata preview for downstream tree generators.
  - [x] Support both “synthetic from parameters” and “bootstrap from empirical tallies” modes (plan format + linker now accept analytic payload definitions alongside bootstrap JSON files).
  - [x] Ensure outputs align with existing `BootstrapResult` metadata contracts by threading the manifest into polygon exporters + upcoming tree generators.
  - [x] **DBH sampler helpers:** codify how manifests feed tree generators.
    - [x] Introduce a `StandDBHSampler` protocol/dataclass that exposes `draw_dbh(rng, count)` and returns both the sampled vector and provenance (`bootstrap_id`, distribution parameters, sample weights when applicable).
    - [x] Add a builder that consumes the manifest (`assignments`, `bootstraps`) and hydrates the correct sampler per `bootstrap_id`, loading `BootstrapResult` JSON only once (maybe via `BootstrapResult.from_dict`) and falling back to analytic sampling via `nemora.distributions` + numeric inversion when no bootstrap vectors exist. Builder likely lives in `nemora.synthesis.helpers` to keep CLI + future tree modules aligned.
    - [x] Capture the helper contract in docs/notes (inputs, expected metadata fields, deterministic seeding requirements) so tree placement + CLI wiring depend on a stable interface. Note: CLI should support `--bootstrap-manifest` + `--sampler-cache` for pre-loading payloads in long workflows.
    - [x] Provide regression fixtures mixing bootstrap + analytic payloads to prove deterministic draws (seeded RNG) and document failure modes (missing parameters, unsupported distributions).
- [x] Build composable pipelines to attach per-tree metadata:
  - [x] Spatial placement within polygons (Poisson, stratified by canopy layer, optional clustering).
    - [x] Define `TreePlacementConfig` (layout mode, stratification bands, optional min spacing) and a `place_trees(polygon, count, config, rng)` helper living in `synthesis.stands` or a new `stems` module.
    - [x] Add bootstrap-aware convenience: `place_trees_with_dbh(stand_feature, sampler, count=None, config=...)` that draws DBH from `StandDBHSampler` and pairs with coordinates.
    - [x] Provide deterministic seeding hooks so repeated exports remain reproducible.
  - [x] Crown metrics, biomass factors, bark thickness, etc., using ingest/sampling configs for consistent units.
    - [x] Define a lightweight `TreeAttributes` dataclass carrying DBH, height (placeholder), crown ratio, biomass, bark thickness; seed with simple allometry placeholders until ingest-derived models land.
    - [x] Wire attributes into GeoJSON/Parquet exporters with provenance (sampler type, distribution params).
  - [x] Tests verifying tree count / basal area per stand stays within configured tolerances plus property-based checks on allometric relationships.
    - [x] Regression fixtures: simple square polygon + analytic sampler to validate placement density and DBH summary stats.
    - [x] Property tests: no negative metrics, mean DBH roughly matches sampler mean, placement respects bounding box and min spacing when configured.
  - [x] CLI/export wiring: expose placement + attribute exports via `synthesis-export-trees` with RNG/spacing controls; document flags/workflow in how-to and cover via CLI regression.
- [x] Tests verifying tree count / basal area per stand stays within configured tolerances plus property-based checks on allometric relationships.
- [ ] Calibration/provenance follow-ups:
  - [ ] Swap placeholder allometry for ingest-derived coefficients and thread provenance into exporters.
  - [ ] Add a gallery/regression fixture combining analytic + bootstrap samplers (clustered mode) and document baseline metrics vs. CJFR/Rlandscape references.
  - [x] Document troubleshooting tips for `synthesis-export-trees` and capture attribute provenance in exported records.
  - [x] Capture a validation snapshot with current fixture metrics in `notes/synthesis_validation.md`.
  - [ ] Interim plan without ingest-derived coefficients:
    - [ ] Keep the current placeholder power laws but stamp a provenance version (e.g., `placeholder-v1`)
      on exported records.
    - [ ] Expose a simple override path for coefficients (env/CLI/config) so downstream experiments
      can inject provisional values without code edits.
    - [ ] Document the stopgap in `notes/synthesis_validation.md` and the how-to, including expected
      metric ranges and where to drop calibrated values once available.

### Phase 3 — Export, visualization, and CLI
- [ ] Add exporters for GeoJSON/GeoPackage (stands + tree points), CSV/Parquet tree lists, and lightweight rasters (canopy height, basal area density).
- [ ] Provide a `nemora synthesis generate-landscape` CLI:
  - [ ] Accept seeds/config files (YAML/JSON) describing tiling + stand recipes.
  - [ ] Optionally chain into the bootstrap inspection CLI so users can preview per-stand distributions before generation.
- [ ] Extend docs with a synthesis how-to that covers landscape generation, CLI usage, and integration with future simulations; include provenance/credit for Rlandscape.

### Phase 4 — Validation & extensions
- [ ] Benchmark outputs against historical Rlandscape/FLG examples (document deviations, add a gallery notebook).
- [ ] Hook into the upcoming `nemora.simulation` module so synthetic landscapes feed observation simulators (plot sampling, remote-sensing emulation).
- [ ] Explore advanced features: disturbance simulation (fire/harvest patches), time-stepped growth, integration with ingest benchmark telemetry for calibration.

### Phase 0 research notes (2025-12-05)

- **CJFR / rlandscape highlights**
  - Section “Generating a landscape” (pp. 79–82) defines four control metrics — number of management units (`n`), coefficient of variation for polygon areas (`CV`), and the mean/standard deviation of vertex degree (`μ_d`, `σ_d`). These map directly to Nemora metadata we can already compute from adjacency tables, so the synthesis module should emit these values for regression checks.
  - The seed-point strategy combines four point processes (uniform, clustering, simple sequential inhibition, lattice grid) with proportions `p_unif`, `p_clust`, `p_SSI`, `p_lat` plus process-specific parameters (cluster size/spread, inhibition distance, lattice resolution). Mixtures allow us to sweep from highly regular to highly variable mosaics.
  - Post-processing uses two tuning parameters: `p_H` (hole fraction) deletes polygons after tessellation to simulate rivers/voids, whereas `p_M` (merge fraction) collapses boundaries to create non-convex units and introduce right angles. Both operations inflate `n_tot` to hit the target `n` after deletions/merges. These controls inform the placeholder dataclasses in `synthesis.tessellation`.
  - Aspect ratio `a` and linear models linking target metrics to control parameters (Fig. 2) mean we eventually need a lightweight regression or lookup table that maps desired `{n, CV, μ_d, σ_d}` → seed config. For Phase 0 we simply record the dependency so the future API (`VoronoiSeedConfig`) exposes the same knobs.

- **FLG (Paradis & Richards 2001) takeaways**
  - FLG is raster-driven: users specify vegetation-type distributions plus age-class CDFs and patch-size Weibull parameters (`W_a`, `W_b`, `W_c`). This matches the data we already keep in sampling manifests, so `StandAttributeTemplate` is shaped around those tuples.
  - Patch growth uses a “mother cell + concentric layers” algorithm with randomised edge selection to avoid geometric bias. Sorting samples by descending patch size before allocation mitigates truncation when space runs out; we’ll mimic this ordering when we integrate bootstrap-driven DBH payloads.
  - Edge effects and merging identical neighbours are explicitly handled (oversized matrix, attribute checks). These behaviours translate into future post-processing hooks for the stands module.
  - Outputs include both raster and dissolved vector layers plus adjacency scripts, reinforcing that our exporter stubs should plan for GeoJSON/CSV outputs in tandem.
