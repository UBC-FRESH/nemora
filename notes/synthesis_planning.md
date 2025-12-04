# `nemora.synthesis` module planning notes


## Rlandscape package

We can base our methods on the methods described in `reference-papers/2012-a-voronoi-tessellation-based-approach-to-generate-hypothetical-forest-landscapes.pdf`.

See R package:
https://rpubs.com/gpassolt/rlandscape

The R package has not been updated in 13 years, so we can safely assume it is "dead code". We should basically scrape the CJFR paper and R package code for details, and just reimplement all of this in Python code (crediting the original package authors for their contributions and code and such, but then possibly expanding on this).

## FLG package

Also have a look at the FLG project documentation under `reference-papers/flg`. Basically I wrote this way back. The Rlandscape package probably has a better implementation, but maybe there is something in the FLG documentation or whatnot worth salvaging (my guess is no, but work carefull scraping through and documenting)

## Multi-phase implementation plan

### Phase 0 — Research & design scaffolding
- [ ] Catalogue requirements from the CJFR paper (`reference-papers/2012-a-voronoi-tessellation-based-approach-to-generate-hypothetical-forest-landscapes.pdf`) and the legacy R source (rpubs link) into this document (algorithms, inputs, stochastic controls, outputs).
- [ ] Review the FLG documentation (`reference-papers/flg/*`) to capture complementary insights (stand attribute templates, historical assumptions) and flag what, if anything, we will reuse.
- [ ] Define the Python module skeleton in `src/nemora/synthesis/` (submodules for tessellation, canopy assignment, stand population, exporters) plus matching `tests/` scaffolding and doc stubs.

### Phase 1 — Landscape geometry & metadata
- [ ] Implement Voronoi-based tiling mirroring Rlandscape behaviour:
  - [ ] Deterministic seed/control of initial plot centers (random, hex-packed, or imported points).
  - [ ] Boundary clipping + min/max polygon size constraints, including optional shapefile/GeoJSON masks.
  - [ ] Optional slope/elevation raster support for physiographic modifiers.
- [ ] Attach stand-level attributes (species mix, age class, site index, crown closure) using probability surfaces or user-specified distributions; expose hooks for ingesting real inventory summaries.
- [ ] Write regression tests comparing polygon statistics vs. Rlandscape reference runs (small seeds stored under `tests/fixtures/synthesis`).

### Phase 2 — Stand tree-list synthesis
- [ ] Integrate the sampling stack so each synthesized stand can draw DBH vectors via bootstrap or analytic sampling:
  - [ ] Support both “synthetic from parameters” and “bootstrap from empirical tallies” modes.
  - [ ] Ensure outputs align with existing `BootstrapResult` metadata contracts.
- [ ] Build composable pipelines to attach per-tree metadata:
  - [ ] Spatial placement within polygons (Poisson, stratified by canopy layer, optional clustering).
  - [ ] Crown metrics, biomass factors, bark thickness, etc., using ingest/sampling configs for consistent units.
- [ ] Tests verifying tree count / basal area per stand stays within configured tolerances plus property-based checks on allometric relationships.

### Phase 3 — Export, visualization, and CLI
- [ ] Add exporters for GeoJSON/GeoPackage (stands + tree points), CSV/Parquet tree lists, and lightweight rasters (canopy height, basal area density).
- [ ] Provide a `nemora synthesis generate-landscape` CLI:
  - [ ] Accept seeds/config files (YAML/JSON) describing tiling + stand recipes.
  - [ ] Optionally chain into the bootstrap inspection CLI so users can preview per-stand distributions before generation.
- [ ] Extend docs with a synthesis how-to that covers landscape generation, CLI usage, and integration with future simulations; include provenance/credit for Rlandscape.

### Phase 4 — Validation & extensions
- [ ] Benchmark outputs against historical Rlandscape/FLG examples (document deviations, add a gallery notebook).
- [ ] Hook into the upcoming `nemora.simulations` module so synthetic landscapes feed observation simulators (plot sampling, remote-sensing emulation).
- [ ] Explore advanced features: disturbance simulation (fire/harvest patches), time-stepped growth, integration with ingest benchmark telemetry for calibration.
