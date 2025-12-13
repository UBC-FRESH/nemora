# Synthesis Phase 2 validation snapshot

This note captures baseline metrics to compare against future Phase 3 refactors.

## CJFR/rlandscape tessellation baseline

- Fixture: `tests/fixtures/synthesis/reference_metrics.json`
- Config: count=25, aspect_ratio=1.5, mix={uniform 0.6, cluster 0.2, inhibition 0.2}, seed=20251207
- Metrics: polygon_count=25, area_mean≈0.06, area_cv≈0.604, vertex_degree_mean=5.2,
  vertex_degree_std≈1.095

## Tree placement baselines

- Analytic sampler (lognormal μ=2.0, σ²=0.25), square polygon:
  - Mean DBH expected ≈8.37 cm; regression asserts 7.5–9.5 cm band under seeded RNG.
  - Min spacing honoured at 0.02–0.2 across Poisson/stratified/clustered modes in tests.
- Clustered mode with bootstrap sampler:
  - Seeded RNG produces 6 trees with mean DBH close to 15.7 cm (two resamples pooled).
  - Min spacing 0.05 respected in a 2×1.5 polygon; points remain in-bounds.

## Attribute provenance

- Placeholder allometry (power laws) and crown ratio are recorded on each tree record under
  `attributes_provenance` so downstream consumers can track coefficient changes.

## Follow-ups before locking Phase 2

- Swap placeholder allometry for ingest-derived coefficients and update this note with the calibrated
  sources/versions.
- Add a gallery figure/notebook showing analytic vs. bootstrap placement density for clustered mode
  using the current fixtures.
