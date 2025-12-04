# Sampling Module Prototype Plan

Date: 2025-11-07
Status: Working outline for Phase 2 sampling deliverables.

## Goals

- Provide a unified API for drawing samples from Nemora distributions, mixtures, and fitted inventories.
- Support both analytic CDF inversion (where closed forms exist) and numeric integration fallback.
- Deliver bootstrap utilities used by distribution fitting, synthetic forest, and simulation workflows.

## Immediate priorities

1. **Analytic inversion coverage**
   - [x] Capture formulas + SciPy references for analytic candidates (`exp`, `pareto`, `u`, `weibull`, `ln`) in `notes/sampling_inverse_matrix.md`.
   - [x] Implement `inverse_cdf` hooks within the distribution registry and fall back to SciPy where closed forms are unavailable.
   - [x] Add regression tests comparing analytic inversion to SciPy stats implementations.
   - [x] Document the logistic/fisk decision: record numeric fallback in `notes/sampling_inverse_matrix.md` + docs so contributors know no analytic helper exists yet.
2. **Numeric PDF→CDF integration**
   - [x] Extend `pdf_to_cdf` to accept trapezoid/Simpson grids and `scipy.integrate.quad`, exposing tolerances via `SamplingConfig`.
   - [x] Cache numeric grids for reuse when sampling repeatedly from the same fit (optional `cache_numeric_cdf=True`).
   - [x] Validate numeric integration against additional reference distributions and capture error bounds/benchmarks in docs/tests (gamma vs. SciPy coverage landed).
3. **Bootstrap API surface**
   - [x] Finalise `bootstrap_inventory` interface and introduce `BootstrapResult` with metadata + helper methods (`stacked()`, `to_dataframe()`), documenting usage in `docs/howto/sampling.md`.
   - [x] Provide helpers/examples for sampling DBH vectors vs stand tables and clarify grouped-fit metadata propagation (`nemora.sampling.bootstrap_dbh_vectors` + CLI export command + docs/tests).
   - [x] Ensure compatibility with synthesis by exposing helper utilities (`nemora.synthesis.helpers`) and documenting how `BootstrapResult` feeds downstream workflows (`docs/howto/synthesis.md`).
   - [x] Ship a CLI inspection path (`nemora sampling-describe-bootstrap`) plus docs/tests so downstream modules can preview bootstrap metadata without scripting.
4. **Mixture sampling enhancements**
   - [ ] Allow direct seeding via `numpy.random.Generator` for mixture helpers and integrate diagnostics.
   - [ ] Add support for truncated mixtures / mixture-of-experts weighting if synthesis requires them.

## December 2025 Task Queue

- [x] **Analytic coverage decision**
  Recorded that logistic/fisk remain numeric-only (see `notes/sampling_inverse_matrix.md`) and added a docs note so downstream modules expect the fallback.
- [x] **Bootstrap DBH helper + examples**
  Ship a focused helper (and CLI stub) that accepts `BootstrapResult` plus stand metadata, emits DBH vectors grouped by stand, and showcases the flow in `docs/examples/faib_manifest_parquet.md`. Extend tests to cover grouped-fit metadata propagation.
- [ ] **Mixture RNG plumbing**
  Thread `numpy.random.Generator` support through `sample_mixture`/`SamplingConfig`, add deterministic regression cases, and write short docs in `docs/howto/sampling.md`.
- [x] Implemented via `sample_mixture_fit(..., random_state=Generator | int)` with regression coverage + doc updates (2025-12-05).
- [ ] **Truncated / weighted mixtures**
  Decide on API for truncation bounds and mixture-of-experts weights, stub the interface, and log follow-up notes for synthesis/simulation consumers.

### BootstrapResult → DBH helper spec

- Helper lives under `nemora.sampling.helpers` and exposes `bootstrap_dbh_vectors(result: BootstrapResult, *, stand_id: str | int, metadata: Mapping[str, Any]) -> DBHBootstrap`.
- `DBHBootstrap` dataclass returns:
  - `stand_id`
  - `dbh_vectors`: `dict[int, np.ndarray]` keyed by `resample`
  - `frame`: optional pandas DataFrame with `resample`, `bin`, `draw`, `dbh`, `weight`
  - `metadata`: union of `result.metadata` plus any stand-level overrides
- CLI: `nemora sampling export-bootstrap-dbh` accepts a stand table path + optional YAML describing stand IDs; writes JSON/Parquet artifact.
- Docs/examples:
  - `docs/examples/faib_manifest_parquet.md`: add a section showing how to call the helper for a manifest entry.
  - `docs/howto/sampling.md`: describe how DBH vectors propagate into synthesis/simulation.
- Tests:
  - Unit test verifying grouped-fit metadata persistence and deterministic RNG ordering.
  - CLI smoke test ensuring JSON output matches helper structure.

## Documentation tasks

- Add a dedicated "Sampling" how-to page with examples (analytic inversion, numeric fallback, bootstrap workflows).
- Update API reference to surface new config objects (`SamplingConfig`, mixture helpers).
- Provide notebook examples comparing analytic vs numeric sampling accuracy.

## Testing strategy

- Deterministic RNG fixtures (`numpy.random.Generator`) for reproducible sampling tests.
- Property-based tests (Hypothesis) checking that sampled distributions approximate expected moments.
- Integration tests ensuring sampling + fit pipelines remain compatible when toggling grouped solver modes.

## Open questions

- How to expose performance-sensitive numeric integration parameters in the CLI without overwhelming users?
- Do we need alternate backends (JAX/CuPy) in the short term, or can we defer to later phases?
- Should bootstrap outputs include diagnostic metadata (variance, confidence intervals) by default?

## Next actions

- [x] Draft distribution-specific inverse CDF capability matrix and wire analytic helpers into the registry.
- [x] Add configurable grid density/tolerance options to `pdf_to_cdf`; document the gamma benchmark results.
- [x] Cache/reuse numeric grids (or memoized integration results) for repeated sampling workloads.
- [x] Add property-based / numeric accuracy tests covering trapezoid, Simpson, and quad integration modes.
- [x] Introduce `BootstrapResult`, document its metadata contract, and expose helper methods (e.g., `stacked`, `to_dataframe`).
- [ ] Align naming/structure with upcoming synthesis sampling needs, including DBH vector helpers and grouped-fit metadata propagation (follow-up: add CLI/tests that exercise the new helper).
- [x] Extend sampling docs/examples with an ingest-produced Parquet manifest walkthrough plus numeric integration tuning examples (ties into ROADMAP Phase 2 sampling adoption task).
