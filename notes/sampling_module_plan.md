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
   - [ ] Decide whether to add logistic/fisk inverses or document their numeric fallback explicitly.
2. **Numeric PDF→CDF integration**
   - [x] Extend `pdf_to_cdf` to accept trapezoid/Simpson grids and `scipy.integrate.quad`, exposing tolerances via `SamplingConfig`.
   - [ ] Cache numeric grids for reuse when sampling repeatedly from the same fit.
   - [x] Validate numeric integration against additional reference distributions and capture error bounds/benchmarks in docs/tests (gamma vs. SciPy coverage landed).
3. **Bootstrap API surface**
   - [x] Finalise `bootstrap_inventory` interface and introduce `BootstrapResult` with metadata + helper methods, documenting usage in `docs/howto/sampling.md`.
   - [ ] Provide helpers/examples for sampling DBH vectors vs stand tables and clarify grouped-fit metadata propagation.
   - [ ] Ensure compatibility with synthforest by planning how `BootstrapResult` feeds downstream workflows (documented adoption guidance now lives in `docs/howto/sampling.md`; next step is wiring synthforest consumers).
4. **Mixture sampling enhancements**
   - [ ] Allow direct seeding via `numpy.random.Generator` for mixture helpers and integrate diagnostics.
   - [ ] Add support for truncated mixtures / mixture-of-experts weighting if synthforest requires them.

## Documentation tasks

- Add a dedicated "Sampling" how-to page with examples (analytic inversion, numeric fallback, bootstrap workflows).
- Update API reference to surface new config objects (`SamplingConfig`, mixture helpers).
- Provide notebook examples comparing analytic vs numeric sampling accuracy.

## Testing strategy

- Deterministic RNG fixtures (`numpy.random.Generator`) for reproducible sampling tests.
- Property-based tests (Hypothesis) checking that sampled distributions approximate expected moments.
- Integration tests ensuring sampling + distfit pipelines remain compatible when toggling grouped solver modes.

## Open questions

- How to expose performance-sensitive numeric integration parameters in the CLI without overwhelming users?
- Do we need alternate backends (JAX/CuPy) in the short term, or can we defer to later phases?
- Should bootstrap outputs include diagnostic metadata (variance, confidence intervals) by default?

## Next actions

- [x] Draft distribution-specific inverse CDF capability matrix and wire analytic helpers into the registry.
- [x] Add configurable grid density/tolerance options to `pdf_to_cdf`; document the gamma benchmark results.
- [ ] Cache/reuse numeric grids (or memoized integration results) for repeated sampling workloads.
- [x] Add property-based / numeric accuracy tests covering trapezoid, Simpson, and quad integration modes.
- [x] Introduce `BootstrapResult`, document its metadata contract, and expose helper methods (e.g., `stacked`).
- [ ] Align naming/structure with upcoming synthforest sampling needs, including DBH vector helpers and grouped-fit metadata propagation.
