# Sampling Module Prototype Plan

Date: 2025-11-07
Status: Working outline for Phase 2 sampling deliverables.

## Goals

- Provide a unified API for drawing samples from Nemora distributions, mixtures, and fitted inventories.
- Support both analytic CDF inversion (where closed forms exist) and numeric integration fallback.
- Deliver bootstrap utilities used by distribution fitting, synthetic forest, and simulation workflows.

## Immediate priorities

1. **Analytic inversion coverage**
   - Analytic candidates: `exp`, `pareto`, `u` (uniform), `weibull`, `ln` (lognormal), `logistic`/`fisk` (verify formulas). Everything else falls back to numeric integration/`scipy.stats` quantiles. Action: capture formulas + SciPy references per distribution (`notes/sampling_inverse_matrix.md`) and add tests comparing to `scipy.stats`.
   - Implement `inverse_cdf` hooks within the distribution registry; fall back to SciPy where possible.
   - Add regression tests comparing analytic inversion to SciPy stats implementations.
2. **Numeric PDF→CDF integration**
   - Extend `pdf_to_cdf` to accept adaptive quadrature/backends (Simpson/trapezoid grids vs `scipy.integrate.quad`) and expose tolerances via config.
   - Cache numeric grids for reuse when sampling repeatedly from the same fit.
   - Validate numeric integration against analytic references to quantify error bounds. *(Initial gamma test: 4k-point trapezoid grid vs SciPy `gamma.cdf` max abs error ≈ 1.5e-5; `quad` integration matched at same tolerance — capture these benchmarks in docs.)*
3. **Bootstrap API surface**
   - Finalise `bootstrap_inventory` interface (naming, return types) and document expected inputs (bins, tallies, RNG). Proposal: return `BootstrapResult` containing stacked DataFrame + RNG metadata rather than a list.
   - Provide helpers for sampling direct DBH vectors vs (dbh, tally) table outputs.
   - Ensure compatibility with grouped fits (respect grouped Weibull offset metadata, propagate solver diagnostics into bootstrap notes).
4. **Mixture sampling enhancements**
   - Allow direct seeding via `numpy.random.Generator` and integrate with mixture diagnostics.
   - Add support for truncated mixtures and mixture-of-experts weighting if needed by synthforest.

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

- [ ] Draft distribution-specific inverse CDF capability matrix.
  - [ ] Enumerate current registry distributions (`b1`, `b2`, `birnbaum_saunders`, … `weibull`) and mark whether analytic inverse exists (e.g., `exp`, `pareto`, `u`, `ln`, `weibull`) vs numeric fallback + SciPy reference.
  - [ ] Implement helper for analytic cases (Weibull: `s + beta * (-ln(1-u))**(1/a)`, Lognormal: `exp(mu + sigma * Phi^{-1}(u))`, Pareto, Uniform, Exponential).
- [ ] Experiment with adaptive quadrature performance for numeric CDFs.
  - [x] Prototype trapezoid grid (4k points) vs `scipy.integrate.quad` for Gamma distribution; both achieved max abs error ≈ 1.5e-5 vs SciPy `gamma.cdf`.
  - [ ] Add configurable grid density + tolerance to `pdf_to_cdf`, expose benchmarking notebook in docs.
- [ ] Prototype enhanced bootstrap API and align naming with synthforest requirements.
  - [x] Draft proposal: return `BootstrapResult` (DataFrame + metadata) instead of raw list; include RNG seed, grouped-fit context, ability to sample DBH vectors.
  - [ ] Align naming/structure with upcoming synthforest sampling needs; add plan to `notes/sampling_module_plan.md` + `docs/howto/sampling.md`.
