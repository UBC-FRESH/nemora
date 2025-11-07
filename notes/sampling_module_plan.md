# Sampling Module Prototype Plan

Date: 2025-11-07
Status: Working outline for Phase 2 sampling deliverables.

## Goals

- Provide a unified API for drawing samples from Nemora distributions, mixtures, and fitted inventories.
- Support both analytic CDF inversion (where closed forms exist) and numeric integration fallback.
- Deliver bootstrap utilities used by distribution fitting, synthetic forest, and simulation workflows.

## Immediate priorities

1. **Analytic inversion coverage**
   - Audit distributions with closed-form inverse CDFs (Weibull, Gamma, Lognormal, Johnson SB/Birnbaum–Saunders?).
   - Implement `inverse_cdf` hooks within the distribution registry; fall back to SciPy where possible.
   - Add regression tests comparing analytic inversion to SciPy stats implementations.
2. **Numeric PDF→CDF integration**
   - Extend `pdf_to_cdf` to accept adaptive quadrature/backends (Simpson, Romberg) and expose tolerances via config.
   - Cache numeric grids for reuse when sampling repeatedly from the same fit.
   - Validate numeric integration against analytic references to quantify error bounds.
3. **Bootstrap API surface**
   - Finalise `bootstrap_inventory` interface (naming, return types) and document expected inputs (bins, tallies, RNG).
   - Provide helpers for sampling direct DBH vectors vs (dbh, tally) table outputs.
   - Ensure compatibility with grouped fits (e.g., respect grouped Weibull offset metadata).
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
- [ ] Experiment with adaptive quadrature performance for numeric CDFs.
- [ ] Prototype enhanced bootstrap API and align naming with synthforest requirements.
