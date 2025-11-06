# nemora Readiness Roadmap

This roadmap tracks the milestones required to deliver a production-ready release. It will evolve as
requirements firm up and is intended to stay in sync with day-to-day development.

## Phase 0 — Foundations (current)
- [x] Establish repo scaffolding, licensing, packaging metadata, and CI skeleton.
- [x] Bootstrap core package layout (distributions, fitting, workflows, CLI).
- [x] Populate extended distribution registry (28+ GB family members) with tests.
- [x] Document coding standards, contribution flow, and review checklist.

## Phase 1 — Core Functionality
- [x] Implement size-biased HPS workflow parity with reference manuscripts.
- [x] Automate figure/table regeneration for reference HPS parity (export assets from parity notebook into `docs/`).
- [x] Implement censored/two-stage workflow with reproducible baselines.
- [x] Add goodness-of-fit metrics (AICc, chi-square, residual diagnostics) to parity reports and docs.
- [x] Provide DataLad dataset hooks for sample tallies and automate fetch in CLI.
- [x] Flesh out Python API docs and examples (Python scripts + notebooks).
- [x] Expose distribution filtering / parameter preview options in the CLI UX.
- [x] Add worked censored + DataLad-backed tutorials alongside notebook-based workflows.

## Phase 2 — Extensibility & Interfaces
- [x] Catalogue ForestFit capabilities and track candidate imports in
      `candidate-import-from-ForestFit-features.md`.
- [x] Implement user-defined distribution registration (entry points + config).
- [x] Design and prototype finite-mixture fitting (starting with two-component Weibull/Gamma EM).
- [x] Implement mixture distribution utilities (density/CDF/sampling) for fitted components. *(Current
      support covers gamma/Weibull mixtures; extend coverage to other registry distributions in a
      follow-up pass.)*
- [ ] Integrate grouped-sample EM estimators for key distributions (Weibull, JSB, Birnbaum-Saunders). *(Prototype grouped fits now land via SciPy curve-fit/MLE hybrids; replace with true EM updates, include covariance estimates, and remove SciPy fallbacks before marking complete.)*
- [ ] Add Johnson SB / Generalised Secant family support to the distribution registry. *(Johnson SB and GSMN (N ≥ 2) entries now ship with grouped estimators; add user-facing docs/examples and stress-test higher-component fits before closing.)*
- [ ] Expand GOF diagnostics (AD, CvM, KS, grouped residuals) in `FitResult`.
- [ ] Explore piecewise / hybrid PDFs for left-right blending and document references in the docs.
- [ ] Document parameter/return details for public APIs (docstrings and developer notes).
- [ ] Add API reference pages to the Sphinx documentation.
- [ ] Deliver Typer CLI parity with FHOPS feature set (fit, compare, report commands).
- [ ] Build R `nemorar` wrapper with unit tests and pkgdown-ready docs.
- [ ] Add CLI/GUI example scripts and integrate with FHOPS-style logging UX.
- [ ] Document comparative positioning against ForestFit and related toolkits.
- [ ] Add regression fixtures for grouped Weibull parity (HPS manuscript PSP and ForestFit spruce–fir
      tallies) to guard the upcoming EM implementation.

## Detailed Next Steps Notes
1. **Grouped Weibull EM/Newton roll-out**
   - [x] Replace the L-BFGS refinement with a guarded Newton iteration that respects the
     `min(DBH) – 0.5 cm` conditional offset (2025-11-06).
   - [x] Validate the solver on synthetic data and the spruce–fir grouped tallies from Zhang et al.
     (2011); capture results under `tests/fixtures/`. *(2025-11-06; see `tests/test_grouped_fixtures.py`)*
   - [x] Expose a feature flag that keeps the least-squares solution as default until fixtures pass,
     then document the offset behaviour in the HPS guide. *(2025-11-06; CLI `--grouped-weibull-mode`
     and `fit_hps_inventory(..., grouped_weibull_mode=…)` documented in `docs/howto/hps_workflow.md`.)*
   - [x] Complete Birnbaum–Saunders grouped EM (latent normal moments per ForestFit) and retire the SciPy fallback. *(2025-11-06 — replaced the fallback with a moment-closed solution that returns `grouped-em` results while preserving the grouped MLE as a secondary guard.)*
   - [x] Extend the grouped regression suite with Birnbaum–Saunders synthetic fixtures to guard updates. *(2025-11-06 — see `tests/test_grouped.py::test_grouped_birnbaum_saunders_em_on_synthetic_counts`.)*
2. **Fixture preparation**
   - [x] Digitise or extract the manuscript PSP tally (`examples/hps_baf12/...`) and the ForestFit
     spruce–fir bins into dedicated fixtures used by regression tests. *(2025-11-06)*
3. **Documentation updates**
   - [x] Once the grouped solver is validated, update `docs/howto/hps_workflow.md` and related pages
     with guidance on the conditional offset and the new grouped-MLE option. *(2025-11-06)*

## Phase 3 — Quality & Release Readiness
- [ ] Expand unit and property-based tests; add golden fixtures for known tallies.
- [ ] Integrate hypothesis-based validation and numerical stability checks.
- [ ] Complete Sphinx documentation (how-tos, theory, API reference).
- [ ] Configure Read the Docs build + link to GitHub releases.
- [ ] Add benchmarking harness for long-run fitting workloads.
- [ ] Add CI smoke tests for `nemora fetch-reference-data --dry-run`.

## Phase 4 — Community & Deployment
- [ ] Prepare v0.1.0 release notes and changelog.
- [ ] Publish package to PyPI; automate release workflow.
- [ ] Finalise CRAN submission plan for `nemorar`.
- [ ] Draft contributor guide, code of conduct, and issue templates.
- [ ] Announce public release (blog post, mailing lists).

## Backlog & Ideas
- [ ] Investigate GPU-accelerated fitting for large tally batches.
- [ ] Evaluate Bayesian fitting backends (PyMC, Stan) for hierarchical modelling.
- [ ] Add interactive visualization module for comparing candidate PDFs.
- [ ] Explore integration with FHOPS web components for shared dashboards.
