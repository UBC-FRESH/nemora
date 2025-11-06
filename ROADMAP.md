# dbhdistfit Readiness Roadmap

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
- [ ] Implement user-defined distribution registration (entry points + config).
- [ ] Design and prototype finite-mixture fitting (starting with two-component Weibull/Gamma EM).
- [ ] Implement mixture distribution utilities (density/CDF/sampling) for fitted components.
- [ ] Integrate grouped-sample EM estimators for key distributions (Weibull, JSB, Birnbaum–Saunders).
- [ ] Add Johnson SB / Generalised Secant family support to the distribution registry.
- [ ] Expand GOF diagnostics (AD, CvM, KS, grouped residuals) in `FitResult`.
- [ ] Explore piecewise / hybrid PDFs for left-right blending and document references in the docs.
- [ ] Deliver Typer CLI parity with FHOPS feature set (fit, compare, report commands).
- [ ] Build R `dbhdistfitr` wrapper with unit tests and pkgdown-ready docs.
- [ ] Add CLI/GUI example scripts and integrate with FHOPS-style logging UX.
- [ ] Document comparative positioning against ForestFit and related toolkits.

## Phase 3 — Quality & Release Readiness
- [ ] Expand unit and property-based tests; add golden fixtures for known tallies.
- [ ] Integrate hypothesis-based validation and numerical stability checks.
- [ ] Complete Sphinx documentation (how-tos, theory, API reference).
- [ ] Configure Read the Docs build + link to GitHub releases.
- [ ] Add benchmarking harness for long-run fitting workloads.
- [ ] Add CI smoke tests for `dbhdistfit fetch-reference-data --dry-run`.

## Phase 4 — Community & Deployment
- [ ] Prepare v0.1.0 release notes and changelog.
- [ ] Publish package to PyPI; automate release workflow.
- [ ] Finalise CRAN submission plan for `dbhdistfitr`.
- [ ] Draft contributor guide, code of conduct, and issue templates.
- [ ] Announce public release (blog post, mailing lists).

## Backlog & Ideas
- [ ] Investigate GPU-accelerated fitting for large tally batches.
- [ ] Evaluate Bayesian fitting backends (PyMC, Stan) for hierarchical modelling.
- [ ] Add interactive visualization module for comparing candidate PDFs.
- [ ] Explore integration with FHOPS web components for shared dashboards.
