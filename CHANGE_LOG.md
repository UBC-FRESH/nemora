# Development Change Log

# Development Change Log

## 2025-11-04 — Project Bootstrap

- Established initial project scaffold with packaging metadata, distribution registry, fitting workflows, CLI entry points, and base documentation; resolved editable install version detection issues uncovered during first commits.
- Authored early HPS workflow how-to, distribution reference materials, and contribution standards; backstopped initial contributions with codified lint/type expectations (`ruff`, `mypy`, pytest smoke tests).
- Assisted with the first `.readthedocs.yaml` commit (addressed trailing whitespace, ensured staging matched working tree) and verified the RTD build instructions captured dependency installation needs for follow-up work.
- Added Codex operating notes, configured lint/type tooling, and set the stage for subsequent Read the Docs integration and PSP parity implementation.

## 2025-11-05 — Documentation, Data Pipelines, and Parity Delivery

### Read the Docs & documentation pipeline
- Added `.readthedocs.yaml`, enabled installing `docs/requirements.txt`, fixed intersphinx inventories, and aligned the theme/skin with the WS3 project for a consistent UBC-FRESH presentation.
- Iteratively diagnosed RTD build failures (missing `myst_parser`, invalid `{}` inventory entries, detached theme) and resolved them by adjusting config, enabling requirements installation, and rerunning builds until live docs rendered correctly.
- Applied figure directive fixes, switched to the Read the Docs theme with WS3 styling overrides, and formalised the expectation to run `sphinx-build -b html docs _build/html -W` after doc edits.
- Authored new guides (`docs/howto/hps_workflow.md`, `docs/howto/hps_api.md`, `docs/howto/custom_distributions.md`, censored workflow notes) and refreshed the overview/README to match shipped features and FAIR deployment messaging; added references to exported tables/figures and notebook parity claims.

### HPS parity, censored workflows & notebooks
- Automated BC PSP HPS dataset preparation (`scripts/prepare_hps_dataset.py` outputs), added parity regression tests, and produced notebooks for parity reproduction (`examples/hps_parity_reference.ipynb`), BC PSP deployment (`examples/hps_bc_psp_demo.ipynb`), and censored meta-plot demonstrations.
- Integrated parity artefacts (tables, PNGs) into the docs, cited the EarthArXiv preprint accurately, and updated workflow guides to clarify manuscript vs. new dataset claims; renamed notebooks when scope changed to avoid overstating parity.
- Added censored/two-stage workflow baselines with regression coverage (`tests/test_censored_workflow.py`), exported supporting tables/figures, and updated docs on how to reproduce the manuscript meta-plot fits and censored workflows.
- Expanded ROADMAP Phase 1 checkboxes with granular task queues, documenting every notebook/test milestone and queuing follow-up work for censored baselines and DataLad-backed tutorials.

### DataLad integration & CLI enhancements
- Implemented `dbhdistfit fetch-reference-data`, added optional `data` extras (installing `datalad[full]`), hardened remote enablement (defaulting to `arbutus-s3`), and ensured CLI messaging reports missing annex siblings gracefully, with fallbacks when users opt out of DataLad.
- Troubleshot DataLad installation edge cases: guided users to `pip install datalad[full]`, added `.gitignore` rules, documented remote-enabling commands, and validated the command after upstream repository fixes corrected annex config.
- Expanded CLI outputs with GOF metrics (RSS, AICc, chi-square, KS, CvM, AD), residual summaries, and parameter tables; stabilized regression tests after output format changes by parsing table output and adjusting expectations for updated RSS values.
- Exercised CLI tests (`tests/test_cli.py`) and added PSP stand-table fixture checks to ensure CLI output matches API behaviour and preserves distribution ordering.

### Phase 1 completion & release prep
- Added roadmap checkpoints, updated notebooks/workflows to regenerate docs assets, and bumped the package version to `v0.0.1` to mark Phase 1 completion; drafted release notes and GitHub announcement text for the milestone.
- Documented differentiation strategy vs. ForestFit, seeded Phase 2/3 roadmap items (mixed models, API docs, ForestFit-inspired features), and captured candid assessments of remaining gaps.
- Established routine test cadence (`pytest`, `mypy src`, `ruff check`, `sphinx-build -b html docs _build/html -W`, CLI smoke tests) to close out the day’s work and ensure parity regressions stay green.

## 2025-11-06 — Solver, Mixtures, and ForestFit Alignment

### Grouped Weibull solver toggle & regression hardening
- Added `grouped_weibull_mode` plumbing to both `fit_hps_inventory` and the Typer CLI (`--grouped-weibull-mode`) so users can pin least-squares, force grouped MLE, or stay in guarded auto mode; invalid modes now raise user-facing errors.
- Recorded the `min(DBH) – 0.5 cm` conditional offset and solver toggle behaviour in `docs/howto/hps_workflow.md`, aligning docs with the manuscript workflow and explaining how the Newton refinement falls back to least squares.
- Introduced PSP and ForestFit spruce–fir grouped fixtures with regression coverage across auto/LS/MLE modes, including assertions on diagnostic notes when Newton refinement fails and verifying the forced MLE path produces the expected alternative parameter set.
- Updated roadmap detailed notes, `notes/weibull_em_next_steps.md`, and `CODING_AGENT.md` to reference the new toggle, document requirements, and change-log protocol; verified ROADMAP progress checkboxes reflect the completed flag work.
- Tests: `PYTHONPATH=src pytest`, `PYTHONPATH=src mypy src`; ad-hoc sanity checks confirmed parameter parity for PSP fixtures across all solver modes.

### Mixture & grouped estimator scaffolding
- Implemented grouped estimators for Weibull, Johnson SB, Birnbaum–Saunders, and generalized secant mixtures, coupling them with EM/curve-fit fallbacks while documenting remaining caveats (e.g., sparse-bin behaviour, covariance estimation).
- Prototyped finite-mixture fitting (`fit_mixture_grouped`, `fit_mixture_samples`) plus helper utilities for PDF/CDF/sampling; wired gamma/Weibull support, updated tests to exercise mixture PDFs/CDFs, and logged follow-ups for additional distributions.
- Captured grouped EM enhancement tasks (Weibull Newton updates, JSB/Birnbaum extensions, grouped Weibull covariance) in `notes/weibull_em_next_steps.md` and the roadmap detailed notes; added reminders to port ForestFit initialisation tricks during the next iteration.
- Added regression tests (`tests/test_grouped.py`, `tests/test_grouped_fixtures.py`, `tests/test_mixture.py`) covering grouped estimators and mixture helpers to guard future refactors.
- Replaced the Johnson SB grouped fallback with a dedicated EM implementation: latent Beta log-moment integrals (Gauss–Legendre via `quad`), Newton updates on digamma equations, and support clamping now deliver `grouped-em` diagnostics with iteration counts; PSF/ForestFit fixtures exercise the new path.
- Added a Birnbaum–Saunders grouped EM attempt that matches truncated normal moments and searches `β` via bounded scalar minimisation; when the variance term degenerates the workflow falls back to the L-BFGS grouped MLE while flagging the chosen mode in diagnostics (tests accept either path for now).

### ForestFit knowledge capture & planning
- Audited the ForestFit R package (source, CRAN manual, arXiv preprint), catalogued transferable features in `candidate-import-from-ForestFit-features.md`, and planned transparent crediting for any ported methods; noted which algorithms (grouped Johnson SB, moment-based starts) map cleanly to dbhdistfit.
- Relocated uploaded literature to `reference-papers/`, created supporting notes (`notes/weibull_em_references.md`, `notes/weibull_em_next_steps.md`), and cross-linked roadmap tasks to specific ForestFit-derived techniques for prioritisation.
- Extended Phase 2 roadmap with docstring/API documentation milestones, mixture/hybrid modelling plans, and detailed “Detailed Next Steps Notes” updates to ensure development remains sequential.
- Established `CHANGE_LOG.md`, backfilled historical summaries from the conversation log, and expanded `CODING_AGENT.md` instructions to require consulting recorded history before proposing new work.
