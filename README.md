# nemora

`nemora` is an open-source toolkit for fitting diameter-at-breast-height (DBH) probability
distributions to forest inventory tally data. It packages the workflows developed by the UBC FRESH
Lab for handling both horizontal point sampling (HPS) and fixed-area plot inventories, including
size-bias corrections, left/right censoring, and extensible distribution libraries.

## Features (planned)
- Weighted HPS fitting that reproduces size-biased estimators using standard-form PDFs.
- Two-stage scaling workflow for censored or truncated tallies without bespoke PDF forms.
- Catalogue of 28+ forestry-relevant distributions, with user-defined plugin support.
- Python API, Typer-powered CLI, and an R wrapper (`nemorar`) built on reticulate.
- Reproducible examples (Jupyter, Python scripts, bash) linked to DataLad managed tallies.
- Sphinx documentation with theory notes, API reference, and worked examples.
- Experimental finite-mixture fitting (two-component EM) for grouped stand tables.

## Relationship to other tools
`nemora` complements earlier diameter-distribution toolkits—most notably the R package
[`ForestFit`](https://cran.r-project.org/package=ForestFit)—by focusing on workflow integration and
cross-language accessibility:

- **Workflow-first design.** Horizontal point sampling (HPS) weighting, censored workflows, and
  parity datasets are bundled as ready-to-run pipelines rather than standalone distribution
  routines.
- **Python ecosystem integration.** A Typer CLI, pandas-friendly API, and forthcoming reticulate
  bridge allow the same scripts to run in notebooks, batch jobs, or mixed Python/R projects.
- **Reproducible data packaging.** DataLad-backed reference datasets and CLI bootstrap commands make
  it easy to pull the manuscript tallies or swap in project-specific inventories.
- **Transparent differentiation.** We actively track ideas from ForestFit and related literature
  (finite mixtures, piecewise PDFs, JSB family support). Candidate imports are logged in
  `candidate-import-from-ForestFit-features.md` so that upstream contributions remain visible while
  we extend the Python implementation.

The goal is to interoperate with ForestFit users rather than replace that package; future releases
will surface mixture and piecewise models inspired by the same body of research.

## Project Layout
```
src/nemora/    # Core Python package (core types, distributions, distfit, workflows, etc.)
docs/              # Sphinx documentation sources
tests/             # Pytest suites and fixtures
examples/          # Jupyter notebooks, scripts, CLI samples
config/            # Distribution registry and package defaults
r/nemorar/     # R wrapper (reticulate bridge) scaffolding
.github/workflows/ # CI pipelines
ROADMAP.md         # Working readiness plan
```

## Getting Started
```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
nemora --help
```

See `CONTRIBUTING.md` for coding standards, testing, and review checklists.

Documentation is built with Sphinx under `docs/`. A Read the Docs configuration will follow once
the initial API stabilises.

## Contributing
Contributions are welcome via pull requests. Please run `ruff`, `mypy`, and `pytest` locally before
submitting patches. Changes should include documentation updates and tests where applicable.

## License
This project is released under the MIT License. See `LICENSE` for details.
