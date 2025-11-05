# dbhdistfit

`dbhdistfit` is an open-source toolkit for fitting diameter-at-breast-height (DBH) probability
distributions to forest inventory tally data. It packages the workflows developed by the UBC FRESH
Lab for handling both horizontal point sampling (HPS) and fixed-area plot inventories, including
size-bias corrections, left/right censoring, and extensible distribution libraries.

## Features (planned)
- Weighted HPS fitting that reproduces size-biased estimators using standard-form PDFs.
- Two-stage scaling workflow for censored or truncated tallies without bespoke PDF forms.
- Catalogue of 28+ forestry-relevant distributions, with user-defined plugin support.
- Python API, Typer-powered CLI, and an R wrapper (`dbhdistfitr`) built on reticulate.
- Reproducible examples (Jupyter, Python scripts, bash) linked to DataLad managed tallies.
- Sphinx documentation with theory notes, API reference, and worked examples.

## Project Layout
```
src/dbhdistfit/    # Core Python package (distributions, weighting, fitting, workflows)
docs/              # Sphinx documentation sources
tests/             # Pytest suites and fixtures
examples/          # Jupyter notebooks, scripts, CLI samples
config/            # Distribution registry and package defaults
r/dbhdistfitr/     # R wrapper (reticulate bridge) scaffolding
.github/workflows/ # CI pipelines
ROADMAP.md         # Working readiness plan
```

## Getting Started
```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
dbhdistfit --help
```

See `CONTRIBUTING.md` for coding standards, testing, and review checklists.

Documentation is built with Sphinx under `docs/`. A Read the Docs configuration will follow once
the initial API stabilises.

## Contributing
Contributions are welcome via pull requests. Please run `ruff`, `mypy`, and `pytest` locally before
submitting patches. Changes should include documentation updates and tests where applicable.

## License
This project is released under the MIT License. See `LICENSE` for details.
