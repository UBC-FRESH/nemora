# Nemora

Nemora is an early-stage meta-package for forest analytics. It aims to provide an
interoperable collection of submodules that cover the typical workflow from *raw inventory data*
through *statistical fitting*, *synthetic forest generation*, and *inventory simulation*. The
project is only a few days old, which means we can move quickly, but it also means the API is
still fluid—expect rapid iteration and watch the changelog.

## High-Level Goals

- **Core types & tooling (`nemora.core`)** – canonical dataclasses and helpers shared across every
  module (e.g., `InventorySpec`, `FitResult`, reproducible random seeds).
- **Central distribution registry (`nemora.distributions`)** – a single source of truth for
  forestry-relevant PDFs/CDFs used by ingestion, fitting, sampling, and synthetic generation.
- **Distribution fitting (`nemora.distfit`)** – grouped estimators, mixture fitting, and goodness-of-fit
  diagnostics. This is the first module we are pushing to alpha.
- **Sampling utilities (`nemora.sampling`)** – analytic/numeric PDF→CDF inversion, bootstrap and
  Monte Carlo samplers, mixture helpers.
- **Ingestion/ETL (`nemora.ingest`)** – transforms raw inventory releases (provincial portals,
  open data) into the tidy secondary forms consumed by the rest of Nemora.
- **Synthetic forest generation (`nemora.synthforest`)** – builds landscape mosaics, stand-level
  attributes, and stem populations for simulation and testing.
- **Inventory simulations (`nemora.simulations`)** – simulates measurement campaigns (plots, LiDAR,
  transects) against synthetic forests with configurable error models.
- **CLI & API parity** – Nemora ships both a Typer-based CLI (`nemora …`) and a user-facing Python
  API. Scripts in `scripts/` remain available; we plan to add CLI shims rather than remove them.

## Current Status (Rapid Iteration)

| Module              | Status / Notes                                                                 |
| ------------------- | ------------------------------------------------------------------------------ |
| `core`              | ✅ Bootstrapped. Hosts shared dataclasses and compatibility shims.             |
| `distributions`     | ✅ Central registry connected to `distfit`, `sampling`, and future modules.    |
| `distfit`           | 🚧 Targeting alpha; grouped EM, mixtures, CLI wiring migrated here.            |
| `sampling`          | 📝 Planned. Will consume registry + distfit outputs after alpha ships.         |
| `ingest`            | 📝 Planned. Will supersede current helper scripts—design in progress.          |
| `synthforest`       | 📝 Planned. Synthetic landscape and stem generation to follow sampling module. |
| `simulations`       | 📝 Planned. Builds on synthforest; design sketches in roadmap.                 |

See [`notes/nemora_modular_reorg_plan.md`](notes/nemora_modular_reorg_plan.md) for the detailed
timeline, sequencing, and dependencies. The plan mirrors the table above and is the source of
truth for day-to-day work.

## Documentation

Live documentation is published on Read the Docs: https://nemora.readthedocs.io/en/latest/
The site tracks the main branch and will expand as new modules come online.

## Nightly Monitoring

A nightly GitHub Actions workflow (`Nightly Ingest Integration`) exercises live FAIB/FIA downloads.
Failures automatically open an issue labeled `nightly-ingest-failure`. To receive notifications:

1. Watch this repository (top-right ➜ **Watch** ➜ *All Activity*) so issue creation triggers emails.
2. In GitHub **Settings → Notifications**, ensure email alerts for *Issues* and *Actions workflow
   runs* are enabled.
3. When an issue arrives, inspect the linked workflow logs to determine whether the failure is
   transient or indicates upstream schema changes.

Re-run the workflow locally with:

```bash
NEMORA_RUN_FAIB_INTEGRATION=1 NEMORA_RUN_FIA_INTEGRATION=1 \
  pytest tests/test_ingest_faib.py::test_build_faib_dataset_source_integration \
         tests/test_ingest_faib.py::test_download_faib_csvs_integration \
         tests/test_ingest_fia.py::test_download_fia_tables_integration
```

## Relationship to Other Toolkits

- **ForestFit (R)** – Nemora borrows ideas from the ForestFit literature and logs planned imports in
  [`candidate-import-from-ForestFit-features.md`](candidate-import-from-ForestFit-features.md).
  We aim to interoperate, not replace: ForestFit covers more mature mixed models today; Nemora is
  focusing on workflow integration and Python-first pipelines.
- **Existing scripts/notebooks** – The repository still contains historic parity notebooks and
  scripts. Many will be rewritten or replaced once the new modules mature. Feel free to use them,
  but watch for “TODO” callouts noting planned refactors.

## Repository Layout

```
src/nemora/
    core/            # Shared dataclasses and helpers
    distributions/   # Canonical distribution registry
    distfit/         # Distribution fitting (alpha focus)
    ingest/          # (planned) inventory ETL pipelines
    sampling/        # (planned) PDF/CDF inversion & sampling
    synthforest/     # (planned) synthetic forest generator
    simulations/     # (planned) inventory simulation module
    cli.py           # Typer CLI entry point (subcommands on the roadmap)
docs/                # Sphinx documentation (How-to, reference, theory)
tests/               # Pytest suites + fixtures
examples/            # Notebooks and scripts (being reorganised)
notes/               # Planning documents and prototypes
scripts/             # Legacy helpers (will be re-housed under ingest)
r/nemorar/           # Reticulate wrapper scaffold for R users
```

## Getting Started

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
nemora --help  # CLI smoke test (distfit alpha commands live here)
```

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for coding standards, testing expectations, and review
checklists. Documentation builds with Sphinx (`docs/`); we’ll flip the switch on Read the Docs once
the module reorganisation stabilises.

## Documentation TODOs

Many doc pages still assume the original scope. As the new modules land we will:

- Rework the “How-to” guides to spotlight ingest, sampling, synthforest, and simulations.
- Expand the reference section with per-module API docs (`nemora.core`, `nemora.distributions`,
  `nemora.distfit`, …).
- Annotate legacy pages with `.. todo::` blocks indicating where scope has changed.

## Contributing

Pull requests are welcome. Please run `ruff`, `mypy`, and `pytest` locally before submitting and
update docs/tests alongside code changes. When touching the reorganised modules, keep an eye on the
alpha plan so we can land the distfit milestone quickly.

## License

MIT – see [`LICENSE`](LICENSE).
