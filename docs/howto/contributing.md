# Contributing Workflow

This short guide complements {doc}`../reference/distributions` and the repository-level
`CONTRIBUTING.md`, showing how to prepare your environment and validate changes before opening a
pull request.

## Setup Checklist

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Development Loop

- Write code or documentation updates.
- Format and lint:

  ```bash
  ruff check src tests
  ruff format src tests
  mypy src
  ```

- Run tests:

  ```bash
  pytest
  ```

- Render docs when relevant:

  ```bash
  sphinx-build -b html docs docs/_build/html
  ```

## Pull Request Checklist

- [ ] Tests pass locally (`pytest`).
- [ ] Linters and type checkers succeed.
- [ ] Documentation updated (if needed).
- [ ] Added or updated roadmap entries when scope changes.
- [ ] Included distribution registry entries/tests for new PDFs.
- [ ] Summary and testing notes provided in the PR description.

Refer back to `CONTRIBUTING.md` for high-level guidelines and review expectations.
