# Contributing to nemora

Thanks for your interest in improving `nemora`! This guide outlines the standards and workflow
for contributing code, documentation, and examples.

## Development Environment

1. Create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   pre-commit install
   ```

2. Run the test suite and formatters before submitting changes:

   ```bash
   ruff check src tests
   ruff format src tests
   mypy src
   pytest
   ```

## Code Standards

- Python 3.10+ only. Prefer type hints and dataclasses where appropriate.
- Keep line length ≤ 100 characters; rely on `ruff format`.
- Favour modular design: reusable utilities live in `src/nemora/`.
- Add docstrings to public functions and classes; extend Sphinx docs when APIs change.
- Include tests for new functionality (`tests/` or platform-specific subdirectories).
- Reference distribution registrations must include sample params in registry tests.

## Documentation

- Update the relevant sections under `docs/` (overview, how-to, reference, API).
- For quick how-to additions, create a new page under `docs/howto/` and link it from
  `docs/howto/index.md`.
- Run `sphinx-build -b html docs docs/_build/html` locally when editing complex pages.

## Workflow

1. Fork the repository and create a feature branch (`feature/topic-name`).
2. Make focused commits with clear messages.
3. Ensure `pre-commit` passes locally.
4. Push and open a pull request against `main`.
5. Fill out the PR template, summarising changes and testing.
6. Respond to review feedback promptly.

## Code Review Checklist

Reviewers verify that:

- [ ] Tests cover new or changed behavior.
- [ ] Documentation reflects API or workflow updates.
- [ ] Distribution registry additions include parameter sanity checks.
- [ ] CLI commands remain user-friendly and documented.
- [ ] No lint/type errors remain.
- [ ] Versioning considerations are noted if relevant.

## Questions

Open a GitHub issue or discussion if you need clarification before starting a contribution.
