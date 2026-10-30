# Phase 12 release candidate

Date: 2026-10-30

Phase: P12

Tasks: P12.1/P12.2

Status: candidate artifacts build and install cleanly; not published.

## Version

- Candidate: `0.1.0a1`
- Runtime and installed metadata agree.

## Artifacts

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| `nemora-0.1.0a1-py3-none-any.whl` | 145 KB | `67034381f2cbfe3d10faeaaef55046c075332c785cf1819e6cc2399d83db2def` |
| `nemora-0.1.0a1.tar.gz` | 1.6 MB | `2923e42084e740d3e2aeb141358329d531fb2dfb7c52a9c790584a451ce4faec` |

The first sdist was 70 MB because it picked up `_build`, `build/`,
`reference-papers/`, and other local artifacts. `tool.hatch.build.targets.sdist`
now uses an explicit include/exclude set; the rebuilt sdist is 1.6 MB with no
`build`, `_build`, `reference-papers`, `dist`, or `docs/_build` entries.

## Verification

- `python -m build` — pass
- `twine check dist/*` — pass
- clean venv install of the wheel — pass
- `import nemora` and `importlib.metadata.version` — `0.1.0a1`
- `nemora --help` and `nemora --version` — pass
- `pip check` — no broken requirements
- `sphinx-build -b html docs _build/html -W` — pass

## Limits

- No TestPyPI/PyPI upload, tag, or GitHub release was performed.
- The candidate still carries the P11 validation and handoff limits.
- Local build artifacts are not tracked.
