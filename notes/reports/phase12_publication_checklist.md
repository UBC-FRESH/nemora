# Phase 12 publication checklist

Date: 2026-10-30

Status: prepared, not executed. Publication requires separate explicit approval.

## Preconditions

- Candidate version is `0.1.0a1`.
- Wheel/sdist hashes are recorded in `notes/reports/phase12_release_candidate.md`.
- Full gates and clean-wheel smoke test pass.
- Maintainer approves the P12 close-out report.
- TestPyPI/PyPI publisher settings are confirmed.
- A release workflow or manual upload path is chosen.

## Current blocker

This repository has no release workflow yet (`.github/workflows/` contains CI
and nightly ingest only). Trusted publishing therefore needs either:

1. a new release workflow plus matching TestPyPI/PyPI trusted-publisher
   settings; or
2. an explicitly approved manual `twine upload` with a token.

Do not guess publisher settings or create credentials.

## Commands (do not run without approval)

```bash
# Build again from the approved commit
python -m build
twine check dist/*

# Tag and push (after explicit approval)
git tag -a v0.1.0a1 -m "Nemora 0.1.0a1"
git push origin v0.1.0a1

# Manual publication alternative (only with an approved token path)
python -m twine upload --repository testpypi dist/*
python -m twine upload dist/*
```

Trusted publishing would replace the `twine upload` commands with a release
workflow using OIDC and matching environments.

## Post-publication checks

```bash
pip install --index-url https://test.pypi.org/simple/ nemora==0.1.0a1
pip install nemora==0.1.0a1
python -c "import nemora; print(nemora.__version__)"
nemora --version
```

## Approval boundary

No tag, GitHub release, TestPyPI upload, PyPI upload, credential creation, or
repository-setting change is authorized by P12 alone.
