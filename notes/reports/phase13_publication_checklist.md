# Phase 13 publication checklist

Date: 2026-10-30

Status: prepared. Not published.

## Candidate

- Version: `0.1.0a1`
- Wheel: `nemora-0.1.0a1-py3-none-any.whl`
  - SHA-256: `67034381f2cbfe3d10faeaaef55046c075332c785cf1819e6cc2399d83db2def`
- Sdist: `nemora-0.1.0a1.tar.gz`
  - Build SHA-256: `2923e42084e740d3e2aeb141358329d531fb2dfb7c52a9c790584a451ce4faec`
  - TestPyPI SHA-256: `7fc39a8911c78da85aa437e53ae243f1f4268d51166b00e525166d34042712cd`
  - The sdist archive hash changes with build timestamps; content was install-verified.

## Workflow

- File: `.github/workflows/release.yaml`
- Trigger: tag `v*`
- TestPyPI environment: `testpypi`
- PyPI environment: `pypi`
- OIDC trusted publishing; no tokens in the repository.
- Attestations are temporarily disabled for the PyPI retry because Rekor returned HTTP 502 during signing; the OIDC trusted-publisher check still applies.
- Tags ending in `-testpypi` run build → TestPyPI.
- Final `v0.1.0a1` runs build → PyPI; TestPyPI is skipped.
- PyPI publication is separately approved.

## External settings the maintainer must confirm

- Repository: `UBC-FRESH/nemora`
- Workflow filename: `release.yaml`
- TestPyPI publisher environment: `testpypi`
- PyPI publisher environment: `pypi`
- The maintainer must create/approve the publishers outside the repository.

## Order of operations

1. Merge the approved P13 PR.
2. Confirm the external trusted-publisher settings.
3. Approve TestPyPI publication explicitly.
4. TestPyPI verification tag `v0.1.0a1-testpypi` published and verified.
5. Approve PyPI publication explicitly.
6. Tag `v0.1.0a1` on the approved commit; the workflow skips TestPyPI and publishes PyPI.
7. Create the GitHub release; verify installs and release assets.

## Approval boundary

No tag, GitHub release, TestPyPI upload, PyPI upload, credential creation, or
settings change is authorized by P13.1. P13.2 and P13.3 each need a separate
explicit yes.
