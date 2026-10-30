# Phase 13 publication checklist

Date: 2026-10-30

Status: prepared. Not published.

## Candidate

- Version: `0.1.0a1`
- Wheel: `nemora-0.1.0a1-py3-none-any.whl`
  - SHA-256: `67034381f2cbfe3d10faeaaef55046c075332c785cf1819e6cc2399d83db2def`
- Sdist: `nemora-0.1.0a1.tar.gz`
  - SHA-256: `2923e42084e740d3e2aeb141358329d531fb2dfb7c52a9c790584a451ce4faec`

## Workflow

- File: `.github/workflows/release.yaml`
- Trigger: tag `v*`
- TestPyPI environment: `testpypi`
- PyPI environment: `pypi`
- OIDC trusted publishing; no tokens in the repository.
- Tag runs build → TestPyPI only.
- PyPI publication is a separate, separately approved step; it is not chained to the tag run.

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
4. Tag `v0.1.0a1` on the approved merged commit and watch the TestPyPI workflow:
   - build;
   - `publish-testpypi`;
   - stop and verify TestPyPI before any PyPI step.
5. Approve PyPI publication explicitly.
6. Add/execute the separately approved PyPI publication path; do not reuse the TestPyPI tag run.
7. Create the GitHub release; verify installs and release assets.

## Approval boundary

No tag, GitHub release, TestPyPI upload, PyPI upload, credential creation, or
settings change is authorized by P13.1. P13.2 and P13.3 each need a separate
explicit yes.
