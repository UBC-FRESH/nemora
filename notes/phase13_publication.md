# Phase 13 plan: release publication

Parent issue: #53

Branch: `feature/p13-publication`

Status: active

## Candidate

- Version: `0.1.0a1`
- Wheel SHA-256: `67034381f2cbfe3d10faeaaef55046c075332c785cf1819e6cc2399d83db2def`
- Sdist SHA-256: `2923e42084e740d3e2aeb141358329d531fb2dfb7c52a9c790584a451ce4faec`

## Approval gates

1. P13.1 prepares the release workflow and checklist. No publication.
2. P13.2 requires explicit approval to publish to TestPyPI.
3. P13.3 requires separate explicit approval to publish to PyPI/tag/GitHub release.
4. P13.4 verifies and closes out.

## Workflow assumptions

- Workflow file: `.github/workflows/release.yaml`
- TestPyPI environment: `testpypi`
- PyPI environment: `pypi`
- Trigger: tag `v*`

These must match the external trusted-publisher settings. A mismatch is a
blocker; do not guess credentials or settings.

## Limits

- No operational claims.
- No credentials in the repository.
- Tag/publication only after approval.
