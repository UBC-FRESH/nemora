# Phase close-out report: P13 Release publication

Phase: P13 Release publication
Parent issue: #53
Pull request: #58, `feature/p13-publication` into `main`
Evidence head: `a2d6dfd`, the code and documentation head covered by the gates and CI run below
Report commit: the commit that adds this report; its CI run is posted on pull request #58
Date: 2026-10-30
Status: queued for approval. Published to TestPyPI/PyPI; phase PR not merged.

## Recommendation

Approve these actions. The agent carries them out in this order, and reads back each result.

1. Squash-merge pull request #58, and delete the branch `feature/p13-publication`.
2. Close parent issue #53 with a closeout comment that cites the merge commit.
3. Leave the `v0.1.0a1` tag and GitHub release in place; the tag was corrected to the squash merge commit `9327266` so the release targets `main`.
4. Launch P14 "Post-release maintenance and next validation cycle" only if the maintainer wants another phase.

Settings changes, each with its exact command. Each one runs only after its own approval.

- None.

## 1. Objectives and acceptance

| Criterion | Evidence | Status |
| --- | --- | --- |
| Release workflow with OIDC trusted publishing | `.github/workflows/release.yaml`; TestPyPI/PyPI runs green | met |
| TestPyPI publication | run 41031560895; clean install/CLI/pip check | met |
| PyPI publication | run 41032225262; clean install/CLI/pip check | met |
| Tag and GitHub release | `v0.1.0a1` corrected to `9327266`; release assets match PyPI | met |
| No credentials in repository | workflow uses OIDC | met |
| No operational claim | release notes and docs | met |

## 2. Workflow compliance

- Parent issue before the branch: parent #53 and child issues #54-#57 were created before `feature/p13-publication`.
- Branch from `main`: `a513bc7` is the merge base.
- Commit messages that start with task IDs: `f819f87`, `178d9c5`, `f8698bb`, `f140ab1`, `471fe82`, `a2d6dfd`.
- Gates: section 3; final head gated before the queue commit.
- Roadmap and changelog in step: publication prep and results are recorded.
- Child issues closed with closeout evidence: #54, #55, #56; #57 closes with this report.
- Approval gates: TestPyPI and PyPI were separately explicitly approved. Tag/GitHub release were included in the approved PyPI action. No settings changes were made beyond the publisher settings the maintainer supplied.

## 3. Gate evidence

- `pytest` — 219 passed, 9 skipped.
- Ruff, mypy, Sphinx, pre-commit — pass.
- TestPyPI install/import/CLI/pip check — pass.
- PyPI install/import/CLI/pip check — pass.
- `twine check` on the PyPI artifacts — pass.

## 4. CI evidence

| Run ID | Head commit | Conclusion | Notes |
| --- | --- | --- | --- |
| 41031560895 | `f8698bb` | success | TestPyPI build + publish |
| 41032225262 | `471fe82` | success | PyPI build + publish after Rekor fallback |
| 41032503945 | `a2d6dfd` | success | Phase PR CI; `deploy-docs` skipped |

## 5. Deviations

| Deviation | Cause | Remedy |
| --- | --- | --- |
| First PyPI attempt failed before upload | Rekor returned HTTP 502 during attestation signing | Retried with optional attestations disabled; OIDC trusted publishing remained active; PyPI still returned 404 before the successful retry |
| Sdist archive hash differs from the local candidate | Rebuilt archive timestamps | Wheel hash matches; sdist content installs cleanly; PyPI hash is recorded |
| Tag was moved twice | First to the attestation-fix commit; after squash merge, corrected to `9327266` so it targets `main` | A duplicate tag-triggered workflow was cancelled; no additional upload occurred |
| Old local feature branches remain | `git branch -d` refused after squash merges | Left in place; not force-deleted without approval |

## 6. Next phase

Proposed P14: **Post-release maintenance and next validation cycle**.

- P14.1 Monitor post-release installs/issues and fix any packaging regressions.
- P14.2 Address the remaining validation gaps (Malcolm Knapp transfer, four LidarBC plots).
- P14.3 Decide whether to cut a `0.1.0b1`/`0.1.0` candidate after fixes.

Acceptance: release stays installable; regressions are triaged; no operational claims.
This phase is optional; the release is complete without it.

## 7. Decision

- [ ] Approve as written.
- [ ] Approve with the changes stated here: <changes>.
