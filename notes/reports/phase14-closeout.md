# Phase close-out report: P14 Post-release stabilization

Phase: P14 Post-release stabilization and validation closure
Parent issue: #60
Pull request: #65, `feature/p14-post-release` into `main`
Evidence head: `3ca85db`, the code and documentation head covered by the gates and CI run below
Report commit: the commit that adds this report; its CI run is posted on pull request #65
Date: 2026-10-30
Status: queued for approval. Not merged.

## Recommendation

Approve these actions:

1. Squash-merge pull request #65, and delete the branch `feature/p14-post-release`.
2. Close parent issue #60 with a closeout comment that cites the merge commit.
3. Enter maintenance mode at `0.1.0a1`.
4. Do not launch another phase unless one of the recorded triggers occurs.

Settings changes, each with its exact command. Each one runs only after its own approval.

- None.

## 1. Objectives and acceptance

| Criterion | Evidence | Status |
| --- | --- | --- |
| Base and lidar-extra installs pass | P14.1; PyPI installs, CLI, `pip check` | met |
| Docs/release assets reachable | PyPI/TestPyPI/docs/release 200; hashes matched | met |
| Validation gaps have dispositions | P14.2; deferred/retained with triggers | met |
| Next version decision recorded | P14.3; maintenance-only | met |
| No new publication | no tag/upload in P14 | met |

## 2. Workflow compliance

- Parent issue before branch: #60 and child issues #61-#64 were created first.
- Branch from `main`: `cedb66f`.
- Commits start with task IDs: `4d25139`, `bbe3ce5`, `3ca85db`.
- Gates: section 3; final head gated.
- Roadmap/changelog in step: P14 activation, release verification, gap dispositions, and version decision recorded.
- Child issues closed with evidence: #61, #62, #63; #64 closes with this report.
- Approval gates: no merge, tag, or publication in P14.

## 3. Gate evidence

- Base and `[lidar]` PyPI installs pass.
- `nemora --version` = `nemora 0.1.0a1`.
- `pip check` clean.
- PyPI/TestPyPI/docs/release notes return 200.
- Release assets match PyPI hashes.

## 4. CI evidence

| Run ID | Head commit | Conclusion | Notes |
| --- | --- | --- | --- |
| 41036392425 | `3ca85db` | success | Phase PR lint/test/docs |

The report commit's CI run is posted on PR #65.

## 5. Deviations

| Deviation | Cause | Remedy |
| --- | --- | --- |
| Trusted publishing needed an attestation fallback | Rekor HTTP 502 during the P13 retry | Documented; OIDC retained, optional attestations disabled for that run |
| Validation gaps remain | No scoped model/data task yet | Deferred with explicit triggers |
| Old local feature branches remain | `git branch -d` refused after squash merges | Left in place; no force-delete without approval |

## 6. Next phase

No next phase. Maintenance mode at `0.1.0a1`.

Triggers for a new phase:
- packaging or installation regression;
- user-visible bug;
- a scoped validation/calibration milestone;
- explicit maintainer request for a new release.

## 7. Decision

- [ ] Approve as written.
- [ ] Approve with the changes stated here: <changes>.
