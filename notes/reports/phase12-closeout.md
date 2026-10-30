# Phase close-out report: P12 Release candidate and publication readiness

Phase: P12 Release candidate and publication readiness
Parent issue: #47
Pull request: #52, `feature/p12-release-candidate` into `main`
Evidence head: `77d479d`, the code and documentation head covered by the gates and CI run below
Report commit: the commit that adds this report; its CI run is posted on pull request #52
Date: 2026-10-30
Status: queued for approval. Not merged. Not tagged. Not published.

## Recommendation

Approve these actions. The agent carries them out in this order, and reads back each result.

1. Squash-merge pull request #52, and delete the branch `feature/p12-release-candidate`.
2. Close parent issue #47 with a closeout comment that cites the merge commit.
3. Launch P13 "Release publication": create its branch, set the phase statuses in `ROADMAP.md`, create its child issues as sub-issues, and start its first task.
4. Do not tag, create a GitHub release, upload to TestPyPI, or upload to PyPI until a separate explicit approval names the exact publication target.

Settings changes, each with its exact command. Each one runs only after its own approval.

- None yet. A release workflow/trusted-publisher configuration would be a P13 action with its own approval.

## 1. Objectives and acceptance

| Criterion | Evidence | Status |
| --- | --- | --- |
| Candidate version frozen | `e8c1d29`; runtime/docs `0.1.0a1` | met |
| Full release gates pass | 218 tests, Ruff, mypy, Sphinx, pre-commit | met |
| Wheel/sdist build | `4da0e37`; wheel 145 KB, sdist 1.6 MB | met |
| Clean-wheel smoke test | clean venv install, import, `nemora --help/--version`, `pip check` | met |
| Artifact contents clean | sdist excludes `_build`, `build`, `reference-papers`, `dist`, `docs/_build` | met |
| Release notes/checklist | `docs/releases/v0.1.0a1.md`, `notes/reports/phase12_publication_checklist.md` | met |
| No tag/publication | no tag, release, or upload performed | met |

The candidate artifact names and hashes are recorded in
`notes/reports/phase12_release_candidate.md`.

## 2. Workflow compliance

- Parent issue before the branch: parent #47 and child issues #48-#51 were created before `feature/p12-release-candidate`.
- Branch from `main`: `5c17e7f` is the merge base.
- Commit messages that start with task IDs: `bfb4783`, `e8c1d29`, `4da0e37`, `77d479d`.
- Gates for each commit: section 3; final head gated after the last change.
- Roadmap and changelog in step: P12 activation, version freeze, artifact build, and release notes are recorded.
- Child issues closed with closeout evidence: #48, #49, #50; #51 closes with this report.
- Approval gates: branch creation, commits, pushes, issue edits, artifact build, and the pull request were allowed actions. No merge, parent closure, branch deletion, tag, release, upload, credential creation, or settings change was performed.

## 3. Gate evidence

All gates were run on the final code/documentation head before the queue commit:

- `pytest` — 218 passed, 9 skipped.
- `ruff format --check src tests scripts` — pass.
- `ruff check src tests scripts` — pass.
- `mypy src` — pass.
- `sphinx-build -b html docs _build/html -W` — pass.
- `pre-commit run --all-files` — pass.
- `python -m build` — pass.
- `twine check dist/*` — pass.
- clean-wheel install, import, `nemora --help`, `nemora --version`, `pip check` — pass.

## 4. CI evidence

| Run ID | Head commit | Conclusion | Notes |
| --- | --- | --- | --- |
| 41027253368 | `77d479d` | success | `lint-and-test` passed; `deploy-docs` skipped on a pull request |

The report commit is after the evidence head and is limited to this report. Its
CI run is posted on pull request #52.

## 5. Deviations

| Deviation | Cause | Remedy |
| --- | --- | --- |
| Initial sdist was 70 MB | Hatch included `_build`, local `build/`, and `reference-papers/` | Explicit sdist include/exclude; rebuilt to 1.6 MB with no junk |
| No release workflow exists | Repository has only CI/nightly workflows | Publication is blocked pending an approved release workflow or manual token path |
| Old local feature branches remain | `git branch -d` refused after squash merges | Left in place; not force-deleted without approval |

## 6. Next phase

Proposed P13: **Release publication**.

- P13.1 Add/verify a release workflow and trusted-publisher settings, or prepare
  an approved manual token path.
- P13.2 Publish to TestPyPI and verify a clean install.
- P13.3 Tag `v0.1.0a1`, publish to PyPI, and create the GitHub release.
- P13.4 Run post-publication checks and close out.

Acceptance: exact publication target approved; artifacts match the approved
hashes; TestPyPI/PyPI installs pass; tag/release point to the merged commit; no
credentials are stored in the repository.

Defaults: publish `0.1.0a1`; use trusted publishing if a workflow is configured,
otherwise request an explicit manual token path; no operational claims.

## 7. Decision

- [ ] Approve as written.
- [ ] Approve with the changes stated here: <changes>.
