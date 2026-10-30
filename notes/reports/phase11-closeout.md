# Phase close-out report: P11 Multi-block validation and release readiness

Phase: P11 Multi-block validation and release readiness
Parent issue: #41
Pull request: #46, `feature/p11-multi-block-readiness` into `main`
Evidence head: `e5433f7`, the code and documentation head covered by the gates and CI run below
Report commit: the commit that adds this report; its CI run is posted on pull request #46
Date: 2026-10-30
Status: queued for approval. Not merged.

## Recommendation

Approve these actions. The agent carries them out in this order, and reads back each result.

1. Squash-merge pull request #46, and delete the branch `feature/p11-multi-block-readiness`.
2. Close parent issue #41 with a closeout comment that cites the merge commit.
3. Launch P12 "Release candidate and publication readiness": create its branch, set the phase statuses in `ROADMAP.md`, create its child issues as sub-issues, and start its first task.
4. Do not tag or publish without a separate explicit approval; P12 prepares the candidate and lists the exact release commands.

Settings changes, each with its exact command. Each one runs only after its own approval.

- None.

## 1. Objectives and acceptance

| Criterion | Evidence | Status |
| --- | --- | --- |
| Multi-block/multi-landing scenario set | #42, `fc5ca1c`; B01/L1 and B02/L2 | met |
| Scenario checksums and catalog | `notes/reports/phase11_scenario_catalog.json` | met |
| Independent yield validation | #43, `d649d53`; Gavin -21.3% vs FHOPS | met |
| MKRf work_required reconciliation | 471.614268 m³; expected warning retained | met with stated limit |
| Sensitivity set and report | #44, `e47f5d0`; truck/rate comparison | met |
| Release/readiness docs | #45, `e5433f7`; FHOPS handoff how-to/API | met |
| No operational/real-tree-map claim | Reports and PR | met |

## 2. Workflow compliance

- Parent issue before the branch: parent #41 and child issues #42-#45 were created before `feature/p11-multi-block-readiness`.
- Branch from `main`: `1b36141` is the merge base.
- Commit messages that start with task IDs: `335b9ff`, `fc5ca1c`, `d649d53`, `e47f5d0`, `e5433f7`.
- Gates for each commit: section 3; the final code/documentation head was gated after the last change.
- Roadmap and changelog in step: P11 activation, scenario set, yield check, sensitivity, and docs are recorded.
- Child issues closed with closeout evidence: #42, #43, #44; #45 closes with this report.
- Approval gates: branch creation, commits, pushes, issue edits, and the pull request were allowed actions. No merge, parent closure, branch deletion, tag, release, or settings change was performed.

## 3. Gate evidence

All gates were run on the final code/documentation head before the queue commit:

- `pytest` — 217 passed, 9 skipped.
- `ruff format --check src tests scripts` — pass.
- `ruff check src tests scripts` — pass.
- `mypy src` — pass.
- `sphinx-build -b html docs _build/html -W` — pass.
- `pre-commit run --all-files` — pass.

FHOPS 1.2.0 scenario validation and system runs were executed separately; both
scenarios validate, all six sensitivity runs completed with 0 unreachable trees.

## 4. CI evidence

| Run ID | Head commit | Conclusion | Notes |
| --- | --- | --- | --- |
| 41025438129 | `e5433f7` | success | `lint-and-test` passed; `deploy-docs` skipped on a pull request |

The report commit is after the evidence head and is limited to this report. Its
CI run is posted on pull request #46.

## 5. Deviations

| Deviation | Cause | Remedy |
| --- | --- | --- |
| MKRf retains a work_required warning | Registered at the independent estimate, not the FHOPS processed volume | Reported explicitly; Gavin remains warning-free |
| Independent yield estimate is 21.3% below FHOPS | Simple paraboloid check vs FHOPS merchantable-volume model | Reported as a discrepancy, not forced to match |
| Truck count dominates makespan | Single truck is the bottleneck in the fixture | Reported; no stand ranking based on makespan alone |
| Old local feature branches remain | `git branch -d` refused after squash merges | Left in place; not force-deleted without approval |

## 6. Next phase

Proposed P12: **Release candidate and publication readiness**.

- P12.1 Freeze the candidate version and run the full release gate set.
- P12.2 Build wheel/sdist, run a clean-wheel smoke test, and verify docs.
- P12.3 Prepare the release notes, tag command, and trusted-publishing checklist.
- P12.4 Close out and request separate approval for tag/publication.

Acceptance: the candidate builds and installs cleanly; docs build with warnings
as errors; provenance and limitations are current; no tag or publication occurs
without explicit approval.

Defaults: use a pre-1.0 version unless the maintainer requests otherwise; keep
FHOPS 1.2.0 as the consumer; no operational claims; do not touch P16 scope.

## 7. Decision

- [ ] Approve as written.
- [ ] Approve with the changes stated here: <changes>.
