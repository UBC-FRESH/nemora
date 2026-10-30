# Phase close-out report: P10 Calibration and handoff hardening

Phase: P10 Calibration and handoff hardening
Parent issue: #36
Pull request: #40, `feature/p10-calibration-handoff` into `main`
Evidence head: `8adb385`, the code and documentation head covered by the gates and CI run below
Report commit: the commit that adds this report; its CI run is posted on pull request #40
Date: 2026-10-30
Status: queued for approval. Not merged.

## Recommendation

Approve these actions. The agent carries them out in this order, and reads back each result.

1. Squash-merge pull request #40, and delete the branch `feature/p10-calibration-handoff`.
2. Close parent issue #36 with a closeout comment that cites the merge commit.
3. Launch P11 "Multi-block validation and release readiness": create its branch, set the phase statuses in `ROADMAP.md`, create its child issues as sub-issues, and start its first task.
4. Keep the age-stratified Malcolm Knapp result as a distributional improvement with a density limitation.

Settings changes, each with its exact command. Each one runs only after its own approval.

- None.

## 1. Objectives and acceptance

| Criterion | Evidence | Status |
| --- | --- | --- |
| Stratification rule recorded before rerun | #37, `dc413e1`; age 80–110 rule | met |
| Malcolm Knapp before/after reported | `notes/reports/phase10_mkrf_stratified.md` | met with a mixed result |
| FHOPS scenario pinning reproducible | #38, `8adb385`; `tests/test_phase10_fhops_scenario.py` | met |
| Block `work_required` reconciled | 530.543340 m³; FHOPS warning removed | met |
| System comparison includes rate sensitivity | `notes/reports/phase10_validation.md` | met |
| No operational/real-tree-map claim | Reports and PR | met |

The age-stratified model improves DBH/height/KS but worsens density/BA. The FHOPS
scenario is internally consistent, but the `work_required` reconciliation uses
the same inventory's processed volume rather than an independent yield estimate.

## 2. Workflow compliance

- Parent issue before the branch: parent #36 and child issues #37-#39 were created before `feature/p10-calibration-handoff`.
- Branch from `main`: `87dc80b` is the merge base.
- Commit messages that start with task IDs: `41555a3`, `dc413e1`, `8adb385`.
- Gates for each commit: section 3; the final code/documentation head was gated after the last change.
- Roadmap and changelog in step: P10 activation, stratification, scenario pinning, and rate comparison are recorded.
- Child issues closed with closeout evidence: #37, #38, #39.
- Approval gates: branch creation, commits, pushes, issue edits, and the pull request were allowed actions. No merge, parent closure, branch deletion, tag, release, or settings change was performed.

## 3. Gate evidence

All gates were run on the final code/documentation head before the queue commit:

- `pytest` — 213 passed, 9 skipped.
- `ruff format --check src tests scripts` — pass.
- `ruff check src tests scripts` — pass.
- `mypy src` — pass.
- `sphinx-build -b html docs _build/html -W` — pass.
- `pre-commit run --all-files` — pass.

The FHOPS loader round-trip and system runs were executed separately with FHOPS
1.2.0; scenario validation and pin checks pass.

## 4. CI evidence

| Run ID | Head commit | Conclusion | Notes |
| --- | --- | --- | --- |
| 41021774803 | `8adb385` | success | `lint-and-test` passed; `deploy-docs` skipped on a pull request |

The report commit is after the evidence head and is limited to this report. Its
CI run is posted on pull request #40.

## 5. Deviations

| Deviation | Cause | Remedy |
| --- | --- | --- |
| Age stratification worsens density/BA | The 80–110 regional pool has lower density; distributional matching and density matching conflict | Reported as a mixed result; not used for system comparison without a density constraint |
| Work-required reconciliation is not independent | It uses the processed volume from the same inventory | Documented; a future phase should use an independent yield estimate |
| Old local feature branches remain | `git branch -d` refused after squash merges | Left in place; not force-deleted without approval |

## 6. Next phase

Proposed P11: **Multi-block validation and release readiness**.

- P11.1 Build a small multi-block/multi-landing scenario set for Gavin and
  Malcolm Knapp using the validated inventories.
- P11.2 Add an independent yield check for FHOPS block `work_required` and a
  repeatable scenario registration workflow.
- P11.3 Run the system sensitivity set (truck count, machine rates) and publish
  the multi-block report.
- P11.4 Prepare release/readiness docs and close out.

Acceptance: scenarios are reproducible; the independent yield check is
documented; sensitivity is reported; no operational or regulatory claim; FHOPS
1.2.0 remains the consumer.

Defaults: keep Gavin primary; Malcolm Knapp is a transfer case; use the existing
FHOPS tree inventory contract; no FHOPS core changes; do not touch P16 scope.

## 7. Decision

- [ ] Approve as written.
- [ ] Approve with the changes stated here: <changes>.
