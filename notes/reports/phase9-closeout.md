# Phase close-out report: P9 Cross-forest validation and FHOPS handoff

Phase: P9 Cross-forest validation and FHOPS handoff
Parent issue: #31
Pull request: #35, `feature/p9-cross-forest-handoff` into `main`
Evidence head: `7530cad`, the code and documentation head covered by the gates and CI run below
Report commit: the commit that adds this report; its CI run is posted on pull request #35
Date: 2026-10-30
Status: queued for approval. Not merged.

## Recommendation

Approve these actions. The agent carries them out in this order, and reads back each result.

1. Squash-merge pull request #35, and delete the branch `feature/p9-cross-forest-handoff`.
2. Close parent issue #31 with a closeout comment that cites the merge commit.
3. Launch P10 "Calibration and handoff hardening": create its branch, set the phase statuses in `ROADMAP.md`, create its child issues as sub-issues, and start its first task.
4. Keep the Malcolm Knapp regional-transfer result as a negative result; do not present CWHdm pooling as local calibration.

Settings changes, each with its exact command. Each one runs only after its own approval.

- None.

## 1. Objectives and acceptance

| Criterion | Evidence | Status |
| --- | --- | --- |
| Malcolm Knapp pooling is explicit and reproducible | #32, `3244c1a`; 6 local hold-out, 43 regional 50–100 km ring | met |
| Cross-forest metrics use the same definitions as Gavin | `notes/reports/phase9_validation.md` | met |
| Exported inventory validates against FHOPS contract | #33, `e6993e3`; FHOPS loader valid, 1,978 trees | met |
| Round-trip catches schema/unit errors | `tests/test_phase9_fhops_handoff.py` | met |
| FHOPS run reproducible from exported inventory | #34, `7530cad`; 530.54 m³, 0 unreachable | met |
| Reports preserve observed/fitted/assumed/synthetic labels | `notes/reports/phase9_validation.md` | met |

The cross-forest result is limited: Malcolm Knapp regional transfer has BA -7.4%,
QMD -2.3%, mean DBH -18.0%, and mean height -26.3%. This is a documented negative
result, not local calibration.

## 2. Workflow compliance

- Parent issue before the branch: parent #31 and child issues #32-#34 were created before `feature/p9-cross-forest-handoff`.
- Branch from `main`: `da48075` is the merge base.
- Commit messages that start with task IDs: `e6981ee`, `3244c1a`, `e6993e3`, `7530cad`.
- Gates for each commit: section 3; the final code/documentation head was gated after the last change.
- Roadmap and changelog in step: P9 activation, Malcolm Knapp validation, FHOPS export, and system run are recorded.
- Child issues closed with closeout evidence: #32, #33, #34.
- Approval gates: branch creation, commits, pushes, issue edits, and the pull request were allowed actions. No merge, parent closure, branch deletion, tag, release, or settings change was performed.

## 3. Gate evidence

All gates were run on the final code/documentation head before the queue commit:

- `pytest` — 208 passed, 9 skipped.
- `ruff format --check src tests scripts` — pass.
- `ruff check src tests scripts` — pass.
- `mypy src` — pass.
- `sphinx-build -b html docs _build/html -W` — pass.
- `pre-commit run --all-files` — pass.

The FHOPS loader round-trip was run separately with FHOPS on `PYTHONPATH` and
passed for the 1,978-tree export. The default test run skips it when FHOPS is not
installed.

## 4. CI evidence

| Run ID | Head commit | Conclusion | Notes |
| --- | --- | --- | --- |
| 41018193575 | `7530cad` | success | `lint-and-test` passed; `deploy-docs` skipped on a pull request |

The report commit is after the evidence head and is limited to this report. Its
CI run is posted on pull request #35.

## 5. Deviations

| Deviation | Cause | Remedy |
| --- | --- | --- |
| Malcolm Knapp regional transfer is poor for DBH/height | The 50–100 km CWHdm ring is a different population and a single BEC label is too coarse | Reported as a negative result; P10 should stratify by site series/age/height or use local adjustment |
| FHOPS fixture block warns about work_required mismatch | The copied FHOPS example block still has its original small work requirement | Report the warning; do not treat the system output as an operational plan; P10 should reconcile block volume |
| Local P6/P7/P8 feature branches remain | `git branch -d` refused after squash merges | Left in place; not force-deleted without approval |

## 6. Next phase

Proposed P10: **Calibration and handoff hardening**.

- P10.1 Stratify the CWHdm regional pool by site series/age/height or add a local
  hierarchical adjustment; rerun Malcolm Knapp.
- P10.2 Automate FHOPS scenario registration/pin checks and reconcile block
  `work_required` with the exported inventory.
- P10.3 Rerun the cross-forest system comparison with uncertainty and document
  machine-rate sensitivity.
- P10.4 Close out with a calibration/handoff report.

Acceptance: the refinement is selected by hold-out transfer; the FHOPS scenario
is internally consistent; all synthetic/observed labels and limits remain; no
operational claim.

Defaults: keep Gavin as the primary validation; treat Malcolm Knapp as a transfer
test; use FHOPS 1.2.0; no FHOPS core changes; do not touch P16 scope.

## 7. Decision

- [ ] Approve as written.
- [ ] Approve with the changes stated here: <changes>.
