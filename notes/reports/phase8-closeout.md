# Phase close-out report: P8 Validation refinement and LiDAR coverage expansion

Phase: P8 Validation refinement and LiDAR coverage expansion
Parent issue: #26
Pull request: #30, `feature/p8-validation-refinement` into `main`
Evidence head: `55111a9`, the code and documentation head covered by the gates and CI run below
Report commit: the commit that adds this report; its CI run is posted on pull request #30
Date: 2026-10-30
Status: queued for approval. Not merged.

## Recommendation

Approve these actions. The agent carries them out in this order, and reads back each result.

1. Squash-merge pull request #30, and delete the branch `feature/p8-validation-refinement`.
2. Close parent issue #26 with a closeout comment that cites the merge commit.
3. Launch P9 "Cross-forest validation and FHOPS handoff": create its branch, set the phase statuses in `ROADMAP.md`, create its child issues as sub-issues, and start its first task.
4. Keep the four missing LidarBC plots as an explicit coverage gap; do not substitute synthetic or unrelated rasters.

Settings changes, each with its exact command. Each one runs only after its own approval.

- None.

## 1. Objectives and acceptance

| Criterion | Evidence | Status |
| --- | --- | --- |
| Compare refined stand models on held-out plots | #27, `493177f`, `21571da`; pooled/species/empirical comparison | met |
| Select by held-out performance | Empirical score 0.319 vs 0.545 pooled and 0.734 species | met |
| Expand LidarBC coverage or explain gaps | #28, `294b9bf`; nodata plot resolved, four plots have no indexed footprint | met with stated limit |
| Rerun with selected model and expanded coverage | #29, `55111a9`; empirical validation with 100 bootstrap replicates | met |
| Report before/after metrics and uncertainty | `notes/reports/phase8_validation.md` | met |
| Preserve the no-real-tree-map/no-operational claim | Reports and PR body | met |

P8 improves the stand validation but does not calibrate it. The four residual
LiDAR gaps and the five-plot hold-out remain explicit limitations.

## 2. Workflow compliance

- Parent issue before the branch: parent #26 and child issues #27-#29 were created before `feature/p8-validation-refinement`.
- Branch from `main`: `003506c` is the merge base.
- Commit messages that start with task IDs: `e5a8b76`, `493177f`, `294b9bf`, `21571da`, `55111a9`.
- Gates for each commit: section 3; the final code/documentation head was gated after the last change.
- Roadmap and changelog in step: P8 activation, model comparison, LidarBC expansion, and refined rerun are recorded.
- Child issues closed with closeout evidence: #27, #28, #29.
- Approval gates: branch creation, commits, pushes, issue edits, and the pull request were allowed actions. No merge, parent closure, branch deletion, tag, release, or settings change was performed.

## 3. Gate evidence

All gates were run on the final code/documentation head before the queue commit:

- `pytest` — 204 passed, 8 skipped.
- `ruff format --check src tests scripts` — pass.
- `ruff check src tests scripts` — pass.
- `mypy src` — pass.
- `sphinx-build -b html docs _build/html -W` — pass.
- `pre-commit run --all-files` — pass.

Skipped tests are the existing network/R/WS4 integration tests. New checks for
this phase include `tests/test_phase8_lidar_coverage.py` and the model comparison
tests in `tests/test_phase7_stand_validation.py`.

## 4. CI evidence

| Run ID | Head commit | Conclusion | Notes |
| --- | --- | --- | --- |
| 41016339518 | `55111a9` | success | `lint-and-test` passed; `deploy-docs` skipped on a pull request |

The report commit is after the evidence head and is limited to this report. Its
CI run is posted on pull request #30.

## 5. Deviations

| Deviation | Cause | Remedy |
| --- | --- | --- |
| Species-stratified fitting worsened BA/QMD | Small species samples produced heavy tails; one rare species fell back to pooled | Retained as a negative result; empirical candidate selected |
| Four plots have no LidarBC indexed footprint | Public tile index does not contain the point even at 25 km | Documented as an explicit coverage gap; no substitution |
| P7 local feature branch remains | `git branch -d` refused after the earlier squash merge | Left in place; not force-deleted without approval |

## 6. Next phase

Proposed P9: **Cross-forest validation and FHOPS handoff**.

- P9.1 Validate Malcolm Knapp with an explicit regional pooling rule.
- P9.2 Export the selected Nemora stand to a FHOPS tree inventory with round-trip tests.
- P9.3 Run the FHOPS tree-scale system on the exported inventory and report provenance/limits.
- P9.4 Close out with a cross-forest validation report.

Acceptance: both forests use the same validation definitions; pooling is explicit;
the FHOPS handoff is tested; all outputs preserve synthetic/observed labels; no
operational or regulatory claim is made.

Defaults: Gavin remains primary; Malcolm Knapp is the pooled secondary; LiDAR
gaps stay visible; FHOPS consumes the inventory but Nemora does not depend on
FHOPS.

## 7. Decision

- [ ] Approve as written.
- [ ] Approve with the changes stated here: <changes>.
