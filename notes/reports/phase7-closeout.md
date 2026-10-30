# Phase close-out report: P7 Research-forest plot and LiDAR validation

Phase: P7 Research-forest plot and LiDAR validation
Parent issue: #20
Pull request: #25, `feature/p7-research-forest-validation` into `main`
Evidence head: `eb50b6b`, the code and documentation head covered by the gates and CI run below
Report commit: the commit that adds this report; its CI run is posted on pull request #25
Date: 2026-10-30
Status: queued for approval. Not merged.

## Recommendation

Approve these actions. The agent carries them out in this order, and reads back each result.

1. Squash-merge pull request #25, and delete the branch `feature/p7-research-forest-validation`.
2. Close parent issue #20 with a closeout comment that cites the merge commit.
3. Launch P8 "Validation refinement and LiDAR preparation": create its branch, set the phase statuses in `ROADMAP.md`, create its child issues as sub-issues, and start its first task.
4. Keep P7.3 deferred until a prepared Gavin Lake DEM/CHM bundle is supplied or an approved external data task creates one.

Settings changes, each with its exact command. Each one runs only after its own approval.

- None.

## 1. Objectives and acceptance

| Criterion | Evidence | Status |
| --- | --- | --- |
| Select public FAIB plots with recorded provenance and deterministic hold-out split | #21, `4749edc`, `972a4d3`; 20 Gavin plots, 15 train / 5 hold-out | met |
| Resolve plot and live-tree detail into canonical tables | `972a4d3`; 2,870 live trees across 20 plots | met |
| Build validating training/hold-out inventory releases | `bc71d7f`; untracked `build/phase7/gavin/{train,holdout}` releases validate | met |
| Generate a seeded plot-conditioned synthetic stand | `bc71d7f`; 1,978-tree one-hectare equivalent | met with stated limit: statistical reconstruction, not a real tree map |
| Report DBH, height, basal area, stems/ha, species mix, and uncertainty | `bc71d7f`; `notes/reports/phase7_stand_validation.md` | met |
| Validate canopy/terrain metrics from prepared rasters | #23, `ebc99c6`; no prepared Gavin Lake DEM/CHM found | deferred |
| Write a limitations-first validation report | #24, `eb50b6b`; `notes/reports/phase7_validation.md` | met |

The phase does not claim a real tree map, an operational plan, or a
field-calibrated stand. The stand validation is explicitly a negative first pass:
BA is 28.8% high, QMD 14.7% high, weighted DBH KS is 0.201, and species TV
distance is 0.217.

## 2. Workflow compliance

- Parent issue before the branch: parent #20 and child issues #21-#24 were created before `feature/p7-research-forest-validation` was created from `main`.
- Branch from `main`: `901098e` is the merge base.
- Commit messages that start with task IDs: `854b4e0`, `8009e8d`, `4749edc`, `972a4d3`, `bc71d7f`, `ebc99c6`, and `eb50b6b`.
- Gates for each commit: section 3; the final code/documentation head was gated after the last change.
- Roadmap and changelog in step: P7 activation, P7.1 selection/tree detail, P7.2 validation, P7.3 coverage gap, and P7.4 report are recorded in `ROADMAP.md` and `CHANGE_LOG.md`.
- Child issues closed with closeout evidence: #21, #22, and #24 completed; #23 closed as not planned/deferred with the coverage-gap note.
- Approval gates: branch creation, commits, pushes, issue edits, and the pull request were allowed actions. No merge, parent closure, branch deletion, tag, release, or settings change was performed.

## 3. Gate evidence

All gates were run on the final code/documentation head before the queue commit:

- `pytest` — 201 passed, 8 skipped.
- `ruff format --check src tests scripts` — pass.
- `ruff check src tests scripts` — pass.
- `mypy src` — pass.
- `sphinx-build -b html docs _build/html -W` — pass.
- `pre-commit run --all-files` — pass.

Skipped tests are the existing network/R/WS4 integration tests. New tests for this
phase are `tests/test_phase7_plot_selection.py` and
`tests/test_phase7_stand_validation.py`.

## 4. CI evidence

| Run ID | Head commit | Conclusion | Notes |
| --- | --- | --- | --- |
| 41011575754 | `eb50b6b` | success | `lint-and-test` passed; `deploy-docs` skipped on a pull request |

The report commit is after the evidence head and is limited to this report. Its
CI run is posted on pull request #25.

## 5. Deviations

| Deviation | Cause | Remedy |
| --- | --- | --- |
| P7.3 could not be completed | No prepared Gavin Lake DEM/CHM exists in the accessible FEMIC inputs; the nearest prepared AFRF CHM is about 47 km south at Knifecreek | Deferred and documented in `notes/reports/phase7_lidar_coverage.md`; no synthetic or unrelated raster substituted |
| The first stand validation is a negative result | One pooled Weibull and a simple height fit do not capture species/plot structure | Reported explicitly with bootstrap intervals and a refinement recommendation |
| The local P6 branch remains | `git branch -d` refused after the earlier squash merge; force deletion was not approved | Left in place; `main` and the remote branch state are unaffected |

## 6. Next phase

Proposed P8: **Validation refinement and LiDAR preparation**.

Goal: improve the stand model and unblock Gavin Lake terrain/canopy evidence
without weakening the provenance boundary.

Tasks:

- P8.1 Compare species-stratified, mixture, and plot-level DBH models; report
  hold-out improvement over the current pooled Weibull.
- P8.2 Define and execute an approved prepared Gavin Lake LidarBC DEM/CHM
  acquisition/preparation task through FEMIC/WS4; record product provenance,
  CRS, resolution, licence, and footprint.
- P8.3 Rerun the Gavin Lake validation with the selected model and prepared
  raster products; update the report and uncertainty.
- P8.4 Close out P8 with the same evidence package.

Acceptance: the refinement is selected by hold-out performance, not by visual
fit; the rasters are prepared products with provenance; the report keeps
observed, fitted, assumed, and synthetic components separate; no operational or
regulatory claim is made.

Defaults: use Gavin Lake first; keep Malcolm Knapp as an exploratory fallback;
defer any raw LiDAR processing to an approved external data task.

## 7. Decision

- [ ] Approve as written.
- [ ] Approve with the changes stated here: <changes>.
