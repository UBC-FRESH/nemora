# Phase close-out report: P6 Plot resolution and LiDAR observation interfaces

Phase: P6 Plot resolution and LiDAR observation interfaces
Parent issue: #15
Pull request: #19, `feature/p6-plot-lidar-interfaces` into `main`
Evidence head: `d52e3b1`, the code and documentation head covered by the gates and CI run below
Report commit: the commit that adds this report; its CI run is posted on pull request #19
Date: 2026-10-30
Status: queued for approval. Not merged.

## Recommendation

Approve these actions. The agent carries them out in this order, and reads back each result.

1. Squash-merge pull request #19, and delete the branch `feature/p6-plot-lidar-interfaces`.
2. Close parent issue #15 with a closeout comment that cites the merge commit.
3. Launch P7 "Research-forest plot and LiDAR validation": create its branch, set the phase
   statuses in `ROADMAP.md`, create its child issues as sub-issues, and start its first task.

Settings changes, each with its exact command. Each one runs only after its own approval.

- None.

## 1. Objectives and acceptance

| Criterion | Evidence | Status |
| --- | --- | --- |
| An ADR states what belongs in Nemora and what remains in FEMIC/WS4 | `12d7f4d`, `notes/adr/0001-plot-lidar-boundaries.md` | met |
| `nemora.ingest.plots` resolves mapped source tables into a validated `InventoryRelease` | `91e4320`, `src/nemora/ingest/plots.py`, `tests/test_ingest_plots.py` | met |
| Tally classes can be materialized into individual tree rows with stable IDs | `91e4320`, `materialize_tally_trees`, tests for expansion, ID preservation, and fractional rejection | met |
| `nemora.lidar` reads a raster, samples at points, and computes canopy metrics when rasterio is installed | `da9e416`, `src/nemora/lidar/raster.py`, `tests/test_lidar.py` | met |
| Missing optional dependencies fail with actionable guidance | `da9e416`, `LidarDependencyError`, dependency-boundary test | met |
| Tests and docs describe the behavior and limits | `91e4320`, `da9e416`, `docs/api/lidar.md`, `docs/howto/lidar.md`, `docs/howto/inventory_schema.md` | met |
| The changelog records the work and verification | `da9e416`, `CHANGE_LOG.md` entry dated 2026-10-30 | met |
| Raw LiDAR production and workflow orchestration stay external | `12d7f4d` ADR; no FEMIC/WS4 code moved | met |

The phase meets its acceptance criteria with one stated limit: the LiDAR adapter is an
observation/metrics layer only. It does not produce or reproject rasters, derive a CHM, or
infer a real tree map.

## 2. Workflow compliance

- Parent issue before the branch: parent #15 and child issues #16-#18 were created before
  branch `feature/p6-plot-lidar-interfaces` was created from `main`.
- Branch from `main`: `2e8776c` is the merge base used by the phase branch.
- Commit messages that start with task IDs: `12d7f4d` (P6.1), `91e4320` (P6.2),
  `da9e416` (P6.3), `5302523` (P6.3), and `d52e3b1` (P6.1-P6.3 queue commit) all start
  with a task ID or phase identifier.
- Gates for each commit: section 3; the final branch head was gated after the last code
  and documentation change.
- Roadmap and changelog in step: `12d7f4d` added the Phase 6 roadmap; `da9e416` added the
  changelog entry; `d52e3b1` moved the roadmap to `pr-open`.
- Child issues closed with closeout evidence: #16, #17, and #18 were closed after their
  closeout comments cited commits, verification, and pull request #19.
- Approval gates: branch creation, commits, pushes, issue edits, and the pull request were
  allowed actions. No merge, parent closure, branch deletion, tag, release, or settings
  change was performed.

## 3. Gate evidence

All gates were run on the final code/documentation head before the queue commit and again
after the queue commit. The exact commands and results:

- `pytest` — 187 passed, 8 skipped.
- `ruff format --check src tests` — pass.
- `ruff check src tests` — pass.
- `mypy src` — pass.
- `sphinx-build -b html docs _build/html -W` — pass.
- `pre-commit run --all-files` — pass.

The skipped tests are the existing network/R/WS4 integration tests, not failures. The
tests for this phase are `tests/test_ingest_plots.py` and `tests/test_lidar.py`.

Commit coverage:

- `12d7f4d`: roadmap, ADR, and plan. Covered by the final head documentation and workflow
  gates.
- `91e4320`: plot helpers, tests, and docs. Covered by the final head test, lint, typing,
  Sphinx, and pre-commit gates.
- `da9e416`: LiDAR adapter, tests, docs, optional extra. Covered by the same final head
  gates.
- `5302523`: GeoJSON newline and pinned-pre-commit compatibility. Covered by the same final
  head gates and by the fixture/export regression.
- `d52e3b1`: roadmap status only. Covered by the same final head gates.

## 4. CI evidence

| Run ID | Head commit | Conclusion | Notes |
| --- | --- | --- | --- |
| 41006997123 | `d52e3b1` | success | `lint-and-test` passed; `deploy-docs` skipped on a pull request |

The report commit is after the evidence head and is limited to this report. Its CI run is
posted on pull request #19.

## 5. Deviations

| Deviation | Cause | Remedy |
| --- | --- | --- |
| The first queue sequence omitted the `notes/reports/` close-out artifact and closed child issues before the report existed | The phase-closeout checklist was read after the initial queue | This report is added in the next commit; the child closeout comments already cite commits, verification, and PR #19 |
| Gates were run at the branch head rather than after every individual commit | The phase was developed as one reviewable branch and the final head covers all code/docs changes | The report names the final-head evidence and lists the commits it covers |
| The pinned pre-commit Ruff version reported `S603` against the existing sgsR calls | The existing `# noqa` comments used a reason form that the pinned version did not suppress | `5302523` moved the suppression to the lines the pinned version reports and kept the explanatory comment |
| The repository's end-of-file hook conflicted with the committed GeoJSON fixture | The export helper and fixture had no trailing newline | `5302523` normalized GeoJSON export output and updated the fixture/regression |

## 6. Next phase

Proposed P7: **Research-forest plot and LiDAR validation**.

Goal: produce a reviewed, reproducible validation package for one or both research
forests without claiming a real tree map.

Tasks:

- P7.1 Select and resolve FAIB PSP plots for the target BEC/forest proximity, recording
  licence, provenance, plot filters, and held-out plots.
- P7.2 Build plot-conditioned `nemora.inventory.v1` releases for mature stands and run
  hold-out checks for DBH, height, basal area, stems/ha, and species mix.
- P7.3 Use prepared FEMIC/LidarBC DEM/CHM products through `nemora.lidar` for terrain and
  canopy metrics, with explicit evidence provenance.
- P7.4 Write a validation report with uncertainty and limitations; keep synthetic data
  labelled and avoid operational/regulatory claims.

Acceptance: the pipeline is reproducible from public-data references; hold-out metrics
are reported; LiDAR metrics use native products; every limitation is explicit; no private
data or credentials are committed.

Defaults: start with Gavin Lake if the local plot sample is adequate, otherwise use
Malcolm Knapp with regional pooling and state the pooling rule. Keep the first validation
to one mature stand per forest.

## 7. Decision

- [ ] Approve as written.
- [ ] Approve with the changes stated here: <changes>.
