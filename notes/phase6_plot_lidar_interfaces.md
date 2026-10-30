# Phase 6 plan: plot resolution and LiDAR observation interfaces

Parent issue: #15

Branch: `feature/p6-plot-lidar-interfaces`

Status: active

## Goal

Add the generic Nemora side of the plot/LiDAR boundary described in
`notes/adr/0001-plot-lidar-boundaries.md`:

- resolve mapped plot, tree, and species tables into `nemora.inventory.v1`;
- materialize tally classes into individual tree records when a consumer needs
  that grain;
- sample existing canopy rasters and compute documented canopy metrics without
  taking on LiDAR production.

## Tasks

### P6.1: boundary and API plan

- Record the ownership decision in an ADR.
- Record the public API and tests in this note.
- Add the Phase 6 roadmap entry and issue links.

### P6.2: plot resolution and tally materialization

Proposed public surface in `nemora.ingest.plots`:

```python
@dataclass(frozen=True)
class PlotInventorySpec:
    release_id: str
    provenance: InventoryProvenance
    plot_columns: Mapping[str, str] = ...
    tree_columns: Mapping[str, str] = ...
    species_columns: Mapping[str, str] = ...
    plot_defaults: Mapping[str, object] = ...
    tree_defaults: Mapping[str, object] = ...
    species_defaults: Mapping[str, object] = ...
    units: InventoryUnits = ...
    crs: str | None = None
    metadata: Mapping[str, object] = ...
    derive_tree_count: bool = True


def build_inventory_release(
    plots: pd.DataFrame,
    trees: pd.DataFrame,
    *,
    spec: PlotInventorySpec,
    species: pd.DataFrame | None = None,
) -> InventoryRelease: ...


def materialize_tally_trees(
    release: InventoryRelease,
    *,
    tree_id_prefix: str = "T",
) -> InventoryRelease: ...
```

`plot_columns`, `tree_columns`, and `species_columns` map canonical schema names
to source column names. A canonical name already present in the source frame is
used as-is. Missing optional columns are filled with defaults or `NA`.

`derive_tree_count` fills a missing `plots.tree_count` with the number of tree
rows per plot. It does not sum expansion tallies; that distinction is documented
because tally weights are not literal stem counts.

`materialize_tally_trees` expands rows with a null `tree_id` and an integer
`tally` into that many rows with `tally=1` and deterministic per-plot tree IDs.
Existing tree IDs are preserved. Fractional tallies are rejected because they
cannot be materialized as individual trees without a separate expansion rule.

### P6.3: LiDAR observation adapter

Proposed public surface in `nemora.lidar`:

```python
@dataclass(frozen=True)
class RasterGrid:
    values: np.ndarray
    transform: tuple[float, float, float, float, float, float]
    crs: str | None
    nodata: float | None
    resolution: tuple[float, float]
    width: int
    height: int


def read_raster(path: str | Path, *, band: int = 1) -> RasterGrid: ...
def sample_raster_at_points(
    grid: RasterGrid,
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
) -> np.ndarray: ...
def canopy_metrics(
    values: np.ndarray | Sequence[float],
    *,
    cover_threshold_m: float = 2.0,
) -> CanopyMetrics: ...
def summarize_canopy_grid(
    grid: RasterGrid,
    *,
    cover_threshold_m: float = 2.0,
) -> CanopyMetrics: ...
```

The raster is sampled in its native CRS. Points outside the raster, pixels
marked as nodata, and non-finite values return `NaN`. Canopy metrics use finite,
non-negative heights only; cover is the fraction at or above the threshold and
the summary reports mean, minimum, maximum, p50, and p95.

The first implementation deliberately excludes vector masking, reprojection,
point-cloud IO, CHM derivation, ITD, and polygon summaries. Those remain in
FEMIC/WS4 or can be added as a later adapter.

## Verification

- `python -m pytest tests/test_ingest_plots.py tests/test_lidar.py -q`
- `python -m pytest -q`
- `ruff format --check src tests`
- `ruff check src tests`
- `mypy src`
- `sphinx-build -b html docs _build/html -W` when Sphinx is available

## Out of scope and limits

- No real tree map is inferred from a CHM.
- No claim that a raster metric is field-calibrated.
- No private inventory data, credentials, or machine-specific paths in tracked
  files.
- No FHOPS or WS4 changes in this phase.
