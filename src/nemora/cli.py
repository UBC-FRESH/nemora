"""Typer-based CLI entry point."""

# ruff: noqa: S603

from __future__ import annotations

import csv
import json
import math
import numbers
import statistics
import subprocess
import time
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .core import FitResult, InventorySpec
from .dataprep import PlotSelection
from .distributions import get_distribution, list_distributions, list_registry_metadata
from .fit import fit_inventory
from .ingest.faib import (
    FAIBManifestResult,
    auto_select_bafs,
    build_faib_dataset_source,
    generate_faib_manifest,
)
from .ingest.faib import build_stand_table_from_csvs as build_faib_stand_table
from .ingest.fia import (
    build_fia_dataset_source,
)
from .ingest.fia import (
    build_stand_table_from_csvs as build_fia_stand_table,
)
from .ingest.hps import (
    DEFAULT_PLOT_HEADER as HPS_DEFAULT_PLOT_HEADER,
)
from .ingest.hps import (
    DEFAULT_SAMPLE_BYVISIT as HPS_DEFAULT_SAMPLE_BYVISIT,
)
from .ingest.hps import (
    DEFAULT_TREE_DETAIL as HPS_DEFAULT_TREE_DETAIL,
)
from .ingest.hps import (
    HPSPipelineResult,
    export_hps_outputs,
    run_hps_pipeline,
)
from .ingest.hps import (
    SelectionCriteria as HPSelectionCriteria,
)
from .ingest.hps import (
    load_plot_selections as load_hps_plot_selections,
)
from .sampling import BootstrapResult, bootstrap_dbh_vectors, bootstrap_inventory
from .synthesis import exporters, stands, tessellation
from .synthesis.helpers import bootstrap_payload
from .workflows.hps import fit_hps_inventory

app = typer.Typer(help="Nemora distribution fitting CLI (fit module).")
console = Console()

REGISTRY_DESCRIBE_OPTION = typer.Option(
    None,
    "--describe",
    "-d",
    help="Show metadata for a specific distribution (case-insensitive).",
    show_default=False,
)
REGISTRY_SHOW_METADATA_OPTION = typer.Option(
    False,
    "--show-metadata",
    help="Include parameter bounds/extras when listing distributions.",
    show_default=False,
)
REGISTRY_JSON_OPTION = typer.Option(
    False,
    "--json",
    help="Output registry metadata as JSON (compatible with --describe/--show-metadata).",
    show_default=False,
)


def _prepare_hps_inputs(
    root: Path,
    *,
    fetch: bool,
    cache_dir: Path,
    overwrite: bool,
    baf: float,
    plot_header_file: str,
    sample_byvisit_file: str,
    tree_detail_file: str,
    encoding: str,
    include_all_visits: bool,
    sample_types: list[str],
    max_plots: int | None,
    quiet: bool,
) -> tuple[list[PlotSelection], Path]:
    """Fetch required FAIB inputs and return selections with the tree detail path."""

    target_root = root
    if fetch:
        destination = cache_dir or root
        dataset = build_faib_dataset_source(
            "psp",
            destination=destination,
            filenames=(plot_header_file, sample_byvisit_file, tree_detail_file),
            overwrite=overwrite,
        )
        try:
            downloaded = list(dataset.fetch())
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed to download FAIB PSP files:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        target_root = destination
        if not quiet and downloaded:
            console.print(
                f"[green]Prepared[/green] {len(downloaded)} files in {destination} "
                f"(overwrite={overwrite})"
            )

    plot_header_path = target_root / plot_header_file
    sample_byvisit_path = target_root / sample_byvisit_file
    tree_detail_path = target_root / tree_detail_file

    missing = [
        str(path)
        for path in (plot_header_path, sample_byvisit_path, tree_detail_path)
        if not path.exists()
    ]
    if missing:
        console.print("[red]Missing required FAIB PSP files:[/red] " + ", ".join(missing))
        raise typer.Exit(code=1)

    criteria = HPSelectionCriteria(
        first_visit_only=not include_all_visits,
        allowed_sample_types=tuple(sample_types) if sample_types else None,
        max_plots=max_plots,
    )
    selections = load_hps_plot_selections(
        plot_header_path,
        sample_byvisit_path,
        baf=baf,
        criteria=criteria,
        encoding=encoding,
    )
    return selections, tree_detail_path


_STAND_TABLE_BIN_COLUMNS = ("bin", "bin_cm", "dbh_cm", "dbh", "diameter_cm")
_STAND_TABLE_TALLY_COLUMNS = ("tally", "tallies", "count", "stand_table", "frequency")


def _load_stand_table(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a stand table CSV/Parquet file into numpy arrays."""

    import pandas as pd

    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
    else:
        frame = pd.read_csv(path)
    lower_columns = {column.lower(): column for column in frame.columns}
    bin_column = next(
        (lower_columns[name] for name in _STAND_TABLE_BIN_COLUMNS if name in lower_columns),
        None,
    )
    tally_column = next(
        (lower_columns[name] for name in _STAND_TABLE_TALLY_COLUMNS if name in lower_columns),
        None,
    )
    if bin_column is None or tally_column is None:
        message = (
            "Stand table must include columns for bins and tallies; "
            "accepted bin labels: "
            f"{', '.join(_STAND_TABLE_BIN_COLUMNS)}, tally labels: "
            f"{', '.join(_STAND_TABLE_TALLY_COLUMNS)}"
        )
        console.print(f"[red]{message}[/red]")
        raise typer.Exit(code=1)
    subset = frame[[bin_column, tally_column]].dropna()
    bins = subset[bin_column].to_numpy(dtype=float)
    tallies = subset[tally_column].to_numpy(dtype=float)
    if bins.size == 0 or tallies.size == 0:
        console.print("[red]Stand table contains no usable rows.[/red]")
        raise typer.Exit(code=1)
    return bins, tallies


def _parse_parameter_assignments(assignments: Iterable[str]) -> dict[str, float]:
    parameters: dict[str, float] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise typer.BadParameter(f"Parameter assignment '{assignment}' must be NAME=VALUE.")
        key, value = assignment.split("=", 1)
        key = key.strip()
        if not key:
            raise typer.BadParameter(f"Invalid parameter assignment '{assignment}'.")
        try:
            parameters[key] = float(value)
        except ValueError as exc:  # noqa: TRY003
            raise typer.BadParameter(f"Parameter '{key}' requires a numeric value.") from exc
    return parameters


def _prepare_bootstrap_result(
    stand_table: Path,
    distribution: str,
    params: list[str] | None,
    *,
    resamples: int,
    sample_size: int,
    seed: numbers.Integral | None,
) -> tuple[FitResult, BootstrapResult, bool]:
    """Fit (when needed) and bootstrap a stand table."""

    bins, tallies = _load_stand_table(stand_table)
    parameter_map = _parse_parameter_assignments(params) if params else {}
    explicit_params = bool(parameter_map)
    if parameter_map:
        fit = FitResult(distribution=distribution, parameters=parameter_map)
    else:
        inventory = InventorySpec(
            name=stand_table.stem,
            sampling="stand-table",
            bins=bins,
            tallies=tallies,
            metadata={"grouped": True},
        )
        try:
            fit = fit_inventory(inventory, [distribution], configs={})[0]
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed to fit {distribution}:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    try:
        bootstrap_result = cast(
            BootstrapResult,
            bootstrap_inventory(
                fit,
                bins,
                tallies,
                resamples=resamples,
                sample_size=sample_size,
                random_state=seed,
                return_result=True,
            ),
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Bootstrap failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    return fit, bootstrap_result, explicit_params


def _render_bootstrap_metadata(metadata: dict[str, object]) -> None:
    table = Table(title="Bootstrap Metadata")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="magenta")
    for field in ("distribution", "resamples", "sample_size", "rng_seed"):
        value = metadata.get(field)
        if value is None:
            continue
        table.add_row(field, str(value))
    bins_value = metadata.get("bins")
    bins = (
        np.asarray(bins_value, dtype=float) if bins_value is not None else np.array([], dtype=float)
    )
    tallies_value = metadata.get("tallies")
    tallies = (
        np.asarray(tallies_value, dtype=float)
        if tallies_value is not None
        else np.array([], dtype=float)
    )
    if bins.size:
        table.add_row("bin_count", str(bins.size))
    if tallies.size:
        table.add_row("tally_total", f"{float(tallies.sum()):.2f}")
    console.print(table)

    params = metadata.get("parameters")
    if isinstance(params, dict) and params:
        param_table = Table(title="Fitted Parameters")
        param_table.add_column("Name", style="cyan")
        param_table.add_column("Value", style="green")
        for key, value in params.items():
            param_table.add_row(key, f"{float(cast(Any, value)):.6f}")
        console.print(param_table)


def _json_ready_metadata(metadata: dict[str, object]) -> dict[str, object]:
    serialisable: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            serialisable[key] = value.tolist()
        elif isinstance(value, np.generic):
            serialisable[key] = value.item()
        else:
            serialisable[key] = value
    return serialisable


DBH_FILE_ARGUMENT = typer.Argument(
    ...,
    exists=True,
    readable=True,
    help="CSV with `dbh_cm` and `tally` columns.",
)

STAND_TABLE_ARGUMENT = typer.Argument(
    ...,
    exists=True,
    readable=True,
    help="CSV/Parquet stand table with bin/tally columns.",
)

SEED_RECIPE_OUTPUT_OPTION = typer.Option(
    Path("synthesis_seed_recipe.json"),
    "--output",
    "-o",
    help="Destination JSON path for Voronoi seed recipes.",
    show_default=True,
)

SEED_RECIPE_INPUT_OPTION = typer.Option(
    ...,
    "--seed-recipe",
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    help="Seed recipe JSON (exported via synthesis-generate-seeds with --include-polygons).",
)

STAND_ATTRIBUTES_INPUT_OPTION = typer.Option(
    ...,
    "--attributes",
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    help="Stand attribute JSON produced by synthesis-sample-attributes.",
)

STAND_GEOJSON_OUTPUT_OPTION = typer.Option(
    Path("synthesis_stands.geojson"),
    "--output",
    "-o",
    help="Destination GeoJSON path for stand polygons + attributes.",
    show_default=True,
)

MASK_GEOJSON_OPTION = cast(
    Any,
    typer.Option(
        None,
        "--mask-geojson",
        help="GeoJSON files for mask overlays (repeat: defaults to clip mode).",
        show_default=False,
    ),  # type: ignore[misc, call-overload]
)

MASK_MODE_OPTION = cast(
    Any,
    typer.Option(
        None,
        "--mask-mode",
        help="Mode per --mask-geojson (clip/exclude). Matches option order; defaults to clip.",
        show_default=False,
    ),  # type: ignore[misc, call-overload]
)

MASK_NAME_OPTION = cast(
    Any,
    typer.Option(
        None,
        "--mask-name",
        help="Optional label per --mask-geojson entry (matches order).",
        show_default=False,
    ),  # type: ignore[misc, call-overload]
)

MASK_RASTER_OPTION = cast(
    Any,
    typer.Option(
        None,
        "--mask-raster",
        help="Raster masks (.npy/.npz/.csv/.txt) spanning the seed bounding box (repeatable).",
        show_default=False,
    ),  # type: ignore[misc, call-overload]
)

MASK_RASTER_THRESHOLD_OPTION = cast(
    Any,
    typer.Option(
        None,
        "--mask-raster-threshold",
        help="Threshold per --mask-raster (matches order; defaults to 0.0).",
        show_default=False,
    ),  # type: ignore[misc, call-overload]
)

MASK_RASTER_MODE_OPTION = cast(
    Any,
    typer.Option(
        None,
        "--mask-raster-mode",
        help="Mode per --mask-raster (keep/exclude); matches option order, defaults to keep.",
        show_default=False,
    ),  # type: ignore[misc, call-overload]
)

MASK_RASTER_NAME_OPTION = cast(
    Any,
    typer.Option(
        None,
        "--mask-raster-name",
        help="Optional label per --mask-raster entry (matches order).",
        show_default=False,
    ),  # type: ignore[misc, call-overload]
)

LAYOUT_MODE_OPTION = typer.Option(
    tessellation.SeedLayoutMode.RANDOM,
    "--layout",
    help="Seed layout strategy: random mix, hex grid, imported points, or geojson centroids.",
    case_sensitive=False,
    show_default=True,
)

LAYOUT_POINTS_OPTION = typer.Option(
    None,
    "--layout-points",
    exists=True,
    readable=True,
    help="Path to CSV/JSON file containing imported seed coordinates (x,y).",
    show_default=False,
)

LAYOUT_GEOJSON_OPTION = cast(
    Any,
    typer.Option(
        None,
        "--layout-geojson",
        help="GeoJSON polygons used for deterministic centroid layouts (repeatable).",
        show_default=False,
    ),  # type: ignore[misc, call-overload]
)

STAND_TEMPLATE_PATH_OPTION = typer.Option(
    None,
    "--templates",
    "-t",
    help="JSON file describing stand templates (vegetation, age-class CDF, Weibull params).",
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    show_default=False,
)

STAND_ATTRIBUTE_OUTPUT_OPTION = typer.Option(
    Path("stand_attributes.json"),
    "--output",
    "-o",
    help="Destination JSON for sampled stand attributes.",
    show_default=True,
)

STAND_TOTAL_AREA_OPTION = typer.Option(
    10.0,
    "--total-area",
    min=0.01,
    help="Target area (hectares) to fill with sampled stand attributes.",
    show_default=True,
)

STAND_SEED_OPTION = typer.Option(
    None,
    "--seed",
    help="Optional RNG seed for reproducible attribute sampling.",
    show_default=False,
)

STAND_BOOTSTRAP_PLAN_OPTION = typer.Option(
    None,
    "--plan",
    "-p",
    help="JSON file describing stand→bootstrap rules (see docs/howto/synthesis.md).",
    exists=True,
    file_okay=True,
    dir_okay=False,
    readable=True,
    show_default=False,
)

STAND_BOOTSTRAP_OUTPUT_OPTION = typer.Option(
    Path("stand_bootstrap_manifest.json"),
    "--output",
    "-o",
    help="Destination JSON for stand→bootstrap assignments.",
    show_default=True,
)

STAND_ID_PREFIX_OPTION = typer.Option(
    "stand",
    "--id-prefix",
    help="Prefix for generated stand identifiers (stand-ID).",
    show_default=True,
)

BOOTSTRAP_DBH_OUTPUT_OPTION = typer.Option(
    Path("bootstrap_dbh.json"),
    "--output",
    "-o",
    help="Destination JSON path for DBH vectors + metadata.",
    show_default=True,
)

BOOTSTRAP_DBH_TABLE_OPTION = typer.Option(
    None,
    "--table-output",
    help="Optional CSV/Parquet path for the long-form bootstrap table.",
    show_default=False,
)

BOOTSTRAP_DBH_STAND_ID_OPTION = typer.Option(
    None,
    "--stand-id",
    help="Override the stand identifier embedded in exports (defaults to file stem).",
    show_default=False,
)
BAF_OPTION = typer.Option(..., "--baf", help="Basal area factor used for the HPS tally.")

REFERENCE_DATA_OUTPUT = Path("reference-data")
REFERENCE_DATA_URL = "https://github.com/UBC-FRESH/nemora-data.git"

OUTPUT_OPTION = typer.Option(
    REFERENCE_DATA_OUTPUT,
    "--output",
    help="Destination directory for the reference dataset.",
)

DATASET_OPTION = typer.Option(
    REFERENCE_DATA_URL,
    "--dataset-url",
    help="DataLad-compatible dataset URL to install.",
    show_default=False,
)

DRY_RUN_OPTION = typer.Option(
    True,
    "--dry-run/--no-dry-run",
    help="When set, only prints the commands without executing them.",
    show_default=True,
)

VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose output.")
VERSION_OPTION = typer.Option(False, "--version", help="Show version and exit.")

DISTRIBUTIONS_OPTION = typer.Option(
    None,
    "--distribution",
    "-d",
    help="Restrict fits to specific distributions (repeat for multiples).",
    show_default=False,
)

SHOW_PARAMETERS_OPTION = typer.Option(
    False,
    "--show-parameters/--hide-parameters",
    help="Include fitted parameter values in the summary table.",
    show_default=False,
)

GROUPED_WEIBULL_MODE_OPTION = typer.Option(
    "auto",
    "--grouped-weibull-mode",
    help="Grouped Weibull solver mode: auto (default), ls, or mle.",
    show_default=True,
)

FAIB_ROOT_ARGUMENT = typer.Argument(
    ...,
    exists=True,
    file_okay=False,
    dir_okay=True,
    readable=True,
    help="Directory containing FAIB CSV extracts (faib_tree_detail.csv, faib_sample_byvisit.csv).",
)

FAIB_OUTPUT_OPTION = typer.Option(
    None,
    "--output",
    "-o",
    help="Optional path to write the stand table CSV.",
    show_default=False,
)

FAIB_FETCH_OPTION = typer.Option(
    False,
    "--fetch/--no-fetch",
    help="Download required FAIB CSV files before building the stand table.",
    show_default=True,
)

FAIB_MANIFEST_FETCH_OPTION = typer.Option(
    True,
    "--fetch/--no-fetch",
    help="Download FAIB CSV files before building the manifest (defaults to fetch when no source).",
    show_default=True,
)

FAIB_MANIFEST_DESTINATION_ARGUMENT = typer.Argument(
    ...,
    help="Directory where the FAIB manifest and stand tables will be written.",
    file_okay=False,
    dir_okay=True,
    writable=True,
)

FAIB_DATASET_OPTION = typer.Option(
    "psp",
    "--dataset",
    help="FAIB dataset to process (psp or non_psp).",
    show_default=True,
)

FAIB_CACHE_OPTION = typer.Option(
    None,
    "--cache-dir",
    help="Destination directory for downloaded FAIB files (defaults to root when omitted).",
    show_default=False,
)

FAIB_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite/--keep-existing",
    help="Re-download FAIB CSV files even when present in the cache directory.",
    show_default=True,
)

FAIB_AUTO_BAF_OPTION = typer.Option(
    False,
    "--auto-bafs/--no-auto-bafs",
    help="Automatically select representative BAF values when generating stand tables.",
    show_default=False,
)

FAIB_SOURCE_OPTION = typer.Option(
    None,
    "--source",
    "-s",
    help="Existing FAIB download directory (skip download when provided).",
    show_default=False,
)

FAIB_BAFS_OPTION = typer.Option(
    None,
    "--baf",
    help="Explicit BAF values to include (repeat for multiple).",
    show_default=False,
)

FAIB_AUTO_COUNT_OPTION = typer.Option(
    3,
    "--auto-count",
    help="Number of representative BAFs to suggest when --auto-bafs is enabled.",
    show_default=True,
)

FAIB_MAX_ROWS_OPTION = typer.Option(
    None,
    "--max-rows",
    help="Limit the number of rows kept in each stand table (default: keep all).",
    show_default=False,
)


FAIB_HPS_OUTPUT_OPTION = typer.Option(
    Path("data/examples/hps_baf12"),
    "--output",
    "-o",
    help="Directory where per-plot HPS tallies will be written.",
    show_default=True,
)

FAIB_HPS_MANIFEST_OPTION = typer.Option(
    None,
    "--manifest",
    help="Optional path for the manifest CSV (defaults to <output>/manifest.csv).",
    show_default=False,
)

FAIB_HPS_FETCH_OPTION = typer.Option(
    True,
    "--fetch/--no-fetch",
    help="Download FAIB PSP CSVs before building HPS tallies.",
    show_default=True,
)

FAIB_HPS_CACHE_OPTION = typer.Option(
    Path("data/external/psp/raw"),
    "--cache-dir",
    help="Cache directory for FAIB PSP downloads.",
    show_default=True,
)

FAIB_HPS_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite/--keep-existing",
    help="Redownload PSP files even when they already exist locally.",
    show_default=True,
)

FAIB_HPS_BAF_OPTION = typer.Option(
    12.0,
    "--baf",
    help="Basal area factor assigned to the output tallies.",
    show_default=True,
)

FAIB_HPS_BIN_WIDTH_OPTION = typer.Option(
    1.0,
    "--bin-width",
    help="DBH bin width in centimetres.",
    show_default=True,
)

FAIB_HPS_BIN_ORIGIN_OPTION = typer.Option(
    0.0,
    "--bin-origin",
    help="Origin for DBH bins in centimetres.",
    show_default=True,
)

FAIB_HPS_CHUNK_OPTION = typer.Option(
    200_000,
    "--chunk-size",
    help="Rows per chunk when streaming the tree detail CSV.",
    show_default=True,
)

FAIB_HPS_STATUS_OPTION = typer.Option(
    [],
    "--status",
    "-s",
    help="Tree status codes considered live (repeatable).",
    show_default=False,
)

FAIB_HPS_INCLUDE_ALL_VISITS_OPTION = typer.Option(
    False,
    "--include-all-visits/--first-visits-only",
    help="Process every visit instead of restricting to first measurements.",
    show_default=True,
)

FAIB_HPS_SAMPLE_TYPE_OPTION = typer.Option(
    [],
    "--sample-type",
    help="Restrict plots to specific sample type codes (repeatable).",
    show_default=False,
)

FAIB_HPS_MAX_PLOTS_OPTION = typer.Option(
    None,
    "--max-plots",
    help="Limit the number of plots processed (helpful for smoke tests).",
    show_default=False,
)

FAIB_HPS_ENCODING_OPTION = typer.Option(
    "latin1",
    "--encoding",
    help="Encoding used when reading the FAIB CSV files.",
    show_default=True,
)

FAIB_HPS_PLOT_HEADER_OPTION = typer.Option(
    HPS_DEFAULT_PLOT_HEADER,
    "--plot-header-file",
    help="Filename of the FAIB plot header CSV.",
    show_default=True,
)

FAIB_HPS_SAMPLE_BYVISIT_OPTION = typer.Option(
    HPS_DEFAULT_SAMPLE_BYVISIT,
    "--sample-byvisit-file",
    help="Filename of the FAIB sample-by-visit CSV.",
    show_default=True,
)

FAIB_HPS_TREE_DETAIL_OPTION = typer.Option(
    HPS_DEFAULT_TREE_DETAIL,
    "--tree-detail-file",
    help="Filename of the FAIB tree detail CSV.",
    show_default=True,
)
INGEST_BENCHMARK_REPORT_OPTION = typer.Option(
    None,
    "--report-path",
    help="Append JSON benchmark metrics to this path (newline-delimited).",
    show_default=False,
)

FAIB_HPS_DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    help="Report the plots that would be generated without writing files.",
    show_default=True,
)

FAIB_HPS_QUIET_OPTION = typer.Option(
    False,
    "--quiet/--verbose",
    help="Suppress progress output when writing files.",
    show_default=True,
)


FIA_ROOT_ARGUMENT = typer.Argument(
    ...,
    exists=True,
    file_okay=False,
    dir_okay=True,
    readable=True,
    help="Directory containing FIA CSV extracts (TREE.csv, COND.csv, PLOT.csv).",
)

FIA_OUTPUT_OPTION = typer.Option(
    None,
    "--output",
    "-o",
    help="Optional path to write the aggregated stand table CSV.",
    show_default=False,
)

FIA_TREE_FILE_OPTION = typer.Option(
    None,
    "--tree-file",
    help=(
        "Name of the FIA TREE CSV file inside the root directory (defaults to the state-specific"
        " download or TREE.csv)."
    ),
    show_default=False,
)

FIA_COND_FILE_OPTION = typer.Option(
    None,
    "--cond-file",
    help=(
        "Name of the FIA COND CSV file inside the root directory (defaults to the state-specific"
        " download or COND.csv)."
    ),
    show_default=False,
)

FIA_PLOT_FILE_OPTION = typer.Option(
    None,
    "--plot-file",
    help=(
        "Name of the FIA PLOT CSV file inside the root directory (defaults to the state-specific"
        " download or PLOT.csv)."
    ),
    show_default=False,
)

FIA_DBH_BIN_OPTION = typer.Option(
    1.0,
    "--dbh-bin-cm",
    help="DBH bin width in centimetres used for aggregation.",
    show_default=True,
)

FIA_FETCH_STATE_OPTION = typer.Option(
    None,
    "--fetch-state",
    help="Download FIA CSV tables for the specified state (two-letter code) before aggregation.",
    show_default=False,
)


FIA_OVERWRITE_OPTION = typer.Option(
    False,
    "--overwrite/--keep-existing",
    help="Re-download FIA CSV files even when present in the root directory.",
    show_default=True,
)


@app.callback(invoke_without_command=True)
def cli_callback(  # noqa: B008
    ctx: typer.Context,
    verbose: bool = VERBOSE_OPTION,
    version: bool = VERSION_OPTION,
) -> None:
    if verbose or version:
        console.print(f"[bold green]nemora {__version__}[/bold green]")
    if ctx.invoked_subcommand is None and not ctx.resilient_parsing:
        raise typer.Exit()


def _format_bound(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.6g}"


def _print_registry_metadata(entries: list[dict[str, object]], *, json_output: bool) -> None:
    if json_output:
        console.print(json.dumps(entries, indent=2, default=str))
        return
    for entry in entries:
        name = str(entry.get("name", ""))
        notes = entry.get("notes") or ""
        console.print(f"[bold]{name}[/bold] {notes}")
        parameters = cast(tuple[str, ...], entry.get("parameters") or ())
        bounds = cast(dict[str, tuple[float | None, float | None]], entry.get("bounds") or {})
        bounds_table = Table(
            title="Parameter Bounds",
            show_header=True,
            header_style="bold magenta",
        )
        bounds_table.add_column("Parameter")
        bounds_table.add_column("Lower", justify="right")
        bounds_table.add_column("Upper", justify="right")
        for param in parameters:
            lower, upper = bounds.get(param, (None, None))
            bounds_table.add_row(param, _format_bound(lower), _format_bound(upper))
        console.print(bounds_table)
        extras = entry.get("extras")
        if extras:
            console.print(f"Extras: {extras}")
        console.print()


@app.command()
def registry(  # noqa: B008
    describe: str | None = REGISTRY_DESCRIBE_OPTION,
    show_metadata: bool = REGISTRY_SHOW_METADATA_OPTION,
    json_output: bool = REGISTRY_JSON_OPTION,
) -> None:
    """List registered distributions."""
    metadata_mode = describe is not None or show_metadata or json_output
    if metadata_mode:
        if describe:
            entries = list_registry_metadata(names=[describe])
            if not entries:
                console.print(f"[red]Unknown distribution '{describe}'.[/red]")
                raise typer.Exit(code=1)
        else:
            entries = list_registry_metadata()
        _print_registry_metadata(entries, json_output=json_output)
        return

    table = Table(title="Registered Distributions")
    table.add_column("Name")
    table.add_column("Parameters")
    table.add_column("Description", overflow="fold")
    for name in list_distributions():
        dist = get_distribution(name)
        params = ", ".join(dist.parameters)
        notes = dist.notes or ""
        table.add_row(dist.name, params, notes)
    console.print(table)


@app.command()
def fit_hps(  # noqa: B008
    dbh_file: Path = DBH_FILE_ARGUMENT,
    baf: float = BAF_OPTION,
    distributions: list[str] | None = DISTRIBUTIONS_OPTION,
    show_parameters: bool = SHOW_PARAMETERS_OPTION,
    grouped_weibull_mode: str = GROUPED_WEIBULL_MODE_OPTION,
) -> None:
    """Fit distributions to HPS tallies stored in a CSV file."""
    import pandas as pd

    data = pd.read_csv(dbh_file)
    dbh = data["dbh_cm"].to_numpy()
    tally = data["tally"].to_numpy()
    chosen = tuple(distributions) if distributions else None
    try:
        results = fit_hps_inventory(
            dbh,
            tally,
            baf=baf,
            distributions=chosen,
            grouped_weibull_mode=grouped_weibull_mode,
        )
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    table = Table(title="HPS Fits", expand=True)
    table.add_column("Distribution", no_wrap=True)
    table.add_column("RSS", justify="right", no_wrap=True)
    table.add_column("AICc", justify="right", no_wrap=True)
    table.add_column("Chi^2", justify="right", no_wrap=True)
    table.add_column("Max |Res|", justify="right", no_wrap=True)

    param_columns: list[str] = []
    if show_parameters:
        seen: set[str] = set()
        for result in results:
            for name in result.parameters:
                if name not in seen:
                    seen.add(name)
                    param_columns.append(name)

    for result in results:
        rss = result.gof.get("rss", float("nan"))
        aicc = result.gof.get("aicc", float("nan"))
        chisq = result.gof.get("chisq", float("nan"))
        residual_summary = result.diagnostics.get("residual_summary", {})
        max_abs = residual_summary.get("max_abs", float("nan"))
        row = [
            result.distribution,
            _format_metric(rss),
            _format_metric(aicc),
            _format_metric(chisq),
            _format_metric(max_abs),
        ]
        table.add_row(*row)
    console.print(table)
    if param_columns:
        param_table = Table(title="Parameter Estimates", expand=True)
        param_table.add_column("Distribution", no_wrap=True)
        for name in param_columns:
            param_table.add_column(name, justify="right", no_wrap=True)
        for result in results:
            row = [result.distribution]
            for name in param_columns:
                row.append(_format_metric(result.parameters.get(name)))
            param_table.add_row(*row)
        console.print(param_table)


PARAMETER_ASSIGNMENTS_OPTION = typer.Option(
    None,
    "--param",
    "-p",
    help="Explicit parameter assignment (NAME=VALUE). Repeat for multiple parameters.",
    show_default=False,
)


@app.command("sampling-describe-bootstrap")
def sampling_describe_bootstrap(  # noqa: B008
    stand_table: Path = STAND_TABLE_ARGUMENT,
    distribution: str = typer.Option(
        "weibull",
        "--distribution",
        "-d",
        help="Distribution to bootstrap (auto-fitted when no parameters provided).",
        show_default=True,
    ),
    params: list[str] | None = PARAMETER_ASSIGNMENTS_OPTION,
    resamples: int = typer.Option(5, "--resamples", "-r", min=1, help="Bootstrap resample count."),
    sample_size: int = typer.Option(
        25,
        "--sample-size",
        "-s",
        min=1,
        help="Samples per resample.",
        show_default=True,
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Optional RNG seed for reproducible sampling.",
        show_default=False,
    ),
    show_samples: bool = typer.Option(
        False,
        "--show-samples/--hide-samples",
        help="Print a preview of sampled (resample, bin, draw) rows.",
        show_default=False,
    ),
    preview_rows: int = typer.Option(
        5,
        "--preview-rows",
        help="Number of rows to preview when --show-samples is enabled.",
        min=1,
        show_default=True,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Emit metadata + preview in JSON format.",
        show_default=False,
    ),
) -> None:
    """Inspect bootstrap metadata for synthesis consumers."""

    rng_seed = cast(numbers.Integral | None, seed)
    fit, bootstrap_result, explicit_params = _prepare_bootstrap_result(
        stand_table,
        distribution,
        params,
        resamples=resamples,
        sample_size=sample_size,
        seed=rng_seed,
    )

    payload = bootstrap_payload(bootstrap_result)
    metadata = payload.metadata
    if json_output:
        preview_records = payload.frame.head(preview_rows).to_dict(orient="records")
        console.print(
            json.dumps(
                {"metadata": _json_ready_metadata(metadata), "preview": preview_records},
                default=str,
                indent=2,
            )
        )
        return

    if not explicit_params:
        console.print(
            f"[green]Auto-fitted[/green] {fit.distribution} parameters before bootstrapping."
        )
    _render_bootstrap_metadata(metadata)
    if show_samples:
        preview_frame = payload.frame.head(preview_rows)
        if preview_frame.empty:
            console.print("[yellow]No samples generated.[/yellow]")
        else:
            sample_table = Table(title=f"Sample Preview (first {len(preview_frame)} rows)")
            sample_table.add_column("Resample", justify="right")
            sample_table.add_column("Bin (cm)", justify="right")
            sample_table.add_column("Draw (cm)", justify="right")
            for row in preview_frame.itertuples(index=False):
                resample_value = float(cast(Any, row.resample))
                bin_value = float(cast(Any, row.bin))
                draw_value = float(cast(Any, row.draw))
                sample_table.add_row(
                    str(int(resample_value)),
                    f"{bin_value:.3f}",
                    f"{draw_value:.3f}",
                )
            console.print(sample_table)


@app.command("sampling-export-bootstrap-dbh")
def sampling_export_bootstrap_dbh(  # noqa: B008
    stand_table: Path = STAND_TABLE_ARGUMENT,
    output: Path = BOOTSTRAP_DBH_OUTPUT_OPTION,
    table_output: Path | None = BOOTSTRAP_DBH_TABLE_OPTION,
    stand_id: str | None = BOOTSTRAP_DBH_STAND_ID_OPTION,
    distribution: str = typer.Option(
        "weibull",
        "--distribution",
        "-d",
        help="Distribution to bootstrap (auto-fitted when no parameters provided).",
        show_default=True,
    ),
    params: list[str] | None = PARAMETER_ASSIGNMENTS_OPTION,
    resamples: int = typer.Option(5, "--resamples", "-r", min=1, help="Bootstrap resample count."),
    sample_size: int = typer.Option(
        25,
        "--sample-size",
        "-s",
        min=1,
        help="Samples per resample.",
        show_default=True,
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Optional RNG seed for reproducible sampling.",
        show_default=False,
    ),
) -> None:
    """Export per-resample DBH vectors + metadata for downstream synthesis."""

    rng_seed = cast(numbers.Integral | None, seed)
    _, bootstrap_result, _ = _prepare_bootstrap_result(
        stand_table,
        distribution,
        params,
        resamples=resamples,
        sample_size=sample_size,
        seed=rng_seed,
    )
    stand_identifier = stand_id or stand_table.stem
    payload = bootstrap_dbh_vectors(bootstrap_result, stand_id=stand_identifier)

    json_payload = {
        "stand_id": stand_identifier,
        "metadata": _json_ready_metadata(payload.metadata),
        "dbh_vectors": {str(idx): values.tolist() for idx, values in payload.dbh_vectors.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_payload, indent=2))
    console.print(f"[green]Bootstrap DBH JSON written[/green] {output}")

    if table_output is not None:
        frame = payload.frame
        if frame is None or frame.empty:
            console.print("[yellow]No bootstrap samples available; skipping table export.[/yellow]")
        else:
            table_output.parent.mkdir(parents=True, exist_ok=True)
            if table_output.suffix.lower() == ".parquet":
                frame.to_parquet(table_output, index=False)
            else:
                frame.to_csv(table_output, index=False)
            console.print(f"[green]Bootstrap DBH table written[/green] {table_output}")


def _load_layout_points(path: Path) -> np.ndarray:
    """Load imported seed coordinates from CSV/JSON."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or {"x", "y"} - set(reader.fieldnames):
                raise ValueError("Layout CSV must include 'x' and 'y' headers.")
            coords = [[float(row["x"]), float(row["y"])] for row in reader]
    elif suffix in {".json", ".geojson"}:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            coords = payload
        elif isinstance(payload, dict) and "points" in payload:
            coords = payload["points"]
        else:
            raise ValueError(
                "Layout JSON must be a list of [x, y] pairs or contain a 'points' key."
            )
    else:
        raise ValueError("Unsupported layout file extension; use .csv or .json/.geojson.")
    array = np.asarray(coords, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Layout coordinates must form an (n, 2) array.")
    if array.size == 0:
        raise ValueError("Layout file did not contain any coordinates.")
    return array


def _ensure_parallel_length(label: str, values: Sequence[object], expected: int) -> None:
    if values and len(values) != expected:
        console.print(
            "[red]"
            f"{label} expects {expected} entries to match the paired option "
            f"(received {len(values)})."
            "[/red]"
        )
        raise typer.Exit(code=1)


def _parse_mask_mode_option(value: str) -> tessellation.MaskMode:
    normalized = value.strip().lower()
    try:
        return tessellation.MaskMode(normalized)
    except ValueError as exc:  # noqa: PERF203
        raise ValueError("Mask mode must be 'clip' or 'exclude'.") from exc


def _parse_raster_mode_option(value: str) -> tessellation.RasterMode:
    normalized = value.strip().lower()
    try:
        return tessellation.RasterMode(normalized)
    except ValueError as exc:
        raise ValueError("Raster mode must be 'keep' or 'exclude'.") from exc


@app.command("synthesis-generate-seeds")
def synthesis_generate_seeds(  # noqa: B008
    count: int = typer.Option(200, "--count", "-c", min=1, help="Target polygon/seed count."),
    aspect_ratio: float = typer.Option(
        1.0,
        "--aspect-ratio",
        "-a",
        min=0.01,
        help="Width/height ratio for the bounding rectangle.",
        show_default=True,
    ),
    mix_uniform: float = typer.Option(
        1.0,
        "--mix-uniform",
        help="Weight for the uniform (CSR) point process.",
        show_default=True,
    ),
    mix_cluster: float = typer.Option(
        0.0,
        "--mix-cluster",
        help="Weight for the clustered process.",
        show_default=True,
    ),
    mix_inhibition: float = typer.Option(
        0.0,
        "--mix-inhibition",
        help="Weight for the inhibition (SSI) process.",
        show_default=True,
    ),
    mix_lattice: float = typer.Option(
        0.0,
        "--mix-lattice",
        help="Weight for the lattice/grid process.",
        show_default=True,
    ),
    cluster_size: int = typer.Option(
        6,
        "--cluster-size",
        min=1,
        help="Approximate number of points per cluster.",
        show_default=True,
    ),
    cluster_spread: float = typer.Option(
        0.05,
        "--cluster-spread",
        min=0.0,
        help="Cluster spread expressed as a fraction of the unit square.",
        show_default=True,
    ),
    inhibition_min_distance: float | None = typer.Option(
        None,
        "--inhibition-min-distance",
        help="Optional SSI minimum distance (defaults to aspect-ratio dependent heuristic).",
        show_default=False,
    ),
    inhibition_attempts: int = typer.Option(
        500,
        "--inhibition-attempts",
        min=1,
        help="Maximum SSI placement attempts per point when no deterministic layout exists.",
        show_default=True,
    ),
    lattice_nx: int | None = typer.Option(
        None,
        "--lattice-nx",
        min=1,
        help="Optional fixed lattice resolution (columns). Provide with --lattice-ny.",
        show_default=False,
    ),
    lattice_ny: int | None = typer.Option(
        None,
        "--lattice-ny",
        min=1,
        help="Optional fixed lattice resolution (rows). Provide with --lattice-nx.",
        show_default=False,
    ),
    lattice_jitter: float = typer.Option(
        0.0,
        "--lattice-jitter",
        min=0.0,
        help="Standard deviation (fraction of cell size) for lattice jitter.",
        show_default=True,
    ),
    hole_fraction: float = typer.Option(
        0.0,
        "--hole-fraction",
        min=0.0,
        help="CJFR hole fraction (p_H) applied after tessellation.",
        show_default=True,
    ),
    merge_fraction: float = typer.Option(
        0.0,
        "--merge-fraction",
        min=0.0,
        help="CJFR merge fraction (p_M) applied after tessellation.",
        show_default=True,
    ),
    seed: int | None = typer.Option(
        None,
        "--seed",
        help="Optional RNG seed for reproducible seed placement.",
        show_default=False,
    ),
    layout_mode: tessellation.SeedLayoutMode = LAYOUT_MODE_OPTION,
    layout_points_path: Path | None = LAYOUT_POINTS_OPTION,
    layout_geojson: list[Path] | None = LAYOUT_GEOJSON_OPTION,
    include_points: bool = typer.Option(
        True,
        "--include-points/--metadata-only",
        help="Toggle writing raw coordinates alongside metadata.",
        show_default=True,
    ),
    include_polygons: bool = typer.Option(
        False,
        "--include-polygons/--no-polygons",
        help="Toggle writing Voronoi polygons to the seed recipe JSON.",
        show_default=True,
    ),
    mask_geojson: list[Path] | None = MASK_GEOJSON_OPTION,
    mask_modes: list[str] | None = MASK_MODE_OPTION,
    mask_names: list[str] | None = MASK_NAME_OPTION,
    mask_rasters: list[Path] | None = MASK_RASTER_OPTION,
    mask_raster_thresholds: list[float] | None = MASK_RASTER_THRESHOLD_OPTION,
    mask_raster_modes: list[str] | None = MASK_RASTER_MODE_OPTION,
    mask_raster_names: list[str] | None = MASK_RASTER_NAME_OPTION,
    output: Path = SEED_RECIPE_OUTPUT_OPTION,
) -> None:
    """Generate Voronoi seed recipes and export them as JSON."""

    if (lattice_nx is None) ^ (lattice_ny is None):
        console.print(
            "[red]Provide both --lattice-nx and --lattice-ny when fixing resolution.[/red]"
        )
        raise typer.Exit(code=1)

    lattice_resolution = None
    if lattice_nx is not None and lattice_ny is not None:
        lattice_resolution = (lattice_nx, lattice_ny)

    mask_geojson = mask_geojson or []
    mask_modes = mask_modes or []
    mask_names = mask_names or []
    mask_rasters = mask_rasters or []
    mask_raster_thresholds = mask_raster_thresholds or []
    mask_raster_modes = mask_raster_modes or []
    mask_raster_names = mask_raster_names or []
    layout_geojson = layout_geojson or []

    if layout_mode == tessellation.SeedLayoutMode.IMPORTED and layout_points_path is None:
        console.print("[red]--layout imported requires --layout-points.[/red]")
        raise typer.Exit(code=1)
    if layout_mode != tessellation.SeedLayoutMode.IMPORTED and layout_points_path is not None:
        console.print("[red]--layout-points is only valid with --layout imported.[/red]")
        raise typer.Exit(code=1)
    if layout_mode == tessellation.SeedLayoutMode.GEOJSON and not layout_geojson:
        console.print("[red]--layout geojson requires at least one --layout-geojson file.[/red]")
        raise typer.Exit(code=1)
    if layout_mode != tessellation.SeedLayoutMode.GEOJSON and layout_geojson:
        console.print("[red]--layout-geojson is only valid with --layout geojson.[/red]")
        raise typer.Exit(code=1)

    rng = np.random.default_rng(seed) if seed is not None else None
    mask_geometries: list[tessellation.MaskGeometry] = []
    if mask_geojson:
        _ensure_parallel_length("--mask-mode", mask_modes, len(mask_geojson))
        _ensure_parallel_length("--mask-name", mask_names, len(mask_geojson))
        for idx, geo_path in enumerate(mask_geojson):
            name_value = mask_names[idx] if idx < len(mask_names) else None
            mode_value = mask_modes[idx] if idx < len(mask_modes) else "clip"
            try:
                mask_geometries.append(
                    tessellation.load_mask_from_geojson(
                        geo_path,
                        name=name_value,
                        mode=_parse_mask_mode_option(mode_value),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                console.print(f"[red]Failed to parse mask geojson:[/red] {exc}")
                raise typer.Exit(code=1) from exc

    _ensure_parallel_length("--mask-raster-threshold", mask_raster_thresholds, len(mask_rasters))
    _ensure_parallel_length("--mask-raster-mode", mask_raster_modes, len(mask_rasters))
    _ensure_parallel_length("--mask-raster-name", mask_raster_names, len(mask_rasters))
    raster_masks: list[tessellation.RasterMask] = []
    for idx, raster_path in enumerate(mask_rasters):
        threshold = mask_raster_thresholds[idx] if idx < len(mask_raster_thresholds) else 0.0
        mode_value = mask_raster_modes[idx] if idx < len(mask_raster_modes) else "keep"
        raster_name = mask_raster_names[idx] if idx < len(mask_raster_names) else None
        try:
            raster_masks.append(
                tessellation.load_raster_mask(
                    raster_path,
                    threshold=threshold,
                    mode=_parse_raster_mode_option(mode_value),
                    name=raster_name,
                )
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed to parse raster mask:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    layout_points = None
    layout_source = None
    if layout_points_path is not None:
        try:
            layout_points = _load_layout_points(layout_points_path)
            layout_source = layout_points_path.name
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed to read layout points:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    layout_polygons: list[np.ndarray] | None = None
    if layout_geojson:
        layout_polygons = []
        try:
            for geo_path in layout_geojson:
                layout_polygons.extend(tessellation.load_polygons_from_geojson(geo_path))
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed to parse layout geojson:[/red] {exc}")
            raise typer.Exit(code=1) from exc

    try:
        config = tessellation.VoronoiSeedConfig(
            count=count,
            aspect_ratio=aspect_ratio,
            mix=tessellation.PointProcessMix(
                uniform=mix_uniform,
                cluster=mix_cluster,
                inhibition=mix_inhibition,
                lattice=mix_lattice,
            ),
            cluster=tessellation.ClusterConfig(size=cluster_size, spread=cluster_spread),
            inhibition=tessellation.InhibitionConfig(
                min_distance=inhibition_min_distance,
                max_attempts_per_point=inhibition_attempts,
            ),
            lattice=tessellation.LatticeConfig(
                resolution=lattice_resolution,
                jitter=lattice_jitter,
            ),
            edit=tessellation.VoronoiEditConfig(
                hole_fraction=hole_fraction,
                merge_fraction=merge_fraction,
            ),
            mask=mask_geometries[0] if mask_geometries else None,
            mask_overlays=mask_geometries[1:] if len(mask_geometries) > 1 else [],
            raster_masks=raster_masks,
            layout=tessellation.SeedLayoutConfig(
                mode=layout_mode,
                points=layout_points,
                source=layout_source,
                geojson_polygons=layout_polygons,
            ),
            rng=rng,
        )
    except ValueError as exc:  # noqa: BLE001
        console.print(f"[red]Invalid seed configuration:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        result = tessellation.generate_seed_points(config)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to generate seeds:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    exporters.export_seed_recipe(
        result,
        output,
        include_points=include_points,
        include_polygons=include_polygons,
    )
    metrics = result.metrics
    mask_suffix = ""
    mask_bits: list[str] = []
    if mask_geometries:
        mask_bits.append(f"vector_masks={len(mask_geometries)}")
    if raster_masks:
        mask_bits.append(f"raster_masks={len(raster_masks)}")
    if mask_bits:
        mask_suffix = ", " + "; ".join(mask_bits)
    console.print(
        "[green]Seed recipe written[/green] "
        f"{output} (area CV={metrics.area_cv:.3f}, μ_d={metrics.vertex_degree_mean:.2f}"
        f"{mask_suffix})"
    )


@app.command("synthesis-sample-attributes")
def synthesis_sample_attributes_cli(  # noqa: B008
    templates_path: Path = STAND_TEMPLATE_PATH_OPTION,
    total_area: float = STAND_TOTAL_AREA_OPTION,
    seed: int | None = STAND_SEED_OPTION,
    output: Path = STAND_ATTRIBUTE_OUTPUT_OPTION,
) -> None:
    """Sample stand attributes from template JSON and export as a manifest."""

    if total_area <= 0:
        console.print("[red]--total-area must be positive.[/red]")
        raise typer.Exit(code=1)
    try:
        templates = stands.load_templates_from_json(templates_path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to load templates:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    rng = np.random.default_rng(seed) if seed is not None else None
    try:
        samples = stands.sample_stand_attributes(templates, total_area=total_area, rng=rng)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to sample stand attributes:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    payload = [
        {
            "vegetation_type": sample.vegetation_type,
            "age_class": sample.age_class,
            "area": sample.area,
        }
        for sample in samples
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    console.print(
        "[green]Stand attributes sampled[/green] "
        f"{len(samples)} patches (~{sum(p.area for p in samples):.2f} ha) → {output}"
    )


@app.command("synthesis-assign-stands")
def synthesis_assign_stands_cli(  # noqa: B008
    seed_recipe_path: Path = SEED_RECIPE_INPUT_OPTION,
    attributes_path: Path = STAND_ATTRIBUTES_INPUT_OPTION,
    output: Path = STAND_GEOJSON_OUTPUT_OPTION,
    crs: str | None = typer.Option(
        None,
        "--crs",
        help="Optional CRS identifier to record in the GeoJSON.",
        show_default=False,
    ),
    strict: bool = typer.Option(
        False,
        "--strict-count/--allow-truncate",
        help="Require attribute/sample counts to match non-empty polygons.",
        show_default=True,
    ),
) -> None:
    """Attach stand attributes to Voronoi polygons and export GeoJSON."""

    try:
        recipe = json.loads(seed_recipe_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to read seed recipe:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    raw_polygons = recipe.get("polygons")
    if not isinstance(raw_polygons, list):
        console.print(
            "[red]Seed recipe is missing polygons. Re-export with --include-polygons enabled.[/red]"
        )
        raise typer.Exit(code=1)
    polygons: list[np.ndarray] = []
    try:
        for coords in raw_polygons:
            array = np.asarray(coords, dtype=float)
            polygons.append(array)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Invalid polygon coordinates in recipe:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        samples = stands.load_samples_from_json(attributes_path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to read stand attributes:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    valid_polygons = [poly for poly in polygons if poly.size > 0]
    valid_polygon_count = len(valid_polygons)
    if not valid_polygon_count or not samples:
        console.print("[red]No polygons or stand samples available for assignment.[/red]")
        raise typer.Exit(code=1)
    if strict and valid_polygon_count != len(samples):
        console.print(
            "[red]Strict mode requires polygon and sample counts to match "
            f"(polygons={valid_polygon_count}, samples={len(samples)}).[/red]"
        )
        raise typer.Exit(code=1)

    try:
        assigned = exporters.export_stand_geojson_from_polygons(
            valid_polygons,
            samples,
            output,
            crs=crs,
            strict=strict,
            expected_count=min(valid_polygon_count, len(samples)),
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to build stand GeoJSON:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if assigned < len(samples):
        console.print(
            "[yellow]Warning:[/yellow] unused samples "
            f"(assigned {assigned} / provided {len(samples)})."
        )
    if assigned < valid_polygon_count:
        console.print(
            "[yellow]Warning:[/yellow] only "
            f"{assigned} of {valid_polygon_count} polygons received attributes."
        )
    console.print(
        "[green]Stand GeoJSON written[/green] "
        f"{output} (stands={assigned}, crs={crs or 'unspecified'})"
    )


@app.command("synthesis-link-bootstraps")
def synthesis_link_bootstraps_cli(  # noqa: B008
    attributes_path: Path = STAND_ATTRIBUTES_INPUT_OPTION,
    plan_path: Path = STAND_BOOTSTRAP_PLAN_OPTION,
    output: Path = STAND_BOOTSTRAP_OUTPUT_OPTION,
    id_prefix: str = STAND_ID_PREFIX_OPTION,
) -> None:
    """Attach bootstrap DBH payloads to sampled stand attributes."""

    if plan_path is None:
        console.print("[red]--plan is required.[/red]")
        raise typer.Exit(code=1)
    try:
        samples = stands.load_samples_from_json(attributes_path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to read stand attributes:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    if not samples:
        console.print("[red]No stand attributes available for linking.[/red]")
        raise typer.Exit(code=1)
    try:
        plan = stands.load_bootstrap_plan(plan_path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to load bootstrap plan:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    try:
        assignments, library = stands.build_bootstrap_assignments(
            samples,
            plan,
            id_prefix=id_prefix,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to link bootstrap payloads:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    manifest = {
        "attributes_source": str(attributes_path),
        "plan_source": str(plan_path),
        "bootstraps": {
            identifier: {
                "source": entry.source,
                "metadata": entry.metadata,
                "dbh_vectors": entry.dbh_vectors,
            }
            for identifier, entry in sorted(library.items())
        },
        "assignments": [
            {
                "stand_id": assignment.stand_id,
                "vegetation_type": assignment.vegetation_type,
                "age_class": assignment.age_class,
                "area": assignment.area,
                "bootstrap_id": assignment.bootstrap_id,
            }
            for assignment in assignments
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    console.print(
        "[green]Linked stand attributes to bootstrap payloads[/green] "
        f"(stands={len(assignments)}, bootstraps={len(library)}) → {output}"
    )


@app.command("ingest-faib")
def ingest_faib(  # noqa: B008
    root: Path = FAIB_ROOT_ARGUMENT,
    baf: float = typer.Option(
        12.0, "--baf", help="Basal area factor to filter (ignored when --auto-bafs is set)."
    ),
    dataset: str = FAIB_DATASET_OPTION,
    fetch: bool = FAIB_FETCH_OPTION,
    cache_dir: Path | None = FAIB_CACHE_OPTION,
    overwrite: bool = FAIB_OVERWRITE_OPTION,
    auto_bafs: bool = FAIB_AUTO_BAF_OPTION,
    output: Path | None = FAIB_OUTPUT_OPTION,
) -> None:
    """Generate a stand table from local FAIB PSP extracts."""
    target_root = root
    if fetch:
        destination = cache_dir or root
        try:
            dataset_source = build_faib_dataset_source(
                dataset,
                destination=destination,
                overwrite=overwrite,
            )
            downloaded = list(dataset_source.fetch())
        except Exception as exc:
            console.print(f"[red]Download failed:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        target_root = destination
        console.print(
            f"[green]Prepared[/green] {len(downloaded)} files in {destination} "
            f"(dataset={dataset}, overwrite={overwrite})"
        )
    plot_file: str | None = None
    plot_header = target_root / "faib_plot_header.csv"
    if plot_header.exists():
        plot_file = "faib_plot_header.csv"

    if auto_bafs:
        suggestions = auto_select_bafs(target_root)
        console.print(
            "[green]Suggested BAFs:[/green] "
            + ", ".join(f"{value:.4f}" for value in suggestions)
            + "\nUse `scripts/generate_faib_manifest.py --auto` to build a manifest."
        )
        raise typer.Exit()

    try:
        stand_table = build_faib_stand_table(target_root, baf, plot_file=plot_file)
    except Exception as exc:
        console.print(f"[red]Failed to build stand table:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if output is not None:
        stand_table.to_csv(output, index=False)
        console.print(
            f"[green]Stand table written[/green] {output} (rows={len(stand_table)}, baf={baf})"
        )
    else:
        console.print(stand_table.head())


@app.command("faib-manifest")
def faib_manifest(  # noqa: B008
    destination: Path = FAIB_MANIFEST_DESTINATION_ARGUMENT,
    dataset: str = FAIB_DATASET_OPTION,
    source: Path | None = FAIB_SOURCE_OPTION,
    cache_dir: Path | None = FAIB_CACHE_OPTION,
    fetch: bool = FAIB_MANIFEST_FETCH_OPTION,
    overwrite: bool = FAIB_OVERWRITE_OPTION,
    bafs: list[float] | None = FAIB_BAFS_OPTION,
    auto_bafs: bool = FAIB_AUTO_BAF_OPTION,
    auto_count: int = FAIB_AUTO_COUNT_OPTION,
    max_rows: int | None = FAIB_MAX_ROWS_OPTION,
    parquet: bool = typer.Option(
        True,
        "--parquet/--no-parquet",
        help="Write a Parquet copy (disable with --no-parquet to emit CSV only).",
        show_default=True,
    ),
) -> None:
    """Fetch FAIB extracts, generate stand tables, and emit a manifest CSV."""

    if auto_bafs and bafs:
        console.print("[red]Specify either --auto-bafs or explicit --baf values, not both.[/red]")
        raise typer.Exit(code=1)
    if not fetch and source is None and cache_dir is None:
        console.print(
            "[red]No source directory provided and downloads disabled; nothing to ingest.[/red]"
        )
        raise typer.Exit(code=1)

    effective_source = source or cache_dir
    fetch_flag = fetch
    if effective_source is None:
        effective_source = destination / "raw"
        effective_source.mkdir(parents=True, exist_ok=True)

    try:
        result: FAIBManifestResult = generate_faib_manifest(
            destination,
            dataset=dataset,
            source=effective_source,
            fetch=fetch_flag,
            overwrite=overwrite,
            bafs=bafs,
            auto_count=auto_count if auto_bafs else None,
            max_rows=max_rows,
            write_parquet=parquet,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Failed to build manifest:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if result.downloaded:
        console.print(
            f"[green]Downloaded[/green] {len(result.downloaded)} files to "
            f"{result.downloaded[0].parent} (dataset={dataset}, overwrite={overwrite})"
        )
    console.print(
        "[green]Manifest generated:[/green] "
        f"{result.manifest_path} (BAFs={', '.join(f'{b:.4f}' for b in result.bafs)})"
    )
    if parquet:
        parquet_path = result.manifest_path.with_suffix(".parquet")
        console.print(f"[green]Parquet manifest written[/green] {parquet_path}")
    for table in result.tables:
        status = "truncated" if result.truncated_flags.get(table, False) else "full"
        console.print(f"  • {table.name} ({status})")


@app.command("ingest-faib-hps")
def ingest_faib_hps(  # noqa: B008
    root: Path = FAIB_ROOT_ARGUMENT,
    output: Path = FAIB_HPS_OUTPUT_OPTION,
    manifest: Path | None = FAIB_HPS_MANIFEST_OPTION,
    fetch: bool = FAIB_HPS_FETCH_OPTION,
    cache_dir: Path = FAIB_HPS_CACHE_OPTION,
    overwrite: bool = FAIB_HPS_OVERWRITE_OPTION,
    baf: float = FAIB_HPS_BAF_OPTION,
    bin_width: float = FAIB_HPS_BIN_WIDTH_OPTION,
    bin_origin: float = FAIB_HPS_BIN_ORIGIN_OPTION,
    chunk_size: int = FAIB_HPS_CHUNK_OPTION,
    status: list[str] = FAIB_HPS_STATUS_OPTION,
    include_all_visits: bool = FAIB_HPS_INCLUDE_ALL_VISITS_OPTION,
    sample_type: list[str] = FAIB_HPS_SAMPLE_TYPE_OPTION,
    max_plots: int | None = FAIB_HPS_MAX_PLOTS_OPTION,
    encoding: str = FAIB_HPS_ENCODING_OPTION,
    plot_header_file: str = FAIB_HPS_PLOT_HEADER_OPTION,
    sample_byvisit_file: str = FAIB_HPS_SAMPLE_BYVISIT_OPTION,
    tree_detail_file: str = FAIB_HPS_TREE_DETAIL_OPTION,
    dry_run: bool = FAIB_HPS_DRY_RUN_OPTION,
    quiet: bool = FAIB_HPS_QUIET_OPTION,
) -> None:
    """Prepare HPS tallies from FAIB PSP extracts."""

    selections, tree_detail_path = _prepare_hps_inputs(
        root,
        fetch=fetch,
        cache_dir=cache_dir,
        overwrite=overwrite,
        baf=baf,
        plot_header_file=plot_header_file,
        sample_byvisit_file=sample_byvisit_file,
        tree_detail_file=tree_detail_file,
        encoding=encoding,
        include_all_visits=include_all_visits,
        sample_types=sample_type,
        max_plots=max_plots,
        quiet=quiet,
    )
    if not selections:
        console.print("[yellow]No PSP plots matched the provided filters.[/yellow]")
        raise typer.Exit(code=1)

    live_status = tuple(status) if status else ("L",)
    result = run_hps_pipeline(
        tree_detail_path,
        selections,
        dbh_column="DBH",
        status_column="LV_D",
        live_status=live_status,
        bin_width=bin_width,
        bin_origin=bin_origin,
        chunk_size=chunk_size,
        encoding=encoding,
    )

    manifest_frame = result.manifest
    if manifest_frame.empty:
        console.print(
            "[yellow]Selected plots produced no tallies. Check filters or status codes.[/yellow]"
        )
        raise typer.Exit(code=1)

    total_trees = int(manifest_frame["trees"].sum()) if "trees" in manifest_frame else 0
    plot_count = len(result.tallies)

    if dry_run:
        console.print(
            f"[cyan][dry-run][/cyan] {plot_count} plots would be written (trees={total_trees})."
        )
        raise typer.Exit()

    manifest_path = manifest or (output / "manifest.csv")
    export_hps_outputs(
        result.tallies,
        manifest_frame,
        output_dir=output,
        manifest_path=manifest_path,
        quiet=quiet,
    )
    if not quiet:
        console.print(
            f"[green]Prepared[/green] {plot_count} plots with {total_trees} live trees → {output}"
        )


@app.command("ingest-benchmark")
def ingest_benchmark(  # noqa: B008
    root: Path = FAIB_ROOT_ARGUMENT,
    iterations: int = typer.Option(
        3,
        "--iterations",
        "-n",
        help="Number of times to execute the HPS pipeline for timing.",
        min=1,
        show_default=True,
    ),
    fetch: bool = FAIB_HPS_FETCH_OPTION,
    cache_dir: Path = FAIB_HPS_CACHE_OPTION,
    overwrite: bool = FAIB_HPS_OVERWRITE_OPTION,
    baf: float = FAIB_HPS_BAF_OPTION,
    bin_width: float = FAIB_HPS_BIN_WIDTH_OPTION,
    bin_origin: float = FAIB_HPS_BIN_ORIGIN_OPTION,
    chunk_size: int = FAIB_HPS_CHUNK_OPTION,
    status: list[str] = FAIB_HPS_STATUS_OPTION,
    include_all_visits: bool = FAIB_HPS_INCLUDE_ALL_VISITS_OPTION,
    sample_type: list[str] = FAIB_HPS_SAMPLE_TYPE_OPTION,
    max_plots: int | None = FAIB_HPS_MAX_PLOTS_OPTION,
    encoding: str = FAIB_HPS_ENCODING_OPTION,
    plot_header_file: str = FAIB_HPS_PLOT_HEADER_OPTION,
    sample_byvisit_file: str = FAIB_HPS_SAMPLE_BYVISIT_OPTION,
    tree_detail_file: str = FAIB_HPS_TREE_DETAIL_OPTION,
    report_path: Path | None = INGEST_BENCHMARK_REPORT_OPTION,
) -> None:
    """Benchmark the FAIB→HPS pipeline without writing outputs."""

    selections, tree_detail_path = _prepare_hps_inputs(
        root,
        fetch=fetch,
        cache_dir=cache_dir,
        overwrite=overwrite,
        baf=baf,
        plot_header_file=plot_header_file,
        sample_byvisit_file=sample_byvisit_file,
        tree_detail_file=tree_detail_file,
        encoding=encoding,
        include_all_visits=include_all_visits,
        sample_types=sample_type,
        max_plots=max_plots,
        quiet=False,
    )
    if not selections:
        console.print("[yellow]No PSP plots matched the provided filters.[/yellow]")
        raise typer.Exit(code=1)

    live_status = tuple(status) if status else ("L",)
    durations: list[float] = []
    final_result: HPSPipelineResult | None = None
    for iteration in range(iterations):
        start = time.perf_counter()
        final_result = run_hps_pipeline(
            tree_detail_path,
            selections,
            dbh_column="DBH",
            status_column="LV_D",
            live_status=live_status,
            bin_width=bin_width,
            bin_origin=bin_origin,
            chunk_size=chunk_size,
            encoding=encoding,
        )
        durations.append(time.perf_counter() - start)
        console.print(
            f"[blue]Iteration {iteration + 1}/{iterations}[/blue]: {durations[-1]:.3f}s",
            highlight=False,
        )

    assert final_result is not None  # for mypy
    manifest = final_result.manifest
    tree_total = int(manifest["trees"].sum()) if not manifest.empty else 0
    avg = statistics.mean(durations)
    fastest = min(durations)
    slowest = max(durations)

    table = Table(title="HPS Pipeline Benchmark", show_edge=True)
    table.add_column("Runs", justify="right")
    table.add_column("Average (s)", justify="right")
    table.add_column("Fastest (s)", justify="right")
    table.add_column("Slowest (s)", justify="right")
    table.add_row(
        str(iterations),
        f"{avg:.3f}",
        f"{fastest:.3f}",
        f"{slowest:.3f}",
    )
    console.print(table)
    console.print(
        f"[green]Tree total:[/green] {tree_total} "
        f"(plots={len(final_result.tallies)}, live_status={', '.join(live_status)})"
    )
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "iterations": iterations,
            "average_seconds": avg,
            "fastest_seconds": fastest,
            "slowest_seconds": slowest,
            "tree_total": tree_total,
            "plots": len(final_result.tallies),
            "live_status": list(live_status),
            "baf": baf,
            "bin_width": bin_width,
            "bin_origin": bin_origin,
            "chunk_size": chunk_size,
        }
        with report_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics))
            handle.write("\n")
        console.print(f"[blue]Appended benchmark metrics to {report_path}[/blue]")


@app.command("ingest-fia")
def ingest_fia(  # noqa: B008
    root: Path = FIA_ROOT_ARGUMENT,
    output: Path | None = FIA_OUTPUT_OPTION,
    plot_cn: list[int] = typer.Option(  # noqa: B008
        [],
        "--plot-cn",
        help="Filter to specific FIA plot CNs (repeatable).",
        show_default=False,
    ),
    tree_file: str | None = FIA_TREE_FILE_OPTION,
    cond_file: str | None = FIA_COND_FILE_OPTION,
    plot_file: str | None = FIA_PLOT_FILE_OPTION,
    dbh_bin_cm: float = FIA_DBH_BIN_OPTION,
    fetch_state: str | None = FIA_FETCH_STATE_OPTION,
    tables: list[str] = typer.Option(  # noqa: B008
        [],
        "--table",
        "-t",
        help="FIA table names to download when --fetch-state is provided.",
        show_default=False,
    ),
    overwrite: bool = FIA_OVERWRITE_OPTION,
) -> None:
    """Aggregate FIA TREE/COND/PLOT CSV extracts into a stand table."""

    import pandas as pd

    state_upper: str | None = fetch_state.strip().upper() if fetch_state else None

    # Determine filenames, favouring state-specific downloads when available.
    def _resolve_filename(candidate: str | None, table: str, fallback: str) -> str:
        if candidate:
            return candidate
        if state_upper:
            return f"{state_upper}_{table}.csv"
        return fallback

    resolved_tree_file = _resolve_filename(tree_file, "TREE", "TREE.csv")
    resolved_cond_file = _resolve_filename(cond_file, "COND", "COND.csv")
    resolved_plot_file = _resolve_filename(plot_file, "PLOT", "PLOT.csv")

    fetch_tables = tuple(table.upper() for table in tables) if tables else ("TREE", "PLOT", "COND")

    if state_upper:
        try:
            dataset = build_fia_dataset_source(
                state_upper,
                destination=root,
                tables=fetch_tables,
                overwrite=overwrite,
            )
            downloaded = list(dataset.fetch())
        except Exception as exc:  # noqa: BLE001
            console.print(f"[red]Failed to download FIA tables:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        console.print(
            f"[green]Fetched[/green] {len(downloaded)} files to {root} "
            f"(state={state_upper}, overwrite={overwrite})"
        )

    targets: list[int | None] = list(plot_cn) if plot_cn else [None]
    frames: list[pd.DataFrame] = []
    for target in targets:
        frame = build_fia_stand_table(
            root,
            plot_cn=target,
            tree_file=resolved_tree_file,
            cond_file=resolved_cond_file,
            plot_file=resolved_plot_file,
            dbh_bin_cm=dbh_bin_cm,
        )
        if frame.empty:
            continue
        frames.append(frame)

    if not frames:
        console.print("[yellow]No FIA records matched the provided filters.[/yellow]")
        raise typer.Exit()

    result = pd.concat(frames, ignore_index=True)
    if output is not None:
        result.to_csv(output, index=False)
        console.print(
            f"[green]Stand table written[/green] {output} "
            f"(rows={len(result)}, plots={result['plot_cn'].nunique()})"
        )
    else:
        console.print(result.head())


def main_entry() -> None:
    app()


def main() -> None:  # pragma: no cover - console entry
    main_entry()


@app.command()
def fetch_reference_data(
    output: Path = OUTPUT_OPTION,
    dataset_url: str = DATASET_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    enable_remote: str = typer.Option(
        "arbutus-s3",
        "--enable-remote",
        help="Name of the DataLad sibling to enable after install (blank to skip).",
        show_default=True,
    ),
) -> None:
    """Fetch the manuscript reference dataset via DataLad (when available)."""

    _fetch_reference_data(Path(output), dataset_url, dry_run, enable_remote)


def _fetch_reference_data(
    output: Path,
    dataset_url: str,
    dry_run: bool,
    enable_remote: str | None,
) -> None:
    console.print(
        "[bold]Reference Dataset Fetch[/bold]\n"
        "This command bootstraps the manuscript dataset used in the parity notebooks.\n"
        "A DataLad installation is optional but recommended for provenance tracking."
    )
    console.print(
        f"\nDataset URL : [cyan]{dataset_url}[/cyan]\nDestination : [cyan]{output}[/cyan]"
    )

    if dry_run:
        console.print(
            "\nDry-run mode: no commands were executed.\n"
            "To perform the download locally, rerun with `--no-dry-run` after installing"
            " DataLad, e.g.\n"
            f"  datalad install --source {dataset_url} {output}\n"
            f"  datalad get {output} --recursive\n"
            f"  datalad siblings --dataset {output}\n"
            f"  datalad siblings --dataset {output} --enabled\n"
        )
        return

    try:  # pragma: no cover - optional dependency execution path
        from datalad import api as datalad_api
        from datalad.support.exceptions import IncompleteResultsError
    except ImportError as exc:  # pragma: no cover
        console.print(
            "[red]DataLad is not installed.[/red] Install it or rerun with --dry-run for"
            " instructions. For pip installs try:\n"
            '  pip install --upgrade "nemora[data]"\n'
            '  pip install "datalad[full]"\n'
            '  pip install -e ".[data]"  # from a source checkout'
        )
        raise typer.Exit(code=1) from exc

    output.mkdir(parents=True, exist_ok=True)
    console.print("\n[green]Installing dataset via DataLad...[/green]")
    try:
        install_results = datalad_api.install(
            path=str(output),
            source=dataset_url,
            on_failure="stop",
            result_renderer="disabled",
            return_type="list",
        )
    except IncompleteResultsError as exc:  # pragma: no cover
        console.print(f"[red]Installation failed:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    install_issues = _check_datalad_results("install", install_results, fatal=False)
    if install_issues:
        console.print(
            "[yellow]DataLad reported issues during install; falling back to direct git "
            "clone.[/yellow]"
        )
        _git_clone_dataset(dataset_url, output)
        return

    remote_name = (enable_remote or "").strip()
    if remote_name:
        console.print(f"[green]Enabling remote '{remote_name}'...[/green]")
        siblings_results = datalad_api.siblings(
            action="enable",
            dataset=str(output),
            name=remote_name,
            on_failure="ignore",
            result_renderer="disabled",
            return_type="list",
        )
        _check_datalad_results("enable remote", siblings_results, fatal=False)

    console.print("[green]Downloading dataset content...[/green]")
    try:
        get_results = datalad_api.get(
            path=str(output),
            on_failure="ignore",
            result_renderer="disabled",
            return_type="list",
            recursive=True,
        )
    except IncompleteResultsError as exc:  # pragma: no cover
        console.print(f"[yellow]Download reported issues:[/yellow] {exc}")
        get_results = getattr(exc, "failed", [])

    _check_datalad_results("get", get_results, fatal=False)
    console.print("[green]Dataset fetched successfully.[/green]")


def _format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, numbers.Real):
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return "-"
        return f"{val:.4f}"
    return str(value)


def _check_datalad_results(
    stage: str,
    results: Iterable[Any],
    *,
    fatal: bool = True,
) -> list[str]:
    issues = []
    for record in results:
        if not isinstance(record, dict):
            continue
        status = record.get("status")
        if status not in {"ok", "notneeded", None}:
            message = record.get("message", "")
            issues.append(f"{status}: {message}")
    if issues:
        colour = "red" if fatal else "yellow"
        console.print(
            f"[{colour}]{stage.capitalize()} encountered issues:[/{colour}]\n" + "\n".join(issues)
        )
        if fatal:
            raise typer.Exit(code=1)
    return issues


def _git_clone_dataset(dataset_url: str, output: Path) -> None:
    """Fallback clone when DataLad cannot complete the install."""
    console.print("[yellow]Running `git clone` fallback...[/yellow]")
    if output.exists():
        if output.is_dir():
            try:
                next(output.iterdir())
            except StopIteration:
                output.rmdir()
            else:
                message = (
                    "[red]Destination {path} already exists and is not empty; aborting clone.[/red]"
                )
                console.print(message.format(path=output))
                raise typer.Exit(code=1)
        else:
            message = "[red]Destination {path} exists and is not a directory; aborting clone.[/red]"
            console.print(message.format(path=output))
            raise typer.Exit(code=1)

    clone_cmd = ["git", "clone", dataset_url, str(output)]
    result = subprocess.run(  # noqa: S603
        clone_cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        message = stderr or stdout or "git clone failed without details."
        console.print(f"[red]Git clone failed:[/red] {message}")
        raise typer.Exit(code=result.returncode or 1)
    console.print("[green]Git clone completed successfully.[/green]")
