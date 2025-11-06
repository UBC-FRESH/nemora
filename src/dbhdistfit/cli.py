"""Typer-based CLI entry point."""

# ruff: noqa: S603

from __future__ import annotations

import math
import numbers
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .distributions import get_distribution, list_distributions
from .workflows.hps import fit_hps_inventory

app = typer.Typer(help="DBH distribution fitting toolkit.")
console = Console()

DBH_FILE_ARGUMENT = typer.Argument(
    ...,
    exists=True,
    readable=True,
    help="CSV with `dbh_cm` and `tally` columns.",
)
BAF_OPTION = typer.Option(..., "--baf", help="Basal area factor used for the HPS tally.")

REFERENCE_DATA_OUTPUT = Path("reference-data")
REFERENCE_DATA_URL = "https://github.com/UBC-FRESH/dbhdistfit-data.git"

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


@app.callback(invoke_without_command=True)
def cli_callback(  # noqa: B008
    ctx: typer.Context,
    verbose: bool = VERBOSE_OPTION,
    version: bool = VERSION_OPTION,
) -> None:
    if verbose or version:
        console.print(f"[bold green]dbhdistfit {__version__}[/bold green]")
    if ctx.invoked_subcommand is None and not ctx.resilient_parsing:
        raise typer.Exit()


@app.command()
def registry() -> None:
    """List registered distributions."""
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
            '  pip install --upgrade "dbhdistfit[data]"\n'
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
