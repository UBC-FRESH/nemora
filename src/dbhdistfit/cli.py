"""Typer-based CLI entry point."""

from __future__ import annotations

from pathlib import Path

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


@app.callback(invoke_without_command=True)
def cli_callback(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output."),
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
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
def fit_hps(
    dbh_file: Path = DBH_FILE_ARGUMENT,
    baf: float = BAF_OPTION,
) -> None:
    """Fit distributions to HPS tallies stored in a CSV file."""
    import pandas as pd

    data = pd.read_csv(dbh_file)
    dbh = data["dbh_cm"].to_numpy()
    tally = data["tally"].to_numpy()
    results = fit_hps_inventory(dbh, tally, baf=baf)
    table = Table(title="HPS Fits")
    table.add_column("Distribution")
    table.add_column("RSS", justify="right")
    for result in results:
        rss = result.gof.get("rss", float("nan"))
        table.add_row(result.distribution, f"{rss:.4f}")
    console.print(table)


def main_entry() -> None:
    app()


def main() -> None:  # pragma: no cover - console entry
    main_entry()


@app.command()
def fetch_reference_data(
    output: Path = OUTPUT_OPTION,
    dataset_url: str = DATASET_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Fetch the manuscript reference dataset via DataLad (when available)."""

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
            f"  datalad get {output}\n"
        )
        return

    try:  # pragma: no cover - optional dependency execution path
        from datalad import api as datalad_api
    except ImportError as exc:  # pragma: no cover
        console.print(
            "[red]DataLad is not installed.[/red] Install it or rerun with --dry-run for"
            ' instructions. For pip users, install via `pip install "datalad[full]"`.'
        )
        raise typer.Exit(code=1) from exc

    output.mkdir(parents=True, exist_ok=True)
    console.print("\n[green]Installing dataset via DataLad...[/green]")
    datalad_api.install(path=str(output), source=dataset_url)
    datalad_api.get(str(output))
    console.print("[green]Dataset fetched successfully.[/green]")
