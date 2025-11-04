"""Typer-based CLI entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .distributions import get_distribution, list_distributions
from .workflows.hps import fit_hps_inventory

app = typer.Typer(help="DBH distribution fitting toolkit.")
console = Console()


@app.callback(invoke_without_command=True)
def main(
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
    dbh_file: Path = typer.Argument(..., exists=True, readable=True, help="CSV with dbh_cm,tally columns."),
    baf: float = typer.Option(..., "--baf", help="Basal area factor used for the HPS tally."),
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
