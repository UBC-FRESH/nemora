"""Helpers for working with USDA FIA datasets."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

__all__ = [
    "FIATables",
    "load_fia_tables",
    "aggregate_plot_stand_table",
    "build_stand_table_from_csvs",
    "download_fia_tables",
]


@dataclass(slots=True)
class FIATables:
    """Container for FIA plot/condition/tree tables."""

    tree: pd.DataFrame
    condition: pd.DataFrame
    plot: pd.DataFrame


def load_fia_tables(
    root: str | Path,
    *,
    tree_file: str = "TREE.csv",
    plot_file: str = "PLOT.csv",
    cond_file: str = "COND.csv",
    columns_tree: Sequence[str] | None = None,
    columns_plot: Sequence[str] | None = None,
    columns_cond: Sequence[str] | None = None,
) -> FIATables:
    """Load FIA CSV extracts from ``root`` and return trimmed dataframes."""

    root_path = Path(root)
    tree_columns = columns_tree or [
        "PLT_CN",
        "SUBP",
        "TREE",
        "CONDID",
        "STATUSCD",
        "SPCD",
        "DIA",
        "TPA_UNADJ",
    ]
    plot_columns = columns_plot or [
        "CN",
        "PLOT",
    ]
    cond_columns = columns_cond or [
        "PLT_CN",
        "CONDID",
        "COND_STATUS_CD",
        "CONDPROP_UNADJ",
        "SUBPPROP_UNADJ",
        "MICRPROP_UNADJ",
        "MACRPROP_UNADJ",
    ]

    tree_df = pd.read_csv(root_path / tree_file, usecols=tree_columns, low_memory=False)
    cond_df = pd.read_csv(root_path / cond_file, usecols=cond_columns, low_memory=False)
    plot_df = pd.read_csv(root_path / plot_file, usecols=plot_columns, low_memory=False)
    return FIATables(tree=tree_df, condition=cond_df, plot=plot_df)


def aggregate_plot_stand_table(
    tables: FIATables,
    *,
    plot_cn: int | None = None,
    plot_number: int | None = None,
    live_status_codes: Iterable[int] = (1,),
    condition_status_codes: Iterable[int] = (1,),
    dbh_bin_cm: float = 1.0,
) -> pd.DataFrame:
    """Aggregate FIA tree records into a stand table summarised by DBH bins."""

    tree = tables.tree.copy()
    condition = tables.condition.copy()
    plot = tables.plot.copy()

    if plot_cn is not None:
        tree = tree[tree["PLT_CN"] == plot_cn]
        condition = condition[condition["PLT_CN"] == plot_cn]

    if live_status_codes:
        tree = tree[tree["STATUSCD"].isin(list(live_status_codes))]
    if condition_status_codes:
        condition = condition[condition["COND_STATUS_CD"].isin(list(condition_status_codes))]

    if tree.empty or condition.empty:
        return pd.DataFrame({"plot_cn": [], "plot": [], "dbh_cm": [], "tally": []})

    merged = tree.merge(
        condition,
        on=["PLT_CN", "CONDID"],
        how="inner",
        suffixes=("", "_COND"),
    )
    if merged.empty:
        return pd.DataFrame({"plot_cn": [], "plot": [], "dbh_cm": [], "tally": []})

    weight_factor = merged["CONDPROP_UNADJ"].fillna(merged["SUBPPROP_UNADJ"])
    weight_factor = weight_factor.fillna(merged["MICRPROP_UNADJ"]).fillna(merged["MACRPROP_UNADJ"])
    weight = merged["TPA_UNADJ"].astype(float) * weight_factor.fillna(1.0)

    dbh_cm = merged["DIA"].astype(float) * 2.54
    if dbh_bin_cm > 0:
        dbh_cm = np.round(dbh_cm / dbh_bin_cm) * dbh_bin_cm
    else:
        dbh_cm = dbh_cm.round(2)

    aggregated = (
        pd.DataFrame(
            {
                "plot_cn": merged["PLT_CN"],
                "dbh_cm": dbh_cm,
                "tally": weight,
            }
        )
        .groupby(["plot_cn", "dbh_cm"], as_index=False)["tally"]
        .sum()
        .sort_values(["plot_cn", "dbh_cm"])
    )

    plot_lookup = plot.rename(columns={"CN": "plot_cn"})
    aggregated = aggregated.merge(plot_lookup, on="plot_cn", how="left")
    if plot_number is not None:
        aggregated["PLOT"] = plot_number

    aggregated = aggregated.rename(columns={"PLOT": "plot"})
    columns = ["plot_cn", "plot", "dbh_cm", "tally"]
    return aggregated[columns]


def build_stand_table_from_csvs(
    root: str | Path,
    *,
    plot_cn: int | None = None,
    plot_number: int | None = None,
    tree_file: str = "TREE.csv",
    plot_file: str = "PLOT.csv",
    cond_file: str = "COND.csv",
    live_status_codes: Iterable[int] = (1,),
    condition_status_codes: Iterable[int] = (1,),
    dbh_bin_cm: float = 1.0,
) -> pd.DataFrame:
    """Convenience wrapper that loads FIA tables and aggregates a stand table."""

    tables = load_fia_tables(
        root,
        tree_file=tree_file,
        plot_file=plot_file,
        cond_file=cond_file,
    )
    return aggregate_plot_stand_table(
        tables,
        plot_cn=plot_cn,
        plot_number=plot_number,
        live_status_codes=live_status_codes,
        condition_status_codes=condition_status_codes,
        dbh_bin_cm=dbh_bin_cm,
    )


def download_fia_tables(
    destination: str | Path,
    state: str,
    tables: Sequence[str] = ("TREE", "PLOT", "COND"),
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Download FIA CSV extracts for a given state.

    Parameters
    ----------
    destination:
        Directory where the CSV files will be written.
    state:
        Two-letter state or territory code (e.g., ``HI``, ``OR``).
    tables:
        Iterable of table names to download (default: TREE, PLOT, COND).
    overwrite:
        When ``True``, re-download files even if they already exist locally.
    """

    if not state:
        raise ValueError("state must be provided (e.g., 'HI', 'OR').")
    state_upper = state.strip().upper()
    if len(state_upper) != 2:
        raise ValueError("state must be a two-letter code.")

    allowed = {"TREE", "PLOT", "COND", "SUBPLOT", "SUBPLOT_COND"}
    normalized_tables = []
    for table in tables:
        table_upper = table.strip().upper()
        if table_upper not in allowed:
            raise ValueError(f"Unsupported FIA table '{table}'.")
        normalized_tables.append(table_upper)

    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)

    base_url = "https://apps.fs.usda.gov/fia/datamart/CSV"
    downloaded: list[Path] = []
    for table in normalized_tables:
        filename = f"{state_upper}_{table}.csv"
        target = dest_path / filename
        if target.exists() and not overwrite:
            downloaded.append(target)
            continue

        tmp_path = target.with_suffix(".part")
        if tmp_path.exists():
            tmp_path.unlink()
        url = f"{base_url}/{filename}"
        try:
            urlretrieve(url, tmp_path)  # noqa: S310 - public FIA HTTPS endpoint
        except URLError as exc:  # pragma: no cover - network failure
            if tmp_path.exists():
                tmp_path.unlink()
            raise RuntimeError(f"Failed to download FIA table {table}: {exc}") from exc
        tmp_path.replace(target)
        downloaded.append(target)
    return downloaded
