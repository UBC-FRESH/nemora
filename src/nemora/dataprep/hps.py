"""Helpers for sourcing HPS tallies from the BC FAIB PSP compilations."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _normalise_cluster(identifier: str) -> str:
    """Return a filesystem friendly cluster identifier."""
    return identifier.replace("/", "_").replace("\\", "_").replace(" ", "_").replace("-", "_")


@dataclass(frozen=True)
class SelectionCriteria:
    """Options that control which plots are retained from the PSP compilations."""

    first_visit_only: bool = True
    allowed_sample_types: tuple[str, ...] | None = None
    max_plots: int | None = None


@dataclass(frozen=True)
class PlotSelection:
    """Metadata describing a PSP plot visit selected for HPS tallies."""

    plot_id: str
    cluster_id: str
    visit_number: int
    plot: int
    baf: float
    measurement_date: pd.Timestamp | None
    sample_type: str | None
    site_identifier: str | None

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.cluster_id, self.visit_number, self.plot)


def load_plot_selections(
    plot_header_source: str | Path,
    sample_byvisit_source: str | Path,
    *,
    baf: float,
    criteria: SelectionCriteria | None = None,
    encoding: str = "latin1",
) -> list[PlotSelection]:
    """Load PSP plot metadata and filter to the subset needed for HPS tallies."""

    criteria = criteria or SelectionCriteria()
    plot_header = pd.read_csv(
        plot_header_source,
        encoding=encoding,
        dtype={
            "CLSTR_ID": "string",
            "VISIT_NUMBER": "int64",
            "PLOT": "int64",
            "SITE_IDENTIFIER": "string",
        },
    )
    sample_byvisit = pd.read_csv(
        sample_byvisit_source,
        encoding=encoding,
        dtype={
            "CLSTR_ID": "string",
            "VISIT_NUMBER": "int64",
            "SAMP_TYP": "string",
            "FIRST_MSMT": "string",
        },
        parse_dates=["MEAS_DT"],
    )

    merged = plot_header.merge(
        sample_byvisit[
            [
                "CLSTR_ID",
                "VISIT_NUMBER",
                "FIRST_MSMT",
                "MEAS_DT",
                "SAMP_TYP",
                "SAMPLE_ESTABLISHMENT_TYPE",
            ]
        ],
        on=["CLSTR_ID", "VISIT_NUMBER"],
        how="inner",
        suffixes=("_plot", "_visit"),
    )

    if criteria.allowed_sample_types:
        mask = merged["SAMP_TYP"].isin(criteria.allowed_sample_types)
        merged = merged.loc[mask]

    if criteria.first_visit_only and "FIRST_MSMT" in merged:
        merged = merged.loc[merged["FIRST_MSMT"].str.upper() == "Y"]

    merged = merged.sort_values(["CLSTR_ID", "VISIT_NUMBER", "PLOT"])

    if criteria.max_plots is not None:
        merged = merged.head(criteria.max_plots)

    selections: list[PlotSelection] = []
    for row in merged.itertuples(index=False):
        cluster_id = str(row.CLSTR_ID)
        visit_number = int(row.VISIT_NUMBER)
        plot = int(row.PLOT)
        slug = f"{_normalise_cluster(cluster_id)}_v{visit_number}_p{plot}"
        measurement_date = pd.to_datetime(row.MEAS_DT).date() if not pd.isna(row.MEAS_DT) else None
        sample_type = None
        if hasattr(row, "SAMP_TYP") and isinstance(row.SAMP_TYP, str):
            sample_type = row.SAMP_TYP.strip() or None
        site_identifier = None
        if hasattr(row, "SITE_IDENTIFIER") and isinstance(row.SITE_IDENTIFIER, str):
            site_identifier = row.SITE_IDENTIFIER.strip() or None
        selections.append(
            PlotSelection(
                plot_id=slug,
                cluster_id=cluster_id,
                visit_number=visit_number,
                plot=plot,
                baf=baf,
                measurement_date=measurement_date,
                sample_type=sample_type,
                site_identifier=site_identifier,
            )
        )

    return selections


def _bin_dbh(values: np.ndarray, *, bin_width: float, bin_origin: float) -> np.ndarray:
    """Return bin midpoints for DBH values."""
    if bin_width <= 0:
        raise ValueError("bin_width must be positive.")
    bin_index = np.floor((values - bin_origin) / bin_width)
    return bin_origin + (bin_index + 0.5) * bin_width


def aggregate_hps_tallies(
    tree_detail_source: str | Path,
    selections: Sequence[PlotSelection],
    *,
    dbh_column: str = "DBH",
    status_column: str | None = "LV_D",
    live_status: Iterable[str] = ("L",),
    bin_width: float = 1.0,
    bin_origin: float = 0.0,
    chunk_size: int = 200_000,
    encoding: str = "latin1",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Aggregate DBH tallies for the provided plot selections."""

    if not selections:
        return {}, pd.DataFrame(
            columns=[
                "plot_id",
                "cluster_id",
                "visit_number",
                "plot",
                "baf",
                "measurement_date",
                "sample_type",
                "site_identifier",
                "trees",
            ]
        )

    live_status = tuple(live_status)
    usecols = ["CLSTR_ID", "VISIT_NUMBER", "PLOT", dbh_column]
    if status_column:
        usecols.append(status_column)

    selection_index = pd.DataFrame(
        [
            {
                "CLSTR_ID": sel.cluster_id,
                "VISIT_NUMBER": sel.visit_number,
                "PLOT": sel.plot,
                "plot_id": sel.plot_id,
            }
            for sel in selections
        ]
    )

    selection_index["CLSTR_ID"] = selection_index["CLSTR_ID"].astype("string")
    selection_index["VISIT_NUMBER"] = selection_index["VISIT_NUMBER"].astype("Int64")
    selection_index["PLOT"] = selection_index["PLOT"].astype("Int64")

    counters: dict[str, Counter[float]] = defaultdict(Counter)
    totals: Counter[str] = Counter()

    reader = pd.read_csv(
        tree_detail_source,
        encoding=encoding,
        usecols=usecols,
        chunksize=chunk_size,
        dtype={
            "CLSTR_ID": "string",
            "VISIT_NUMBER": "Int64",
            "PLOT": "Int64",
        },
    )

    for chunk in reader:
        chunk = chunk.dropna(subset=["CLSTR_ID", "VISIT_NUMBER", "PLOT", dbh_column])
        chunk = chunk.merge(selection_index, on=["CLSTR_ID", "VISIT_NUMBER", "PLOT"], how="inner")
        if chunk.empty:
            continue

        if status_column and status_column in chunk.columns and live_status:
            chunk = chunk[chunk[status_column].isin(live_status)]
            if chunk.empty:
                continue

        dbh_values = chunk[dbh_column].astype(float).to_numpy()
        dbh_bins = _bin_dbh(dbh_values, bin_width=bin_width, bin_origin=bin_origin)
        chunk = chunk.assign(dbh_cm=dbh_bins)

        grouped = chunk.groupby(["plot_id", "dbh_cm"]).size()
        for (plot_id, dbh_cm), tally in grouped.items():
            counters[plot_id][float(dbh_cm)] += int(tally)
            totals[plot_id] += int(tally)

    tallies = {
        plot_id: pd.DataFrame(
            sorted(counter.items(), key=lambda item: item[0]),
            columns=["dbh_cm", "tally"],
        ).reset_index(drop=True)
        for plot_id, counter in counters.items()
    }

    metadata = pd.DataFrame(
        [
            {
                "plot_id": sel.plot_id,
                "cluster_id": sel.cluster_id,
                "visit_number": sel.visit_number,
                "plot": sel.plot,
                "baf": sel.baf,
                "measurement_date": sel.measurement_date,
                "sample_type": sel.sample_type,
                "site_identifier": sel.site_identifier,
                "trees": totals.get(sel.plot_id, 0),
            }
            for sel in selections
            if sel.plot_id in tallies
        ]
    ).reset_index(drop=True)

    return tallies, metadata
