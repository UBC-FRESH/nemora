"""Helpers for working with BC FAIB ground sample datasets."""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PSP_DICTIONARY_URL = (
    "ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/psp/"
    "PSP_data_dictionary_20250514.xlsx"
)
NON_PSP_DICTIONARY_URL = (
    "ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/non_psp/"
    "non_PSP_data_dictionary_20250514.xlsx"
)

__all__ = [
    "DataDictionary",
    "load_data_dictionary",
    "load_psp_dictionary",
    "load_non_psp_dictionary",
    "aggregate_stand_table",
    "build_stand_table_from_csvs",
    "PSP_DICTIONARY_URL",
    "NON_PSP_DICTIONARY_URL",
]


@dataclass(slots=True)
class DataDictionary:
    """Structured representation of FAIB data dictionary entries."""

    sheets: Mapping[str, pd.DataFrame]

    def get_table_schema(self, table: str) -> pd.DataFrame:
        """Return the schema for a specific table."""
        key = table.lower()
        for name, frame in self.sheets.items():
            if name.lower() == key:
                return frame
        raise KeyError(f"Unknown FAIB table '{table}'. Available: {list(self.sheets)}")


def load_data_dictionary(url: str) -> DataDictionary:
    """Download and parse a FAIB data dictionary XLSX file."""
    if url.startswith("ftp://"):
        from urllib.request import urlopen

        with urlopen(url) as fh:  # noqa: S310 - FAIB publishes public datasets here
            buffer = io.BytesIO(fh.read())
        sheets = pd.read_excel(buffer, sheet_name=None, engine="openpyxl")
    else:
        path = Path(url)
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    normalized: dict[str, pd.DataFrame] = {}
    for name, frame in sheets.items():
        frame = frame.copy()
        if "Attribute" in frame.columns:
            frame = frame.dropna(subset=["Attribute"])
        normalized[name] = frame
    return DataDictionary(normalized)


def load_psp_dictionary() -> DataDictionary:
    """Convenience wrapper for the PSP data dictionary."""

    return load_data_dictionary(PSP_DICTIONARY_URL)


def load_non_psp_dictionary() -> DataDictionary:
    """Convenience wrapper for the non-PSP data dictionary."""

    return load_data_dictionary(NON_PSP_DICTIONARY_URL)


def aggregate_stand_table(
    tree_detail: pd.DataFrame,
    sample_byvisit: pd.DataFrame,
    *,
    baf: float,
    dbh_col: str = "DBH_CM",
    expansion_col: str = "TREE_EXP",
    group_keys: tuple[str, ...] = ("CLSTR_ID", "VISIT_NUMBER", "PLOT"),
) -> pd.DataFrame:
    """Aggregate tree detail records into a stand table for a given BAF.

    Parameters
    ----------
    tree_detail:
        Raw FAIB tree detail records.
    sample_byvisit:
        Sample-by-visit records with BAF metadata.
    baf:
        Target basal area factor to filter (e.g., 12).
    dbh_col:
        Column containing diameter at breast height in centimetres.
    expansion_col:
        Column representing tree expansion weights.
    group_keys:
        Keys used to join tree detail with sample-by-visit metadata.
    """

    for column in group_keys:
        if column not in tree_detail.columns or column not in sample_byvisit.columns:
            raise KeyError(f"Missing join column '{column}' in inputs.")
    if dbh_col not in tree_detail.columns:
        raise KeyError(f"Tree detail missing DBH column '{dbh_col}'.")
    if expansion_col not in tree_detail.columns:
        raise KeyError(f"Tree detail missing expansion column '{expansion_col}'.")
    if "BAF" not in sample_byvisit.columns:
        raise KeyError("sample_byvisit must contain a 'BAF' column.")

    join_columns = list(group_keys) + ["BAF"]
    merged = tree_detail.merge(sample_byvisit[join_columns], on=list(group_keys), how="left")
    filtered = merged[np.isclose(merged["BAF"], baf)]
    if filtered.empty:
        return pd.DataFrame({"dbh_cm": [], "tally": []})

    dbh_cm = filtered[dbh_col].astype(float)
    weights = filtered[expansion_col].astype(float)
    filtered = filtered.assign(dbh_cm=np.round(dbh_cm, 0), _weight=weights)
    aggregated = (
        filtered.groupby("dbh_cm", as_index=False)["_weight"]
        .sum()
        .rename(columns={"_weight": "tally"})
        .sort_values("dbh_cm")
    )
    return aggregated.reset_index(drop=True)


def build_stand_table_from_csvs(
    root: str | Path,
    baf: float,
    *,
    tree_file: str = "faib_tree_detail.csv",
    sample_file: str = "faib_sample_byvisit.csv",
) -> pd.DataFrame:
    """Load FAIB CSV extracts from ``root`` and build a stand table for ``baf``.

    Parameters
    ----------
    root:
        Directory containing the FAIB CSV extracts.
    baf:
        Desired basal area factor to filter.
    tree_file:
        Filename for the tree detail CSV within ``root``.
    sample_file:
        Filename for the sample-by-visit CSV within ``root``.
    """

    root_path = Path(root)
    tree_path = root_path / tree_file
    sample_path = root_path / sample_file
    if not tree_path.exists() or not sample_path.exists():
        missing = [str(path) for path in (tree_path, sample_path) if not path.exists()]
        raise FileNotFoundError(f"Missing FAIB CSV file(s): {', '.join(missing)}")

    tree_detail = pd.read_csv(tree_path)
    sample_byvisit = pd.read_csv(sample_path)
    return aggregate_stand_table(tree_detail, sample_byvisit, baf=baf)
