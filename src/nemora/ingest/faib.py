"""Helpers for working with BC FAIB ground sample datasets."""

from __future__ import annotations

import io
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PSP_DICTIONARY_URL = (
    "ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/psp/"
    "PSP_data_dictionary_20250514.xlsx"
)
NON_PSP_DICTIONARY_URL = (
    "ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/non_psp/"
    "non_PSP_data_dictionary_20250514.xlsx"
)


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
