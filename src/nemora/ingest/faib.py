"""Helpers for working with BC FAIB ground sample datasets."""

from __future__ import annotations

import io
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from ftplib import FTP
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

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
    "download_faib_csvs",
    "auto_select_bafs",
    "generate_faib_manifest",
    "FAIBManifestResult",
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
    plot_info: pd.DataFrame,
    *,
    baf: float,
    dbh_col: str = "DBH_CM",
    expansion_col: str = "TREE_EXP",
    baf_col: str = "BAF",
    group_keys: tuple[str, ...] = ("CLSTR_ID", "VISIT_NUMBER", "PLOT"),
) -> pd.DataFrame:
    """Aggregate tree detail records into a stand table for a given BAF.

    Parameters
    ----------
    tree_detail:
        Raw FAIB tree detail records.
    plot_info:
        Plot-level records containing BAF metadata (sample-by-visit or plot header).
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
        if column not in tree_detail.columns or column not in plot_info.columns:
            raise KeyError(f"Missing join column '{column}' in inputs.")
    if dbh_col not in tree_detail.columns:
        raise KeyError(f"Tree detail missing DBH column '{dbh_col}'.")
    if expansion_col not in tree_detail.columns:
        raise KeyError(f"Tree detail missing expansion column '{expansion_col}'.")
    if baf_col not in plot_info.columns:
        raise KeyError(f"Plot information missing BAF column '{baf_col}'.")

    join_columns = list(group_keys) + [baf_col]
    merged = tree_detail.merge(plot_info[join_columns], on=list(group_keys), how="left")
    filtered = merged[np.isclose(merged[baf_col], baf)]
    if filtered.empty:
        return pd.DataFrame({"dbh_cm": [], "tally": []})

    dbh_numeric = pd.to_numeric(filtered[dbh_col], errors="coerce")
    weight_numeric = pd.to_numeric(filtered[expansion_col], errors="coerce")
    valid_mask = dbh_numeric.notna() & weight_numeric.notna()
    if not bool(valid_mask.any()):
        raise ValueError(
            f"No numeric records remain after cleaning columns '{dbh_col}' and '{expansion_col}'."
        )
    filtered = filtered.loc[valid_mask].copy()
    filtered = filtered.assign(
        dbh_cm=np.round(dbh_numeric.loc[valid_mask], 0),
        _weight=weight_numeric.loc[valid_mask].astype(float),
    )
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
    sample_file: str | None = "faib_sample_byvisit.csv",
    plot_file: str | None = None,
    dbh_col: str | None = None,
    expansion_col: str | None = None,
    baf_col: str | None = None,
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
    info_path = None
    if plot_file is not None:
        plot_path = root_path / plot_file
        info_path = plot_path
    elif sample_file is not None:
        info_path = root_path / sample_file

    missing = [str(tree_path)] if not tree_path.exists() else []
    if info_path is not None and not info_path.exists():
        missing.append(str(info_path))
    if missing:
        raise FileNotFoundError(f"Missing FAIB CSV file(s): {', '.join(missing)}")

    tree_detail = pd.read_csv(tree_path, low_memory=False)
    if info_path is None:
        raise ValueError("Either sample_file or plot_file must be provided.")
    plot_info = pd.read_csv(info_path, low_memory=False)

    inferred_dbh = dbh_col or ("DBH_CM" if "DBH_CM" in tree_detail.columns else "DBH")
    inferred_expansion = expansion_col or (
        "TREE_EXP" if "TREE_EXP" in tree_detail.columns else "PHF_TREE"
    )
    inferred_baf = baf_col or ("BAF" if "BAF" in plot_info.columns else "BLOWUP_MAIN")

    return aggregate_stand_table(
        tree_detail,
        plot_info,
        baf=baf,
        dbh_col=inferred_dbh,
        expansion_col=inferred_expansion,
        baf_col=inferred_baf,
    )


def auto_select_bafs(
    root: str | Path,
    count: int = 3,
    *,
    plot_file: str = "faib_plot_header.csv",
    sample_file: str = "faib_sample_byvisit.csv",
) -> list[float]:
    """Select representative BAF values from FAIB metadata.

    Parameters
    ----------
    root:
        Directory containing FAIB CSV extracts.
    count:
        Number of representative BAF values to return.
    plot_file, sample_file:
        CSV filenames to inspect for BAF metadata (plot header preferred).
    """

    root_path = Path(root)
    candidates: list[pd.Series] = []
    plot_path = root_path / plot_file
    if plot_path.exists():
        df = pd.read_csv(plot_path, usecols=["CLSTR_ID", "VISIT_NUMBER", "PLOT", "BLOWUP_MAIN"])
        candidates.append(df["BLOWUP_MAIN"])
    sample_path = root_path / sample_file
    if sample_path.exists():
        df = pd.read_csv(sample_path, usecols=["CLSTR_ID", "VISIT_NUMBER", "PLOT", "BAF"])
        candidates.append(df["BAF"])
    if not candidates:
        raise FileNotFoundError("No FAIB metadata files found for BAF detection.")

    series = pd.concat(candidates)
    series = _coerce_numeric(series, "BAF candidates")
    series = series.replace(0, np.nan).dropna().unique()
    series = np.sort(series)
    if series.size == 0:
        raise ValueError("Unable to infer BAF values from provided metadata.")

    quantiles = np.linspace(0, 1, num=count, endpoint=True)
    selected: list[float] = []
    for q in quantiles:
        idx = min(int(round(q * (series.size - 1))), series.size - 1)
        value = float(series[idx])
        if not selected or not np.isclose(selected[-1], value):
            selected.append(value)
    while len(selected) < count and series.size > 0:
        for value in series:
            if len(selected) >= count:
                break
            if not any(np.isclose(value, existing) for existing in selected):
                selected.append(float(value))
    return selected[:count]


def _list_ftp_files(directory: str) -> Iterable[str]:
    parsed = urlparse(directory)
    host = parsed.hostname or parsed.netloc
    if not host:
        raise ValueError(f"Invalid FTP directory: {directory}")
    ftp = FTP(timeout=30)  # noqa: S321 - FAIB publishes data via anonymous FTP
    ftp.encoding = "latin-1"
    ftp.connect(host, parsed.port or 21)  # noqa: S321 - public FAIB FTP host
    ftp.login()
    ftp.cwd(parsed.path)
    listing = ftp.nlst()
    ftp.quit()
    return listing


def download_faib_csvs(
    destination: str | Path,
    dataset: str = "psp",
    *,
    overwrite: bool = False,
    filenames: Iterable[str] | None = None,
) -> list[Path]:
    """Download FAIB CSV extracts via FTP into ``destination``.

    Parameters
    ----------
    destination:
        Directory where files will be written.
    dataset:
        Either ``"psp"`` or ``"non_psp"``.
    """

    if dataset not in {"psp", "non_psp"}:
        raise ValueError("dataset must be 'psp' or 'non_psp'.")

    base = f"ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/{dataset}/"
    parsed = urlparse(base)
    host = parsed.hostname or parsed.netloc
    if not host:
        raise ValueError(f"Invalid FTP base: {base}")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)

    files = _list_ftp_files(base)
    csv_files = [name for name in files if name.endswith(".csv")]
    if filenames is not None:
        requested = set(filenames)
        to_fetch = [name for name in csv_files if name in requested]
        missing = requested - set(csv_files)
        if missing:
            raise FileNotFoundError(
                f"Requested files not present in FAIB directory: {', '.join(sorted(missing))}"
            )
    else:
        to_fetch = csv_files
    downloaded: list[Path] = []

    ftp = FTP(timeout=60)  # noqa: S321 - FAIB publishes data via anonymous FTP
    ftp.encoding = "latin-1"
    ftp.connect(host, parsed.port or 21)  # noqa: S321 - public FAIB FTP host
    ftp.login()
    ftp.cwd(parsed.path)

    for name in to_fetch:
        dest = target / name
        if dest.exists() and not overwrite:
            downloaded.append(dest)
            continue
        tmp_path = dest.with_suffix(dest.suffix + ".part")
        if tmp_path.exists():
            tmp_path.unlink()
        with tmp_path.open("wb") as fh:
            ftp.retrbinary(f"RETR {name}", fh.write)
        tmp_path.replace(dest)
        downloaded.append(dest)

    ftp.quit()
    return downloaded


def _coerce_numeric(series: pd.Series, column: str) -> pd.Series:
    """Return a float series, coercing invalid entries to NaN and dropping them."""

    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.dropna()
    if numeric.empty:
        raise ValueError(f"Column '{column}' does not contain numeric values.")
    return numeric.astype(float)


@dataclass(slots=True)
class FAIBManifestResult:
    """Summary of outputs produced by :func:`generate_faib_manifest`."""

    manifest_path: Path
    tables: list[Path]
    bafs: list[float]
    truncated_flags: dict[Path, bool]
    downloaded: list[Path]


def generate_faib_manifest(
    destination: str | Path,
    *,
    dataset: str = "psp",
    source: str | Path | None = None,
    fetch: bool = False,
    overwrite: bool = False,
    bafs: Sequence[float] | None = None,
    auto_count: int | None = None,
    max_rows: int | None = None,
) -> FAIBManifestResult:
    """Fetch FAIB extracts, build stand tables, and emit a manifest.

    Parameters
    ----------
    destination:
        Directory where the manifest and stand tables will be written.
    dataset:
        FAIB dataset to process (``"psp"`` or ``"non_psp"``).
    source:
        Optional directory containing pre-downloaded FAIB CSV files. When omitted,
        files will be fetched into ``destination / "raw"`` if ``fetch`` is true.
    fetch:
        When set, download the FAIB CSV files before building stand tables.
    overwrite:
        Force re-download of CSV files even when they already exist locally.
    bafs:
        Explicit BAF values to build stand tables for.
    auto_count:
        When provided, automatically select ``auto_count`` representative BAF values
        instead of using ``bafs``.
    max_rows:
        Optional limit on the number of rows retained in each stand table.
    """

    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)

    if dataset not in {"psp", "non_psp"}:
        raise ValueError("dataset must be 'psp' or 'non_psp'.")
    if bafs is not None and auto_count is not None:
        raise ValueError("Pass either explicit `bafs` or `auto_count`, not both.")

    source_path = Path(source) if source is not None else dest_path / "raw"
    source_path.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    if fetch:
        downloaded = download_faib_csvs(source_path, dataset=dataset, overwrite=overwrite)

    plot_header = source_path / "faib_plot_header.csv"
    plot_file: str | None = "faib_plot_header.csv" if plot_header.exists() else None

    if auto_count is not None:
        baf_values = auto_select_bafs(source_path, count=auto_count)
    else:
        baf_values = list(bafs) if bafs is not None else [4.0, 8.0, 12.0]

    records: list[dict[str, object]] = []
    tables: list[Path] = []
    truncated_flags: dict[Path, bool] = {}

    for baf in baf_values:
        table = build_stand_table_from_csvs(source_path, baf=baf, plot_file=plot_file)
        truncated = False
        if max_rows is not None and len(table) > max_rows:
            table = table.sort_values("dbh_cm").head(max_rows)
            truncated = True

        slug = _format_baf_slug(baf)
        table_path = dest_path / f"stand_table_baf{slug}.csv"
        table.to_csv(table_path, index=False)
        tables.append(table_path)
        truncated_flags[table_path] = truncated
        records.append(
            {
                "dataset": dataset,
                "baf": baf,
                "rows": len(table),
                "path": str(table_path.relative_to(dest_path)),
                "truncated": truncated,
            }
        )

    manifest = pd.DataFrame(records)
    manifest_path = dest_path / "faib_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return FAIBManifestResult(
        manifest_path=manifest_path,
        tables=tables,
        bafs=[float(value) for value in baf_values],
        truncated_flags=truncated_flags,
        downloaded=downloaded,
    )


def _format_baf_slug(baf: float) -> str:
    """Return a filesystem-safe representation of a BAF value."""

    text = f"{baf:.6f}".rstrip("0").rstrip(".")
    text = text.replace(".", "_").replace("-", "neg")
    return text or "0"
