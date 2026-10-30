"""Select public FAIB plots for a research-forest validation target.

The selection is deliberately explicit: a target has a coordinate, radius, BEC
codes, projected-age floor, and optional sample-type/first-visit filters. The
result keeps plot identity, provenance-relevant metadata, and distance so a
selection manifest can be regenerated from public inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "ForestPlotTarget",
    "RESEARCH_FOREST_TARGETS",
    "assign_holdout_split",
    "resolve_tree_detail",
    "select_forest_plots",
    "selection_manifest",
    "write_selection_manifest",
]

_SITE_COLUMNS = ("SITE_IDENTIFIER", "Latitude", "Longitude", "BEC_ZONE", "BEC_SBZ", "BEC_VAR")
_PLOT_COLUMNS = ("SITE_IDENTIFIER", "CLSTR_ID", "VISIT_NUMBER", "PLOT")
_SAMPLE_COLUMNS = (
    "SITE_IDENTIFIER",
    "CLSTR_ID",
    "VISIT_NUMBER",
    "FIRST_MSMT",
    "MEAS_DT",
    "PROJ_AGE_1",
    "SAMP_TYP",
)
_TREE_REQUIRED = ("CLSTR_ID", "VISIT_NUMBER", "PLOT", "TREE_NO", "SPECIES", "DBH", "LV_D")
_TREE_OPTIONAL = ("HEIGHT", "PHF_TREE")
_TREE_OUTPUT = (
    "plot_id",
    "tree_id",
    "species_code",
    "dbh_cm",
    "height_m",
    "tally",
    "status",
)


@dataclass(frozen=True, slots=True)
class ForestPlotTarget:
    """Describe a forest validation target and its plot filters.

    Parameters
    ----------
    name:
        Human-readable target name.
    latitude, longitude:
        Target centre in WGS84 decimal degrees.
    radius_km:
        Great-circle search radius.
    bec_codes:
        Accepted BEC codes, such as ``("SBSdw1", "ICHmk3")``.
    min_projected_age:
        Minimum ``PROJ_AGE_1`` in years.
    sample_type:
        Optional FAIB ``SAMP_TYP`` value to keep.
    first_visit_only:
        Keep only visits marked ``FIRST_MSMT == "Y"``.
    """

    name: str
    latitude: float
    longitude: float
    radius_km: float
    bec_codes: tuple[str, ...]
    min_projected_age: float = 60.0
    max_projected_age: float | None = None
    sample_type: str | None = "F"
    first_visit_only: bool = True
    min_distance_km: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must not be blank")
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be in [-90, 90]")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be in [-180, 180]")
        if self.radius_km <= 0.0 or not math.isfinite(self.radius_km):
            raise ValueError("radius_km must be finite and positive")
        if not self.bec_codes:
            raise ValueError("bec_codes must contain at least one code")
        if self.min_projected_age < 0.0 or not math.isfinite(self.min_projected_age):
            raise ValueError("min_projected_age must be finite and non-negative")
        if self.min_distance_km < 0.0 or not math.isfinite(self.min_distance_km):
            raise ValueError("min_distance_km must be finite and non-negative")
        if self.max_projected_age is not None:
            if self.max_projected_age < self.min_projected_age or not math.isfinite(
                self.max_projected_age
            ):
                raise ValueError("max_projected_age must be finite and >= min_projected_age")


RESEARCH_FOREST_TARGETS: dict[str, ForestPlotTarget] = {
    "gavin": ForestPlotTarget(
        name="Alex Fraser Research Forest, Gavin Lake",
        latitude=52.462616,
        longitude=-121.763919,
        radius_km=25.0,
        bec_codes=("ICHmk3", "SBSdw1"),
    ),
    "mkrf": ForestPlotTarget(
        name="Malcolm Knapp Research Forest",
        latitude=49.30663,
        longitude=-122.551375,
        radius_km=50.0,
        bec_codes=("CWHdm",),
    ),
    "mkrf_regional": ForestPlotTarget(
        name="Malcolm Knapp regional CWHdm pool",
        latitude=49.30663,
        longitude=-122.551375,
        radius_km=100.0,
        bec_codes=("CWHdm",),
        min_distance_km=50.0,
    ),
    "mkrf_age80_110": ForestPlotTarget(
        name="Malcolm Knapp regional CWHdm pool, 80-110 years",
        latitude=49.30663,
        longitude=-122.551375,
        radius_km=100.0,
        bec_codes=("CWHdm",),
        min_projected_age=80.0,
        max_projected_age=110.0,
        min_distance_km=50.0,
    ),
}


def select_forest_plots(
    site_header: pd.DataFrame,
    plot_header: pd.DataFrame,
    sample_by_visit: pd.DataFrame,
    *,
    target: ForestPlotTarget,
) -> pd.DataFrame:
    """Select FAIB plots for one target.

    Parameters
    ----------
    site_header:
        FAIB site-level table with coordinates and BEC fields.
    plot_header:
        FAIB plot-level table with cluster, visit, and plot numbers.
    sample_by_visit:
        FAIB visit-level table with first-measurement, date, age, and sample type.
    target:
        Target and filter definition.

    Returns
    -------
    pandas.DataFrame
        One row per selected plot with canonical identity, distance, BEC, age,
        and provenance-relevant metadata. The frame is sorted by distance then
        plot id.

    Raises
    ------
    KeyError
        If a required source column is missing.
    ValueError
        If no plots match.
    """
    _require_columns(site_header, _SITE_COLUMNS, "site_header")
    _require_columns(plot_header, _PLOT_COLUMNS, "plot_header")
    _require_columns(sample_by_visit, _SAMPLE_COLUMNS, "sample_by_visit")

    plots = plot_header.copy()
    plots["plot_id"] = (
        plots["CLSTR_ID"].astype(str)
        + "_v"
        + plots["VISIT_NUMBER"].astype("int64").astype(str)
        + "_p"
        + plots["PLOT"].astype("int64").astype(str)
    )
    visits = sample_by_visit.copy()
    visits["MEAS_DT"] = pd.to_datetime(visits["MEAS_DT"], errors="coerce")
    merged = plots.merge(
        visits[
            [
                "SITE_IDENTIFIER",
                "CLSTR_ID",
                "VISIT_NUMBER",
                "FIRST_MSMT",
                "MEAS_DT",
                "PROJ_AGE_1",
                "SAMP_TYP",
            ]
        ],
        on=["SITE_IDENTIFIER", "CLSTR_ID", "VISIT_NUMBER"],
        how="inner",
    )
    sites = site_header.copy()
    sites["bec"] = sites.apply(_bec_code, axis=1)
    merged = merged.merge(
        sites[["SITE_IDENTIFIER", "Latitude", "Longitude", "bec"]],
        on="SITE_IDENTIFIER",
        how="left",
    )
    merged["distance_km"] = _haversine_km(
        target.latitude,
        target.longitude,
        merged["Latitude"].to_numpy(dtype=float),
        merged["Longitude"].to_numpy(dtype=float),
    )

    mask = (
        merged["bec"].isin(target.bec_codes)
        & (merged["PROJ_AGE_1"] >= target.min_projected_age)
        & (merged["distance_km"] <= target.radius_km)
        & (merged["distance_km"] >= target.min_distance_km)
    )
    if target.max_projected_age is not None:
        mask &= merged["PROJ_AGE_1"] <= target.max_projected_age
    if target.first_visit_only:
        mask &= merged["FIRST_MSMT"].eq("Y")
    if target.sample_type is not None:
        mask &= merged["SAMP_TYP"].eq(target.sample_type)
    selected = merged.loc[mask].copy()
    if selected.empty:
        raise ValueError(f"no plots matched target {target.name!r}")

    result = pd.DataFrame(
        {
            "plot_id": selected["plot_id"],
            "cluster_id": selected["CLSTR_ID"],
            "visit_number": selected["VISIT_NUMBER"],
            "plot_number": selected["PLOT"],
            "site_identifier": selected["SITE_IDENTIFIER"],
            "measurement_date": selected["MEAS_DT"].dt.date,
            "sample_type": selected["SAMP_TYP"],
            "latitude": selected["Latitude"],
            "longitude": selected["Longitude"],
            "bec": selected["bec"],
            "projected_age": selected["PROJ_AGE_1"],
            "distance_km": selected["distance_km"],
        }
    )
    return result.sort_values(["distance_km", "plot_id"]).reset_index(drop=True)


def resolve_tree_detail(
    tree_detail_source: str | Path | pd.DataFrame,
    selected_plots: pd.DataFrame,
    *,
    live_only: bool = True,
    chunk_size: int = 200_000,
) -> pd.DataFrame:
    """Resolve FAIB tree detail for selected plots into canonical tree rows.

    Parameters
    ----------
    tree_detail_source:
        Path to ``faib_tree_detail.csv`` or an equivalent DataFrame.
    selected_plots:
        Output of :func:`select_forest_plots` (or a frame with the canonical
        ``plot_id``, ``cluster_id``, ``visit_number``, and ``plot_number``
        columns).
    live_only:
        Keep only ``LV_D == "L"`` records. When false, dead records are kept with
        ``status="dead"``.
    chunk_size:
        CSV chunk size for streaming reads.

    Returns
    -------
    pandas.DataFrame
        Canonical ``nemora.inventory.v1`` tree columns only.

    Raises
    ------
    KeyError
        If required selection or tree columns are missing.
    ValueError
        If ``chunk_size`` is not positive.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    selection_columns = ("plot_id", "cluster_id", "visit_number", "plot_number")
    _require_columns(selected_plots, selection_columns, "selected_plots")
    selection = selected_plots.loc[:, list(selection_columns)].copy()
    selection["cluster_id"] = selection["cluster_id"].astype(str)
    selection["visit_number"] = pd.to_numeric(selection["visit_number"], errors="coerce").astype(
        "Int64"
    )
    selection["plot_number"] = pd.to_numeric(selection["plot_number"], errors="coerce").astype(
        "Int64"
    )

    available = _source_columns(tree_detail_source)
    missing = [column for column in _TREE_REQUIRED if column not in available]
    if missing:
        raise KeyError("tree_detail_source is missing required columns: " + ", ".join(missing))
    usecols = [column for column in (*_TREE_REQUIRED, *_TREE_OPTIONAL) if column in available]
    frames: list[pd.DataFrame] = []
    for chunk in _tree_chunks(tree_detail_source, usecols=usecols, chunk_size=chunk_size):
        frame = chunk.copy()
        frame["CLSTR_ID"] = frame["CLSTR_ID"].astype(str)
        frame["VISIT_NUMBER"] = pd.to_numeric(frame["VISIT_NUMBER"], errors="coerce").astype(
            "Int64"
        )
        frame["PLOT"] = pd.to_numeric(frame["PLOT"], errors="coerce").astype("Int64")
        frame = frame.merge(
            selection.rename(
                columns={
                    "cluster_id": "CLSTR_ID",
                    "visit_number": "VISIT_NUMBER",
                    "plot_number": "PLOT",
                }
            ),
            on=["CLSTR_ID", "VISIT_NUMBER", "PLOT"],
            how="inner",
        )
        if frame.empty:
            continue
        if live_only:
            frame = frame.loc[frame["LV_D"].eq("L")]
        else:
            frame = frame.loc[frame["LV_D"].notna()]
        frame = frame.assign(
            dbh_cm=pd.to_numeric(frame["DBH"], errors="coerce"),
            height_m=(
                pd.to_numeric(frame["HEIGHT"], errors="coerce")
                if "HEIGHT" in frame.columns
                else np.nan
            ),
            tally=(
                pd.to_numeric(frame["PHF_TREE"], errors="coerce")
                if "PHF_TREE" in frame.columns
                else 1.0
            ),
        )
        frame = frame.loc[frame["dbh_cm"].notna() & (frame["dbh_cm"] > 0.0)]
        if frame.empty:
            continue
        frame["tally"] = frame["tally"].fillna(1.0)
        frame["species_code"] = frame["SPECIES"].astype("string")
        frame["status"] = np.where(frame["LV_D"].eq("L"), "live", "dead")
        frame["tree_id"] = _make_tree_ids(frame)
        frames.append(frame.loc[:, list(_TREE_OUTPUT)])

    if not frames:
        return pd.DataFrame(columns=list(_TREE_OUTPUT))
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["plot_id", "tree_id"])
        .reset_index(drop=True)
    )


def assign_holdout_split(
    plots: pd.DataFrame,
    *,
    holdout_fraction: float = 0.25,
    seed: int = 20261030,
) -> pd.DataFrame:
    """Assign a deterministic train/hold-out split.

    The split uses a SHA-256 key over ``seed`` and ``plot_id``. It is stable
    across machines and Python versions and does not depend on row order.

    Parameters
    ----------
    plots:
        Selected plot frame with a ``plot_id`` column.
    holdout_fraction:
        Fraction of plots assigned to the hold-out set.
    seed:
        Integer split seed.

    Returns
    -------
    pandas.DataFrame
        A copy with a ``split`` column containing ``train`` or ``holdout``.

    Raises
    ------
    ValueError
        If there are fewer than two plots or the fraction is outside ``(0, 1)``.
    """
    if "plot_id" not in plots.columns:
        raise KeyError("plots is missing required column 'plot_id'")
    if len(plots) < 2:
        raise ValueError("at least two plots are needed for a train/hold-out split")
    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError("holdout_fraction must be between 0 and 1")

    result = plots.copy()
    keys = [
        hashlib.sha256(f"{seed}:{plot_id}".encode()).hexdigest()
        for plot_id in result["plot_id"].astype(str)
    ]
    order = np.argsort(np.asarray(keys, dtype=object), kind="stable")
    holdout_count = max(1, min(len(result) - 1, round(len(result) * holdout_fraction)))
    holdout_index = set(order[:holdout_count].tolist())
    result["split"] = [
        "holdout" if index in holdout_index else "train" for index in range(len(result))
    ]
    return result


def selection_manifest(
    selected: pd.DataFrame,
    *,
    target: ForestPlotTarget,
    holdout_fraction: float,
    split_seed: int,
) -> dict[str, Any]:
    """Build a location-light manifest for a selected plot set."""
    required = {"plot_id", "bec", "projected_age", "distance_km", "split"}
    missing = sorted(required - set(selected.columns))
    if missing:
        raise KeyError("selected is missing columns: " + ", ".join(missing))
    plot_ids = selected["plot_id"].astype(str).sort_values().tolist()
    id_hash = hashlib.sha256("\n".join(plot_ids).encode("utf-8")).hexdigest()
    ages = selected["projected_age"].astype(float)
    distances = selected["distance_km"].astype(float)
    return {
        "target": {
            "name": target.name,
            "latitude": target.latitude,
            "longitude": target.longitude,
            "radius_km": target.radius_km,
            "bec_codes": list(target.bec_codes),
            "min_projected_age": target.min_projected_age,
            "max_projected_age": target.max_projected_age,
            "sample_type": target.sample_type,
            "first_visit_only": target.first_visit_only,
            "min_distance_km": target.min_distance_km,
        },
        "filters": {
            "holdout_fraction": holdout_fraction,
            "split_seed": split_seed,
            "age_column": "PROJ_AGE_1",
        },
        "counts": {
            "total": int(len(selected)),
            "train": int((selected["split"] == "train").sum()),
            "holdout": int((selected["split"] == "holdout").sum()),
        },
        "summary": {
            "bec_counts": {
                str(key): int(value)
                for key, value in selected["bec"].value_counts().sort_index().items()
            },
            "projected_age_min": float(ages.min()),
            "projected_age_median": float(ages.median()),
            "projected_age_max": float(ages.max()),
            "distance_km_min": float(distances.min()),
            "distance_km_max": float(distances.max()),
            "plot_ids_sha256": id_hash,
        },
    }


def write_selection_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    """Write a selection manifest as stable, indented JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_columns(source: str | Path | pd.DataFrame) -> set[str]:
    if isinstance(source, pd.DataFrame):
        return set(source.columns)
    return set(pd.read_csv(Path(source), nrows=0).columns)


def _tree_chunks(
    source: str | Path | pd.DataFrame,
    *,
    usecols: list[str],
    chunk_size: int,
) -> Iterator[pd.DataFrame]:
    if isinstance(source, pd.DataFrame):
        yield source.loc[:, usecols]
        return
    yield from pd.read_csv(
        Path(source),
        usecols=usecols,
        chunksize=chunk_size,
        low_memory=False,
    )


def _make_tree_ids(frame: pd.DataFrame) -> pd.Series:
    plot_ids = frame["plot_id"].astype(str).tolist()
    tree_numbers = pd.to_numeric(frame["TREE_NO"], errors="coerce").tolist()
    counts: dict[str, int] = {}
    ids: list[str] = []
    for plot_id, tree_number in zip(plot_ids, tree_numbers, strict=True):
        base = f"{plot_id}-T00000" if pd.isna(tree_number) else f"{plot_id}-T{int(tree_number):05d}"
        count = counts.get(base, 0)
        counts[base] = count + 1
        ids.append(base if count == 0 else f"{base}-{count}")
    return pd.Series(ids, index=frame.index, dtype="string")


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {', '.join(missing)}")


def _bec_code(row: pd.Series) -> str:
    zone = str(row["BEC_ZONE"]).strip() if pd.notna(row["BEC_ZONE"]) else ""
    subzone = str(row["BEC_SBZ"]).strip() if pd.notna(row["BEC_SBZ"]) else ""
    variant = row["BEC_VAR"]
    if pd.isna(variant):
        variant_text = ""
    else:
        variant_value = float(variant)
        variant_text = (
            str(int(variant_value))
            if variant_value.is_integer()
            else str(variant_value).rstrip("0").rstrip(".")
        )
    return zone + subzone + variant_text


def _haversine_km(
    latitude: float,
    longitude: float,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    radius_km = 6371.0088
    lat1 = math.radians(latitude)
    lon1 = math.radians(longitude)
    lat2 = np.radians(latitudes)
    lon2 = np.radians(longitudes)
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    haversine = (
        np.sin(delta_lat / 2.0) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
