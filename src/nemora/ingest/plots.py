"""Resolve mapped inventory tables into canonical Nemora releases.

This module is the generic part of the ingest boundary. Source-specific adapters
still own downloading, unit conversion, and source validation; this layer owns
the common mapping into :mod:`nemora.ingest.schema`, optional tree-count
derivation, and opt-in materialization of tally classes into individual trees.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, SupportsFloat

import pandas as pd

from .schema import (
    PLOT_COLUMNS,
    SPECIES_COLUMNS,
    TREE_COLUMNS,
    InventoryProvenance,
    InventoryRelease,
    InventoryUnits,
    InventoryValidationError,
    empty_species_table,
    validate_inventory_release,
)

__all__ = [
    "PlotInventorySpec",
    "build_inventory_release",
    "materialize_tally_trees",
]


@dataclass(frozen=True, slots=True)
class PlotInventorySpec:
    """Describe how source tables map to the canonical inventory schema.

    Column mappings use canonical names as keys and source-column names as
    values. A canonical name that already exists in the source frame is used
    without a mapping entry. Missing optional columns are filled from the
    corresponding defaults or with ``NA``.

    Parameters
    ----------
    release_id:
        Stable identifier written to the release manifest.
    provenance:
        Source and retrieval metadata required by ``nemora.inventory.v1``.
    plot_columns:
        Mapping from canonical plot columns to source column names.
    tree_columns:
        Mapping from canonical tree columns to source column names.
    species_columns:
        Mapping from canonical species columns to source column names.
    plot_defaults:
        Values used when a canonical plot column is absent.
    tree_defaults:
        Values used when a canonical tree column is absent.
    species_defaults:
        Values used when a canonical species column is absent.
    units:
        Canonical release units.
    crs:
        CRS for plot coordinates, when present.
    metadata:
        Additional JSON-serializable release metadata.
    derive_tree_count:
        Fill a missing ``plots.tree_count`` with the number of tree rows per
        plot. This is a row count, not a sum of expansion tallies.
    """

    release_id: str
    provenance: InventoryProvenance
    plot_columns: Mapping[str, str] = field(default_factory=dict)
    tree_columns: Mapping[str, str] = field(default_factory=dict)
    species_columns: Mapping[str, str] = field(default_factory=dict)
    plot_defaults: Mapping[str, object] = field(default_factory=dict)
    tree_defaults: Mapping[str, object] = field(default_factory=dict)
    species_defaults: Mapping[str, object] = field(default_factory=dict)
    units: InventoryUnits = field(default_factory=InventoryUnits)
    crs: str | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
    derive_tree_count: bool = True


def build_inventory_release(
    plots: pd.DataFrame,
    trees: pd.DataFrame,
    *,
    spec: PlotInventorySpec,
    species: pd.DataFrame | None = None,
) -> InventoryRelease:
    """Resolve source tables into a validated ``nemora.inventory.v1`` release.

    Parameters
    ----------
    plots:
        Source plot table. It must provide, or map to, ``plot_id``.
    trees:
        Source tree or tally-class table. It must provide, or map to,
        ``plot_id``, ``dbh_cm``, and ``tally``.
    spec:
        Release metadata, column mappings, and defaults.
    species:
        Optional species lookup table.

    Returns
    -------
    InventoryRelease
        A validated canonical release.

    Raises
    ------
    InventoryValidationError
        If a mapping names an unknown source column or the resolved release is
        invalid.
    """
    if not isinstance(spec, PlotInventorySpec):
        raise TypeError("spec must be a PlotInventorySpec")
    if not isinstance(plots, pd.DataFrame) or not isinstance(trees, pd.DataFrame):
        raise TypeError("plots and trees must be pandas DataFrames")
    if species is not None and not isinstance(species, pd.DataFrame):
        raise TypeError("species must be a pandas DataFrame or None")

    plot_table = _normalise_table(
        plots,
        PLOT_COLUMNS,
        spec.plot_columns,
        spec.plot_defaults,
        table="plots",
    )
    tree_table = _normalise_table(
        trees,
        TREE_COLUMNS,
        spec.tree_columns,
        spec.tree_defaults,
        table="trees",
    )
    species_table = (
        _normalise_table(
            species,
            SPECIES_COLUMNS,
            spec.species_columns,
            spec.species_defaults,
            table="species",
        )
        if species is not None
        else empty_species_table()
    )

    tree_count_supplied = (
        "tree_count" in spec.plot_columns
        or "tree_count" in plots.columns
        or "tree_count" in spec.plot_defaults
    )
    if spec.derive_tree_count and not tree_count_supplied:
        counts = tree_table.groupby("plot_id", dropna=True).size()
        plot_table["tree_count"] = plot_table["plot_id"].map(counts).fillna(0).astype("int64")

    metadata: dict[str, Any] = dict(spec.metadata)
    metadata.setdefault(
        "column_mapping",
        {
            "plots": dict(spec.plot_columns),
            "trees": dict(spec.tree_columns),
            "species": dict(spec.species_columns),
        },
    )

    release = InventoryRelease(
        release_id=spec.release_id,
        provenance=spec.provenance,
        plots=plot_table,
        trees=tree_table,
        species=species_table,
        units=spec.units,
        crs=spec.crs,
        metadata=metadata,
    )
    validate_inventory_release(release)
    return release


def materialize_tally_trees(
    release: InventoryRelease,
    *,
    tree_id_prefix: str = "T",
) -> InventoryRelease:
    """Expand tally classes into individual tree rows with stable IDs.

    Rows that already have a ``tree_id`` are preserved. Rows without one are
    expanded into ``tally`` copies with ``tally=1`` and deterministic
    ``<prefix><plot_id>-<sequence>`` identifiers. Fractional tallies are
    rejected because materializing them requires an explicit expansion rule.

    Parameters
    ----------
    release:
        A validated ``nemora.inventory.v1`` release.
    tree_id_prefix:
        Prefix for generated tree identifiers.

    Returns
    -------
    InventoryRelease
        A validated release at the individual-tree grain.

    Raises
    ------
    InventoryValidationError
        If the input release is invalid or a tally is not a non-negative
        integer.
    """
    validate_inventory_release(release)
    trees = release.trees.copy()
    if trees.empty:
        return _copy_release_with_trees(release, trees, materialized=0)

    existing_ids = {
        str(value)
        for value in trees.loc[trees["tree_id"].notna(), "tree_id"].tolist()
        if str(value).strip()
    }
    counters: dict[str, int] = defaultdict(int)
    records: list[dict[str, object]] = []

    for _, row in trees.iterrows():
        tree_id = row.get("tree_id")
        if not pd.isna(tree_id) and str(tree_id).strip():
            records.append(dict(row))
            continue

        tally_value = _as_non_negative_integer(row.get("tally"), row.get("plot_id"))
        plot_id = str(row["plot_id"])
        for _ in range(tally_value):
            counters[plot_id] += 1
            candidate = f"{tree_id_prefix}{plot_id}-{counters[plot_id]:04d}"
            while candidate in existing_ids:
                counters[plot_id] += 1
                candidate = f"{tree_id_prefix}{plot_id}-{counters[plot_id]:04d}"
            existing_ids.add(candidate)
            record = dict(row)
            record["tree_id"] = candidate
            record["tally"] = 1
            records.append(record)

    materialized = pd.DataFrame(records, columns=list(TREE_COLUMNS))
    return _copy_release_with_trees(release, materialized, materialized=len(materialized))


def _normalise_table(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    mapping: Mapping[str, str],
    defaults: Mapping[str, object],
    *,
    table: str,
) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)
    for column in columns:
        if column in mapping:
            source = mapping[column]
            if source not in frame.columns:
                raise InventoryValidationError(
                    [f"{table} mapping for {column!r} names missing source column {source!r}"]
                )
            result[column] = frame[source]
        elif column in frame.columns:
            result[column] = frame[column]
        elif column in defaults:
            result[column] = pd.Series(
                [defaults[column]] * len(frame),
                index=frame.index,
            )
        else:
            result[column] = pd.NA
    return result


def _as_non_negative_integer(value: object, plot_id: object) -> int:
    if not isinstance(value, str) and not isinstance(value, SupportsFloat):
        raise InventoryValidationError([f"tally for plot {plot_id!r} is not numeric: {value!r}"])
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise InventoryValidationError(
            [f"tally for plot {plot_id!r} is not numeric: {value!r}"]
        ) from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise InventoryValidationError(
            [f"tally for plot {plot_id!r} must be finite and non-negative: {value!r}"]
        )
    rounded = int(round(numeric))
    if not math.isclose(numeric, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise InventoryValidationError(
            [
                f"tally for plot {plot_id!r} is fractional ({numeric!r}); "
                "an explicit expansion rule is required before materialization"
            ]
        )
    return rounded


def _copy_release_with_trees(
    release: InventoryRelease,
    trees: pd.DataFrame,
    *,
    materialized: int,
) -> InventoryRelease:
    metadata: dict[str, Any] = dict(release.metadata)
    metadata["tally_grain"] = "individual_tree"
    metadata["materialized_tree_count"] = materialized
    result = InventoryRelease(
        release_id=release.release_id,
        provenance=release.provenance,
        plots=release.plots.copy(),
        trees=trees,
        species=release.species.copy(),
        units=release.units,
        crs=release.crs,
        metadata=metadata,
    )
    validate_inventory_release(result)
    return result
