"""Shared tidy inventory schema for Nemora ingest adapters.

The schema describes one release as three tables plus release-level units and
provenance:

- ``plots``: one row per plot visit;
- ``trees``: one row per tree or per DBH tally class; and
- ``species``: an optional species-code lookup.

The canonical units are fixed so downstream fitting and sampling do not have to
guess. Adapters convert source units before constructing a release.

The schema is intentionally additive: an adapter may keep its historical output
bytes while also emitting a validated ``nemora.inventory.v1`` release.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = "nemora.inventory.v1"

PLOT_COLUMNS: tuple[str, ...] = (
    "plot_id",
    "cluster_id",
    "visit_number",
    "plot_number",
    "measurement_date",
    "sample_type",
    "site_identifier",
    "baf",
    "tree_count",
    "latitude",
    "longitude",
)
TREE_COLUMNS: tuple[str, ...] = (
    "plot_id",
    "tree_id",
    "species_code",
    "dbh_cm",
    "height_m",
    "tally",
    "status",
)
SPECIES_COLUMNS: tuple[str, ...] = (
    "species_code",
    "common_name",
    "scientific_name",
)

PLOT_REQUIRED: tuple[str, ...] = ("plot_id",)
TREE_REQUIRED: tuple[str, ...] = ("plot_id", "dbh_cm", "tally")
SPECIES_REQUIRED: tuple[str, ...] = ("species_code",)

ALLOWED_UNITS: dict[str, frozenset[str]] = {
    "dbh": frozenset({"cm"}),
    "height": frozenset({"m"}),
    "tally": frozenset({"count"}),
    "baf": frozenset({"m2_per_ha"}),
}

_TABLE_FILES: dict[str, str] = {
    "plots": "plots.csv",
    "trees": "trees.csv",
    "species": "species.csv",
}


class InventoryValidationError(ValueError):
    """Raised when an inventory release does not match the shared schema."""

    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        detail = "\n".join(f"- {error}" for error in self.errors)
        super().__init__(f"Inventory release is invalid:\n{detail}")


@dataclass(frozen=True, slots=True)
class InventoryUnits:
    """Canonical units for the inventory tables."""

    dbh: str = "cm"
    height: str = "m"
    tally: str = "count"
    baf: str = "m2_per_ha"


@dataclass(frozen=True, slots=True)
class InventoryProvenance:
    """Release-level source and retrieval metadata."""

    source_id: str
    source_version: str
    retrieved_utc: str
    source_uri: str | None = None
    licence: str | None = None
    checksum: str | None = None
    notes: str | None = None


@dataclass(slots=True)
class InventoryRelease:
    """A validated, tidy inventory release."""

    release_id: str
    provenance: InventoryProvenance
    plots: pd.DataFrame
    trees: pd.DataFrame
    species: pd.DataFrame = field(default_factory=lambda: empty_species_table())
    units: InventoryUnits = field(default_factory=InventoryUnits)
    crs: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def schema_version(self) -> str:
        """Return the schema version written by this release."""
        return SCHEMA_VERSION

    def to_manifest(self) -> dict[str, Any]:
        """Return the JSON-serializable release manifest."""
        return {
            "schema_version": self.schema_version,
            "release_id": self.release_id,
            "crs": self.crs,
            "units": asdict(self.units),
            "provenance": asdict(self.provenance),
            "metadata": dict(self.metadata),
            "tables": dict(_TABLE_FILES),
            "table_columns": {
                "plots": list(PLOT_COLUMNS),
                "trees": list(TREE_COLUMNS),
                "species": list(SPECIES_COLUMNS),
            },
        }


def empty_plots_table() -> pd.DataFrame:
    """Return an empty plots table with the canonical columns."""
    return pd.DataFrame(columns=list(PLOT_COLUMNS))


def empty_trees_table() -> pd.DataFrame:
    """Return an empty trees table with the canonical columns."""
    return pd.DataFrame(columns=list(TREE_COLUMNS))


def empty_species_table() -> pd.DataFrame:
    """Return an empty species lookup with the canonical columns."""
    return pd.DataFrame(columns=list(SPECIES_COLUMNS))


def _missing_columns(frame: pd.DataFrame, required: Iterable[str], table: str) -> list[str]:
    missing = [column for column in required if column not in frame.columns]
    if not missing:
        return []
    return [f"{table} table is missing required column(s): {', '.join(missing)}"]


def _blank_mask(series: pd.Series) -> pd.Series:
    strings = series.astype("string")
    return strings.isna() | strings.str.strip().eq("")


def _numeric_errors(
    frame: pd.DataFrame,
    column: str,
    table: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[str]:
    if column not in frame.columns:
        return []
    numeric = pd.to_numeric(frame[column], errors="coerce")
    invalid = frame[column].notna() & numeric.isna()
    errors: list[str] = []
    if bool(invalid.any()):
        errors.append(f"{table}.{column} contains non-numeric values")
    values = numeric.dropna()
    if values.empty:
        return errors
    if minimum is not None and bool((values < minimum).any()):
        errors.append(f"{table}.{column} must be >= {minimum}")
    if maximum is not None and bool((values > maximum).any()):
        errors.append(f"{table}.{column} must be <= {maximum}")
    if not bool(np.isfinite(values.to_numpy(dtype=float)).all()):
        errors.append(f"{table}.{column} contains non-finite values")
    return errors


def _duplicate_errors(frame: pd.DataFrame, columns: Sequence[str], table: str) -> list[str]:
    if any(column not in frame.columns for column in columns):
        return []
    subset = frame.loc[:, list(columns)].dropna()
    if subset.empty:
        return []
    if bool(subset.duplicated().any()):
        return [f"{table} table has duplicate key(s): {', '.join(columns)}"]
    return []


def _validate_provenance(provenance: InventoryProvenance) -> list[str]:
    errors: list[str] = []
    for name in ("source_id", "source_version", "retrieved_utc"):
        value = getattr(provenance, name)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"provenance.{name} must be a non-empty string")
    if isinstance(provenance.retrieved_utc, str) and provenance.retrieved_utc.strip():
        try:
            timestamp = pd.Timestamp(provenance.retrieved_utc)
        except (TypeError, ValueError):
            errors.append("provenance.retrieved_utc must be an ISO-8601 timestamp")
        else:
            if timestamp.tzinfo is None:
                errors.append("provenance.retrieved_utc must include a timezone")
    return errors


def _validate_units(units: InventoryUnits) -> list[str]:
    errors: list[str] = []
    for name, allowed in ALLOWED_UNITS.items():
        value = getattr(units, name)
        if value not in allowed:
            options = ", ".join(sorted(allowed))
            errors.append(f"units.{name} must be one of: {options}; got {value!r}")
    return errors


def _validate_plots(release: InventoryRelease) -> list[str]:
    frame = release.plots
    errors = _missing_columns(frame, PLOT_REQUIRED, "plots")
    if "plot_id" in frame.columns:
        if bool(_blank_mask(frame["plot_id"]).any()):
            errors.append("plots.plot_id must not be blank")
        errors.extend(_duplicate_errors(frame, ("plot_id",), "plots"))
    errors.extend(_numeric_errors(frame, "baf", "plots", minimum=0.0))
    errors.extend(_numeric_errors(frame, "tree_count", "plots", minimum=0.0))
    errors.extend(_numeric_errors(frame, "latitude", "plots", minimum=-90.0, maximum=90.0))
    errors.extend(_numeric_errors(frame, "longitude", "plots", minimum=-180.0, maximum=180.0))
    has_coordinates = (
        frame.get("latitude", pd.Series(dtype=float)).notna().any()
        or frame.get("longitude", pd.Series(dtype=float)).notna().any()
    )
    if has_coordinates and not release.crs:
        errors.append("release.crs is required when plot coordinates are present")
    return errors


def _validate_trees(release: InventoryRelease) -> list[str]:
    frame = release.trees
    errors = _missing_columns(frame, TREE_REQUIRED, "trees")
    if "plot_id" in frame.columns:
        if bool(_blank_mask(frame["plot_id"]).any()):
            errors.append("trees.plot_id must not be blank")
        if "plot_id" in release.plots.columns:
            plot_ids = set(release.plots["plot_id"].dropna().astype(str))
            tree_ids = set(frame["plot_id"].dropna().astype(str))
            orphaned = sorted(tree_ids - plot_ids)
            if orphaned:
                errors.append(
                    "trees.plot_id has values absent from plots: " + ", ".join(orphaned[:5])
                )
    errors.extend(_numeric_errors(frame, "dbh_cm", "trees", minimum=0.0))
    errors.extend(_numeric_errors(frame, "height_m", "trees", minimum=0.0))
    errors.extend(_numeric_errors(frame, "tally", "trees", minimum=0.0))
    if "tree_id" in frame.columns:
        tree_rows = frame.loc[frame["tree_id"].notna()]
        tally_rows = frame.loc[frame["tree_id"].isna()]
        errors.extend(_duplicate_errors(tree_rows, ("plot_id", "tree_id"), "trees"))
        errors.extend(_duplicate_errors(tally_rows, ("plot_id", "dbh_cm"), "trees"))
    if "species_code" in frame.columns and bool(frame["species_code"].notna().any()):
        if bool(_blank_mask(frame["species_code"]).any()):
            errors.append("trees.species_code must not be blank when present")
        if release.species.empty:
            errors.append("trees.species_code values require a species lookup table")
        elif "species_code" in release.species.columns:
            known = set(release.species["species_code"].dropna().astype(str))
            used = set(frame["species_code"].dropna().astype(str))
            unknown = sorted(used - known)
            if unknown:
                errors.append(
                    "trees.species_code has values absent from species: " + ", ".join(unknown[:5])
                )
    return errors


def _validate_species(release: InventoryRelease) -> list[str]:
    frame = release.species
    if frame.empty and len(frame.columns) == 0:
        return []
    errors = _missing_columns(frame, SPECIES_REQUIRED, "species")
    if "species_code" in frame.columns:
        if bool(_blank_mask(frame["species_code"]).any()):
            errors.append("species.species_code must not be blank")
        errors.extend(_duplicate_errors(frame, ("species_code",), "species"))
    return errors


def validate_inventory_release(release: InventoryRelease) -> None:
    """Validate a release against ``nemora.inventory.v1``.

    Raises
    ------
    InventoryValidationError
        If the release id, schema tables, units, or provenance are invalid.
    """
    errors: list[str] = []
    if not isinstance(release.release_id, str) or not release.release_id.strip():
        errors.append("release_id must be a non-empty string")
    if not isinstance(release.provenance, InventoryProvenance):
        errors.append("provenance must be an InventoryProvenance")
    else:
        errors.extend(_validate_provenance(release.provenance))
    if not isinstance(release.units, InventoryUnits):
        errors.append("units must be an InventoryUnits")
    else:
        errors.extend(_validate_units(release.units))
    for name in ("plots", "trees", "species"):
        if not isinstance(getattr(release, name), pd.DataFrame):
            errors.append(f"{name} must be a pandas DataFrame")
    if errors:
        raise InventoryValidationError(errors)

    errors.extend(_validate_plots(release))
    errors.extend(_validate_trees(release))
    errors.extend(_validate_species(release))
    if errors:
        raise InventoryValidationError(errors)


def _write_table(frame: pd.DataFrame, columns: Sequence[str], path: Path) -> None:
    normalized = frame.reindex(columns=list(columns))
    normalized.to_csv(path, index=False, lineterminator="\n")


def write_inventory_release(release: InventoryRelease, directory: str | Path) -> None:
    """Validate and write a canonical release directory.

    The directory contains ``release.json``, ``plots.csv``, ``trees.csv``, and
    ``species.csv``. CSV columns are always the full schema, so a minimal input
    table still produces a stable artifact.
    """
    validate_inventory_release(release)
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        release.to_manifest(),
        indent=2,
        sort_keys=True,
        default=str,
    )
    (target / "release.json").write_text(payload + "\n", encoding="utf-8")
    _write_table(release.plots, PLOT_COLUMNS, target / _TABLE_FILES["plots"])
    _write_table(release.trees, TREE_COLUMNS, target / _TABLE_FILES["trees"])
    _write_table(release.species, SPECIES_COLUMNS, target / _TABLE_FILES["species"])


def _read_csv(path: Path, table: str) -> pd.DataFrame:
    dtypes: dict[str, str] = {
        "plot_id": "string",
        "cluster_id": "string",
        "measurement_date": "string",
        "sample_type": "string",
        "site_identifier": "string",
        "tree_id": "string",
        "species_code": "string",
        "status": "string",
    }
    try:
        return pd.read_csv(path, dtype=dtypes)
    except FileNotFoundError as exc:
        raise InventoryValidationError([f"missing {table} table: {path}"]) from exc


def read_inventory_release(directory: str | Path) -> InventoryRelease:
    """Read and validate a canonical release directory."""
    target = Path(directory)
    manifest_path = target / "release.json"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InventoryValidationError([f"missing release manifest: {manifest_path}"]) from exc
    except json.JSONDecodeError as exc:
        raise InventoryValidationError([f"invalid release manifest JSON: {exc}"]) from exc

    if not isinstance(payload, dict):
        raise InventoryValidationError(["release manifest must be a JSON object"])
    required = {"schema_version", "release_id", "units", "provenance", "tables"}
    missing = sorted(required - set(payload))
    if missing:
        raise InventoryValidationError(
            [f"release manifest is missing required key(s): {', '.join(missing)}"]
        )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise InventoryValidationError(
            [
                "release manifest schema_version must be "
                f"{SCHEMA_VERSION!r}; got {payload['schema_version']!r}"
            ]
        )

    tables = payload["tables"]
    if not isinstance(tables, dict) or any(name not in tables for name in _TABLE_FILES):
        raise InventoryValidationError(
            ["release manifest tables must define plots, trees, and species"]
        )

    try:
        units = InventoryUnits(**payload["units"])
    except (TypeError, ValueError) as exc:
        raise InventoryValidationError([f"invalid units block: {exc}"]) from exc
    try:
        provenance = InventoryProvenance(**payload["provenance"])
    except (TypeError, ValueError) as exc:
        raise InventoryValidationError([f"invalid provenance block: {exc}"]) from exc
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise InventoryValidationError(["release metadata must be a JSON object"])

    release = InventoryRelease(
        release_id=str(payload["release_id"]),
        provenance=provenance,
        plots=_read_csv(target / str(tables["plots"]), "plots"),
        trees=_read_csv(target / str(tables["trees"]), "trees"),
        species=_read_csv(target / str(tables["species"]), "species"),
        units=units,
        crs=payload.get("crs"),
        metadata=metadata,
    )
    validate_inventory_release(release)
    return release


def inventory_schema() -> dict[str, Any]:
    """Return the portable schema document for external consumers."""
    return {
        "schema_version": SCHEMA_VERSION,
        "tables": {
            "plots": {
                "columns": list(PLOT_COLUMNS),
                "required": list(PLOT_REQUIRED),
                "key": ["plot_id"],
            },
            "trees": {
                "columns": list(TREE_COLUMNS),
                "required": list(TREE_REQUIRED),
                "key": ["plot_id", "tree_id"],
                "tally_key": ["plot_id", "dbh_cm"],
            },
            "species": {
                "columns": list(SPECIES_COLUMNS),
                "required": list(SPECIES_REQUIRED),
                "key": ["species_code"],
            },
        },
        "units": {name: sorted(values) for name, values in ALLOWED_UNITS.items()},
        "provenance_required": ["source_id", "source_version", "retrieved_utc"],
    }


__all__ = [
    "ALLOWED_UNITS",
    "PLOT_COLUMNS",
    "PLOT_REQUIRED",
    "SCHEMA_VERSION",
    "SPECIES_COLUMNS",
    "SPECIES_REQUIRED",
    "TREE_COLUMNS",
    "TREE_REQUIRED",
    "InventoryProvenance",
    "InventoryRelease",
    "InventoryUnits",
    "InventoryValidationError",
    "empty_plots_table",
    "empty_species_table",
    "empty_trees_table",
    "inventory_schema",
    "read_inventory_release",
    "validate_inventory_release",
    "write_inventory_release",
]
