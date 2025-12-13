"""Lightweight exporters for upcoming synthesis artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from . import stands, stems

if TYPE_CHECKING:
    from .stands import StandAttributeSample
    from .tessellation import VoronoiSeedConfig, VoronoiSeedResult

__all__ = [
    "export_geojson",
    "export_metadata_json",
    "export_seed_recipe",
    "seed_recipe_payload",
    "export_stand_geojson_from_polygons",
    "export_tree_geojson",
    "tree_records_to_dataframe",
    "export_tree_table",
]


def export_metadata_json(metadata: Mapping[str, object], path: Path) -> None:
    """Write synthesis metadata (control knobs, metrics, provenance) to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def export_geojson(
    features: Iterable[Mapping[str, object]],
    path: Path,
    crs: str | None = None,
) -> None:
    """Emit a FeatureCollection skeleton for downstream GIS tooling."""

    feature_list = list(features)
    collection: MutableMapping[str, object] = {
        "type": "FeatureCollection",
        "features": feature_list,
    }
    if crs is not None:
        collection["crs"] = {
            "type": "name",
            "properties": {"name": crs},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(collection, indent=2), encoding="utf-8")


def seed_recipe_payload(
    result: VoronoiSeedResult,
    *,
    include_points: bool = True,
    include_polygons: bool = False,
) -> dict[str, Any]:
    """Return a JSON-ready payload describing the seed configuration + metadata."""

    payload: dict[str, Any] = {
        "config": _config_payload(result.config),
        "metadata": result.metadata(),
    }
    if include_points:
        payload["points"] = result.points.tolist()
        payload["hole_points"] = result.hole_points.tolist()
        payload["merge_pairs"] = result.merge_pairs.tolist()
    if include_polygons:
        payload["polygons"] = [polygon.tolist() for polygon in result.polygons]
    return payload


def export_seed_recipe(
    result: VoronoiSeedResult,
    path: Path,
    *,
    include_points: bool = True,
    include_polygons: bool = False,
) -> None:
    """Persist a seed recipe JSON (config + metadata, optionally raw coordinates)."""

    payload = seed_recipe_payload(
        result,
        include_points=include_points,
        include_polygons=include_polygons,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def export_stand_geojson_from_polygons(
    polygons: Sequence[np.ndarray],
    samples: Sequence[StandAttributeSample],
    path: Path,
    *,
    crs: str | None = None,
    strict: bool = False,
    expected_count: int | None = None,
    assignments: Sequence[stands.StandBootstrapAssignment] | None = None,
    bootstrap_library: Mapping[str, stands.StandBootstrapLibraryEntry] | None = None,
) -> int:
    """Export a GeoJSON pairing polygons and stand samples.

    Parameters
    ----------
    polygons:
        Iterable of polygon arrays (already filtered to remove empty polygons).
    samples:
        Stand attribute samples produced by `sample_stand_attributes`.
    path:
        Output GeoJSON file path.
    crs:
        Optional CRS identifier stored in the GeoJSON metadata.
    strict:
        When True, require the generated feature count to match ``expected_count`` (or the min of
        polygons/samples when the expectation is omitted). A mismatch raises ``ValueError``.
    expected_count:
        Optional explicit expected feature count for strict mode.
    """

    features = stands.build_stand_features(
        polygons,
        samples,
        assignments=assignments,
        bootstrap_library=bootstrap_library,
    )
    assigned = len(features)
    if not features:
        raise ValueError("No stand features could be built from the provided inputs.")
    if strict:
        expected = (
            expected_count if expected_count is not None else min(len(polygons), len(samples))
        )
        if assigned != expected:
            raise ValueError(
                "Strict assignment requires a 1:1 mapping between polygons and samples "
                f"(expected {expected}, assigned {assigned})."
            )
    export_geojson(features, path, crs=crs)
    return assigned


def _config_payload(config: VoronoiSeedConfig) -> dict[str, Any]:
    lattice_resolution = None
    if config.lattice.resolution is not None:
        lattice_resolution = list(config.lattice.resolution)
    return {
        "count": config.count,
        "aspect_ratio": config.aspect_ratio,
        "mix": {
            "uniform": config.mix.uniform,
            "cluster": config.mix.cluster,
            "inhibition": config.mix.inhibition,
            "lattice": config.mix.lattice,
        },
        "cluster": {
            "size": config.cluster.size,
            "spread": config.cluster.spread,
        },
        "inhibition": {
            "min_distance": config.inhibition.min_distance,
            "max_attempts_per_point": config.inhibition.max_attempts_per_point,
        },
        "lattice": {
            "resolution": lattice_resolution,
            "jitter": config.lattice.jitter,
        },
        "edit": {
            "hole_fraction": config.edit.hole_fraction,
            "merge_fraction": config.edit.merge_fraction,
        },
    }


def export_tree_geojson(
    records: Sequence[Mapping[str, object]],
    path: Path,
    *,
    crs: str | None = None,
) -> None:
    """Export stem records (with DBH/attributes) as a GeoJSON FeatureCollection."""

    features: list[dict[str, object]] = []
    for record in records:
        x = _coerce_float(record.get("x"))
        y = _coerce_float(record.get("y"))
        props = dict(record)
        props.pop("x", None)
        props.pop("y", None)
        attrs = props.get("attributes")
        if isinstance(attrs, stems.TreeAttributes):
            props["attributes"] = {
                "dbh_cm": attrs.dbh_cm,
                "height_m": attrs.height_m,
                "crown_ratio": attrs.crown_ratio,
                "basal_area_m2": attrs.basal_area_m2,
                "biomass_tonnes": attrs.biomass_tonnes,
                "bark_thickness_cm": attrs.bark_thickness_cm,
            }
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {"type": "Point", "coordinates": [x, y]},
            }
        )
    export_geojson(features, path, crs=crs)


def _coerce_float(value: object | None) -> float:
    if isinstance(value, int | float | np.floating):
        return float(value)
    if isinstance(value, str | bytes | bytearray):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def tree_records_to_dataframe(records: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Convert stem records to a flat DataFrame (attributes expanded when present)."""

    rows: list[dict[str, object]] = []
    for record in records:
        row = dict(record)
        attrs = row.pop("attributes", None)
        if isinstance(attrs, stems.TreeAttributes):
            row.update(
                {
                    "dbh_cm": attrs.dbh_cm,
                    "height_m": attrs.height_m,
                    "crown_ratio": attrs.crown_ratio,
                    "basal_area_m2": attrs.basal_area_m2,
                    "biomass_tonnes": attrs.biomass_tonnes,
                    "bark_thickness_cm": attrs.bark_thickness_cm,
                }
            )
        elif isinstance(attrs, Mapping):
            row.update(dict(attrs))
        rows.append(row)
    return pd.DataFrame.from_records(rows)


def export_tree_table(
    records: Sequence[Mapping[str, object]],
    path: Path,
) -> None:
    """Export stem records (flat) to CSV or Parquet based on file suffix."""

    frame = tree_records_to_dataframe(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
