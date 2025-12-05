"""Lightweight exporters for upcoming synthesis artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .tessellation import VoronoiSeedConfig, VoronoiSeedResult

__all__ = [
    "export_geojson",
    "export_metadata_json",
    "export_seed_recipe",
    "seed_recipe_payload",
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
