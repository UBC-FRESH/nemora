"""Lightweight exporters for upcoming synthesis artifacts."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, MutableMapping
from pathlib import Path

__all__ = ["export_metadata_json", "export_geojson"]


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
