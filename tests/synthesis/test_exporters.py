from __future__ import annotations

import json
from pathlib import Path

from nemora.synthesis import exporters


def test_export_metadata_json(tmp_path: Path) -> None:
    target = tmp_path / "meta.json"
    exporters.export_metadata_json({"n": 10, "cv": 0.35}, target)
    payload = json.loads(target.read_text())
    assert payload["n"] == 10


def test_export_geojson(tmp_path: Path) -> None:
    target = tmp_path / "landscape.geojson"
    feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": []},
        "properties": {"id": 1},
    }
    exporters.export_geojson([feature], target, crs="EPSG:3857")
    payload = json.loads(target.read_text())
    assert payload["type"] == "FeatureCollection"
    assert payload["crs"]["properties"]["name"] == "EPSG:3857"
