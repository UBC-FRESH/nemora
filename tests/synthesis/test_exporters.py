from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nemora.synthesis import exporters, tessellation


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


def test_seed_recipe_payload_includes_config_when_points_disabled(tmp_path: Path) -> None:
    cfg = tessellation.VoronoiSeedConfig(
        count=5,
        mix=tessellation.PointProcessMix(uniform=0.5, cluster=0.5),
        rng=np.random.default_rng(42),
    )
    result = tessellation.generate_seed_points(cfg)
    payload = exporters.seed_recipe_payload(result, include_points=False)
    assert payload["config"]["count"] == 5
    assert "points" not in payload
    target = tmp_path / "recipe.json"
    exporters.export_seed_recipe(result, target, include_points=False)
    persisted = json.loads(target.read_text())
    assert persisted["config"]["mix"]["cluster"] == 0.5
    assert persisted["metadata"]["metrics"]["polygon_count"] == 5
