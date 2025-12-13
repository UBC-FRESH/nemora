from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np

from nemora.synthesis import exporters, stands, stems, tessellation


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
    mask_polygon = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    mask = tessellation.MaskGeometry(polygons=[mask_polygon], name="unit-square")
    cfg = tessellation.VoronoiSeedConfig(
        count=5,
        mix=tessellation.PointProcessMix(uniform=0.5, cluster=0.5),
        rng=np.random.default_rng(42),
        mask=mask,
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
    mask_metadata = cast(dict[str, object], persisted["metadata"]["mask"])
    primary = cast(dict[str, object], mask_metadata["primary"])
    assert primary["name"] == "unit-square"


def test_seed_recipe_payload_can_include_polygons(tmp_path: Path) -> None:
    cfg = tessellation.VoronoiSeedConfig(
        count=3,
        rng=np.random.default_rng(7),
    )
    result = tessellation.generate_seed_points(cfg)
    target = tmp_path / "recipe_polygons.json"
    exporters.export_seed_recipe(
        result,
        target,
        include_points=False,
        include_polygons=True,
    )
    payload = json.loads(target.read_text())
    polygons = cast(list[list[list[float]]], payload["polygons"])
    assert len(polygons) == 3
    assert isinstance(polygons[0][0][0], float)


def test_export_stand_geojson_from_polygons(tmp_path: Path) -> None:
    polygons = [
        np.array([[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]], dtype=float),
        np.array([[0.5, 0.5], [0.7, 0.5], [0.6, 0.7]], dtype=float),
    ]
    samples = [
        stands.StandAttributeSample("fir", "20-40", 3.0),
        stands.StandAttributeSample("pine", "40-60", 2.0),
    ]
    path = tmp_path / "stands.geojson"
    assigned = exporters.export_stand_geojson_from_polygons(polygons, samples, path)
    assert assigned == 2
    payload = json.loads(path.read_text())
    assert payload["type"] == "FeatureCollection"


def test_export_tree_geojson(tmp_path: Path) -> None:
    records = [
        {
            "stand_id": "stand-0001",
            "bootstrap_id": "bootstrap-1",
            "sampler_type": "analytic",
            "x": 0.1,
            "y": 0.2,
            "dbh": 22.5,
            "attributes": stems.TreeAttributes(
                dbh_cm=22.5,
                height_m=14.6,
                crown_ratio=0.4,
                basal_area_m2=0.04,
                biomass_tonnes=0.003,
                bark_thickness_cm=0.45,
            ),
        }
    ]
    output = tmp_path / "trees.geojson"
    exporters.export_tree_geojson(records, output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 1
    feature = payload["features"][0]
    assert feature["geometry"]["type"] == "Point"
    assert feature["properties"]["stand_id"] == "stand-0001"
    attrs = feature["properties"]["attributes"]
    assert attrs["dbh_cm"] == 22.5
