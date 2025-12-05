from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import numpy as np

from nemora.synthesis import stands


def test_build_templates_and_sampling() -> None:
    records = [
        {
            "vegetation_type": "fir",
            "age_classes": (("0-20", 0.4), ("20-40", 1.0)),
            "patch_weibull": (2.0, 5.0, 1.0),
        }
    ]
    templates = stands.build_templates(records)
    assert len(templates) == 1
    template = templates[0]
    rng = np.random.default_rng(42)
    age = template.sample_age_class(rng)
    assert age in {"0-20", "20-40"}
    patch_sizes = template.sample_patch_size(rng, size=3)
    assert patch_sizes.shape == (3,)
    assert np.all(patch_sizes >= 1.0)


def test_sample_stand_attributes_and_loader(tmp_path: Path) -> None:
    json_path = tmp_path / "templates.json"
    json_path.write_text(
        '[{"vegetation_type":"fir","age_classes":[["0-20",1.0]],"patch_weibull":[1,2,1]},'
        '{"vegetation_type":"pine","age_classes":[["mature",1.0]],"patch_weibull":[1,1,1]}]',
        encoding="utf-8",
    )
    templates = stands.load_templates_from_json(json_path)
    assert len(templates) == 2
    rng = np.random.default_rng(123)
    samples = stands.sample_stand_attributes(templates, total_area=5.0, rng=rng)
    assert samples
    assert sum(sample.area for sample in samples) <= 5.0 + 1e-6


def test_load_stand_samples_from_json(tmp_path: Path) -> None:
    json_path = tmp_path / "samples.json"
    json_path.write_text(
        '[{"vegetation_type":"fir","age_class":"20-40","area":3.4}]',
        encoding="utf-8",
    )
    samples = stands.load_samples_from_json(json_path)
    assert len(samples) == 1
    assert samples[0].vegetation_type == "fir"


def test_build_stand_features_pairs_polygons() -> None:
    polygons = [
        np.array([[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]], dtype=float),
        np.array([[0.5, 0.5], [0.7, 0.5], [0.6, 0.7]], dtype=float),
    ]
    samples = [
        stands.StandAttributeSample("fir", "20-40", 3.0),
        stands.StandAttributeSample("pine", "40-60", 2.0),
    ]
    features = stands.build_stand_features(polygons, samples)
    assert len(features) == 2
    props = cast(dict[str, object], features[0]["properties"])
    assert props["veg_type"] == "fir"


def test_load_bootstrap_plan_and_assignments(tmp_path: Path) -> None:
    bootstrap_a = tmp_path / "bootstrap_a.json"
    bootstrap_a.write_text(
        json.dumps(
            {
                "metadata": {"distribution": "weibull", "resamples": 2},
                "dbh_vectors": {"0": [10.0, 12.0], "1": [11.0]},
            }
        ),
        encoding="utf-8",
    )
    bootstrap_b = tmp_path / "bootstrap_b.json"
    bootstrap_b.write_text(
        json.dumps(
            {
                "metadata": {"distribution": "lognormal", "resamples": 1},
                "dbh_vectors": {"0": [15.0]},
            }
        ),
        encoding="utf-8",
    )
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "name": "fir-old",
                        "vegetation_type": "fir",
                        "bootstrap": bootstrap_a.name,
                    }
                ],
                "default_bootstrap": bootstrap_b.name,
            }
        ),
        encoding="utf-8",
    )
    plan = stands.load_bootstrap_plan(plan_path)
    samples = [
        stands.StandAttributeSample("fir", "60-80", 3.0),
        stands.StandAttributeSample("pine", "20-40", 2.0),
    ]
    assignments, library = stands.build_bootstrap_assignments(
        samples,
        plan,
        id_prefix="unit",
    )
    assert len(assignments) == 2
    assert assignments[0].bootstrap_id == "fir-old"
    assert assignments[1].bootstrap_id != assignments[0].bootstrap_id
    assert set(library.keys()) == {assignments[0].bootstrap_id, assignments[1].bootstrap_id}
    fir_payload = library[assignments[0].bootstrap_id]
    assert fir_payload.metadata["distribution"] == "weibull"
    assert "0" in fir_payload.dbh_vectors


def test_build_stand_features_includes_bootstrap_metadata(tmp_path: Path) -> None:
    polygons = [
        np.array([[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]], dtype=float),
    ]
    samples = [
        stands.StandAttributeSample("fir", "60-80", 3.0),
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "attributes_source": "attributes.json",
                "plan_source": "plan.json",
                "bootstraps": {
                    "fir-old": {
                        "source": "bootstrap_a.json",
                        "metadata": {"distribution": "weibull", "parameters": {"shape": 2.1}},
                        "dbh_vectors": {"0": [10.0, 12.0]},
                    }
                },
                "assignments": [
                    {
                        "stand_id": "stand-0001",
                        "vegetation_type": "fir",
                        "age_class": "60-80",
                        "area": 3.0,
                        "bootstrap_id": "fir-old",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = stands.load_bootstrap_manifest(manifest_path)
    features = stands.build_stand_features(
        polygons,
        samples,
        assignments=manifest.assignments,
        bootstrap_library=manifest.bootstraps,
    )
    props = cast(dict[str, object], features[0]["properties"])
    assert props["stand_id"] == "stand-0001"
    assert props["bootstrap_id"] == "fir-old"
    bootstrap_meta = cast(dict[str, object], props["bootstrap_metadata"])
    assert bootstrap_meta["distribution"] == "weibull"
    assert cast(dict[str, object], bootstrap_meta["parameters"])["shape"] == 2.1
