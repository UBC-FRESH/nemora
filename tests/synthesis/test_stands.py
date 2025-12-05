from __future__ import annotations

from pathlib import Path

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
