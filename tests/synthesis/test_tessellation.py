from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from nemora.synthesis import tessellation


def test_uniform_seed_generation_respects_aspect_ratio() -> None:
    cfg = tessellation.VoronoiSeedConfig(
        count=10,
        aspect_ratio=2.0,
        rng=np.random.default_rng(1234),
    )
    result = tessellation.generate_seed_points(cfg)
    assert result.points.shape == (10, 2)
    assert np.all(result.points[:, 0] >= 0)
    assert np.all(result.points[:, 0] <= 2.0 + 1e-9)
    assert np.all(result.points[:, 1] >= 0)
    assert np.all(result.points[:, 1] <= 1.0 + 1e-9)


def test_cluster_only_mix_is_supported() -> None:
    mix = tessellation.PointProcessMix(uniform=0.0, cluster=1.0)
    cfg = tessellation.VoronoiSeedConfig(
        count=7,
        mix=mix,
        rng=np.random.default_rng(9876),
    )
    result = tessellation.generate_seed_points(cfg)
    assert result.points.shape == (7, 2)
    assert result.process_counts["cluster"] == 7
    assert result.process_counts["uniform"] == 0


def test_editing_knobs_respected() -> None:
    cfg = tessellation.VoronoiSeedConfig(
        count=20,
        mix=tessellation.PointProcessMix(uniform=1.0),
        edit=tessellation.VoronoiEditConfig(hole_fraction=0.1, merge_fraction=0.2),
        rng=np.random.default_rng(4321),
    )
    result = tessellation.generate_seed_points(cfg)
    assert result.points.shape[0] == 20
    hole_expected = int(round(0.1 * 20))
    merge_expected = int(round(0.2 * 20))
    assert result.hole_points.shape[0] == hole_expected
    assert result.merge_pairs.shape[0] == merge_expected
    metadata = result.metadata()
    assert cast(int, metadata["initial_seed_count"]) == 20 + hole_expected + merge_expected
    edit = cast(dict[str, object], metadata["edit"])
    assert cast(int, edit["hole_count"]) == hole_expected
    assert cast(int, edit["merge_count"]) == merge_expected


def test_invalid_edit_fraction_rejected() -> None:
    with pytest.raises(ValueError):
        tessellation.VoronoiEditConfig(hole_fraction=0.6, merge_fraction=0.5)
