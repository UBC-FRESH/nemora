from __future__ import annotations

import numpy as np
import pytest

from nemora.synthesis import tessellation


def test_uniform_seed_generation_respects_aspect_ratio() -> None:
    cfg = tessellation.VoronoiSeedConfig(count=10, aspect_ratio=2.0)
    points = tessellation.generate_seed_points(cfg)
    assert points.shape == (10, 2)
    assert np.all(points[:, 0] >= 0)
    assert np.all(points[:, 0] <= 2.0 + 1e-9)
    assert np.all(points[:, 1] >= 0)
    assert np.all(points[:, 1] <= 1.0 + 1e-9)


def test_non_uniform_mix_not_yet_supported() -> None:
    mix = tessellation.PointProcessMix(uniform=0.5, cluster=0.5)
    cfg = tessellation.VoronoiSeedConfig(count=5, mix=mix)
    with pytest.raises(NotImplementedError):
        tessellation.generate_seed_points(cfg)
