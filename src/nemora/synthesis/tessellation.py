"""Seed-point configuration helpers for Voronoi-based landscape tiling.

The CJFR rlandscape paper describes a mixture of four point processes and two
editing knobs (hole/merge fractions) that collectively control the number of
management units, polygon area variation, and vertex-degree statistics.  This
module captures those inputs so downstream synthesis code (Phase 1 of the plan)
can generate repeatable seed sets and share the configuration with docs/tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["PointProcessMix", "VoronoiSeedConfig", "generate_seed_points"]


@dataclass(slots=True)
class PointProcessMix:
    """Proportions for the four rlandscape point processes."""

    uniform: float = 1.0
    cluster: float = 0.0
    inhibition: float = 0.0
    lattice: float = 0.0

    def normalized(self) -> PointProcessMix:
        """Return a version scaled so the proportions sum to 1."""

        weights = np.array(
            [self.uniform, self.cluster, self.inhibition, self.lattice],
            dtype=float,
        )
        total = float(weights.sum())
        if total <= 0.0:
            raise ValueError("At least one point-process weight must be positive.")
        weights = weights / total
        return PointProcessMix(*weights.tolist())


@dataclass(slots=True)
class VoronoiSeedConfig:
    """Input knobs for the (future) Voronoi generator."""

    count: int
    aspect_ratio: float = 1.0
    mix: PointProcessMix = field(default_factory=PointProcessMix)
    cluster_spread: float = 25.0
    cluster_size: int = 5
    inhibition_distance: float | None = None
    lattice_resolution: tuple[int, int] | None = None
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("VoronoiSeedConfig.count must be positive.")
        if self.aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be > 0.")


def generate_seed_points(config: VoronoiSeedConfig) -> np.ndarray:
    """Generate provisional seed coordinates for Phase 1 prototypes.

    For now we only support the uniform process so the scaffolding can land in
    the repository; the cluster/inhibition/lattice paths will arrive alongside
    the Phase 1 implementation.  The placeholder still enforces the aspect
    ratio contract so tests/docs can reason about coordinate bounds.
    """

    mix = config.mix.normalized()
    if mix.cluster or mix.inhibition or mix.lattice:
        raise NotImplementedError(
            "Non-uniform point-process mixtures will be implemented with the "
            "Phase 1 tessellation workstream.",
        )
    rng = config.rng or np.random.default_rng()
    xs = rng.random(config.count) * config.aspect_ratio
    ys = rng.random(config.count)
    return np.column_stack((xs, ys))
