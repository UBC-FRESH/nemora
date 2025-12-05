"""Seed-point configuration helpers for Voronoi-based landscape tiling.

The CJFR rlandscape paper describes a mixture of four point processes and two
editing knobs (hole/merge fractions) that collectively control the number of
management units, polygon area variation, and vertex-degree statistics. This
module captures those inputs so downstream synthesis code (Phase 1 of the plan)
can generate repeatable seed sets and share the configuration with docs/tests.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import QhullError, Voronoi

__all__ = [
    "ClusterConfig",
    "InhibitionConfig",
    "LatticeConfig",
    "PointProcessMix",
    "VoronoiEditConfig",
    "VoronoiSeedConfig",
    "VoronoiMetrics",
    "VoronoiSeedResult",
    "generate_seed_points",
]


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

    def as_array(self) -> np.ndarray:
        """Return the normalised weights as an array."""

        norm = self.normalized()
        return np.array(
            [norm.uniform, norm.cluster, norm.inhibition, norm.lattice],
            dtype=float,
        )


@dataclass(slots=True)
class ClusterConfig:
    """Parameters controlling the cluster point process."""

    size: int = 6
    spread: float = 0.05  # expressed as fraction of the unit square

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("Cluster size must be positive.")
        if self.spread <= 0:
            raise ValueError("Cluster spread must be positive.")


@dataclass(slots=True)
class InhibitionConfig:
    """Parameters controlling the inhibition (SSI) process."""

    min_distance: float | None = None
    max_attempts_per_point: int = 500

    def resolved_distance(self, aspect_ratio: float, count: int) -> float:
        """Derive a fallback inhibition distance when none was provided."""

        if self.min_distance is not None:
            return self.min_distance
        domain = math.sqrt(aspect_ratio)
        baseline = domain / max(count, 1) ** 0.5
        return 0.2 * baseline


@dataclass(slots=True)
class LatticeConfig:
    """Parameters controlling the lattice/grid process."""

    resolution: tuple[int, int] | None = None
    jitter: float = 0.0

    def resolved_resolution(self, count: int, aspect_ratio: float) -> tuple[int, int]:
        if self.resolution is not None:
            return self.resolution
        side = math.sqrt(count)
        nx = max(1, int(math.ceil(side * math.sqrt(aspect_ratio))))
        ny = max(1, int(math.ceil(side / math.sqrt(aspect_ratio))))
        return nx, ny


@dataclass(slots=True)
class VoronoiEditConfig:
    """Placeholder for polygon deletion/merging controls."""

    hole_fraction: float = 0.0
    merge_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.hole_fraction < 0 or self.merge_fraction < 0:
            raise ValueError("Edit fractions must be non-negative.")
        if self.hole_fraction >= 1 or self.merge_fraction >= 1:
            raise ValueError("Edit fractions must be < 1.")
        if self.hole_fraction + self.merge_fraction >= 1:
            raise ValueError("Hole + merge fractions must sum to < 1.")


@dataclass(slots=True)
class VoronoiSeedConfig:
    """Input knobs for the (future) Voronoi generator."""

    count: int
    aspect_ratio: float = 1.0
    mix: PointProcessMix = field(default_factory=PointProcessMix)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    inhibition: InhibitionConfig = field(default_factory=InhibitionConfig)
    lattice: LatticeConfig = field(default_factory=LatticeConfig)
    edit: VoronoiEditConfig = field(default_factory=VoronoiEditConfig)
    rng: np.random.Generator | None = None

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("VoronoiSeedConfig.count must be positive.")
        if self.aspect_ratio <= 0:
            raise ValueError("aspect_ratio must be > 0.")


@dataclass(slots=True)
class VoronoiMetrics:
    """Summary statistics matching the CJFR target controls."""

    polygon_count: int
    area_mean: float
    area_cv: float
    vertex_degree_mean: float
    vertex_degree_std: float

    def as_dict(self) -> dict[str, float]:
        return {
            "polygon_count": int(self.polygon_count),
            "area_mean": self.area_mean,
            "area_cv": self.area_cv,
            "vertex_degree_mean": self.vertex_degree_mean,
            "vertex_degree_std": self.vertex_degree_std,
        }


@dataclass(slots=True)
class VoronoiSeedResult:
    """Container for the generated seeds plus bookkeeping metadata."""

    points: np.ndarray
    config: VoronoiSeedConfig
    process_counts: dict[str, int]
    hole_points: np.ndarray
    merge_pairs: np.ndarray
    polygons: list[np.ndarray]
    metrics: VoronoiMetrics

    def metadata(self) -> dict[str, object]:
        """Return a JSON-serialisable summary of the seed configuration."""

        return {
            "target_count": self.config.count,
            "initial_seed_count": int(
                self.points.shape[0] + self.hole_points.shape[0] + self.merge_pairs.shape[0]
            ),
            "aspect_ratio": self.config.aspect_ratio,
            "mix": self.config.mix.as_array().tolist(),
            "process_counts": self.process_counts,
            "edit": {
                "hole_fraction": self.config.edit.hole_fraction,
                "hole_count": int(self.hole_points.shape[0]),
                "merge_fraction": self.config.edit.merge_fraction,
                "merge_count": int(self.merge_pairs.shape[0]),
            },
            "metrics": self.metrics.as_dict(),
        }


def generate_seed_points(config: VoronoiSeedConfig) -> VoronoiSeedResult:
    """Generate provisional seed coordinates for Phase 1 prototypes."""

    rng = config.rng or np.random.default_rng()
    target_count = config.count
    hole_target = _resolve_edit_count(config.edit.hole_fraction, target_count)
    merge_target = _resolve_edit_count(config.edit.merge_fraction, target_count)
    seed_count = target_count + hole_target + merge_target
    mix_array = config.mix.as_array()
    counts = rng.multinomial(seed_count, mix_array)
    generators: list[Callable[[int], np.ndarray]] = [
        lambda n: _generate_uniform(n, config, rng),
        lambda n: _generate_cluster(n, config, rng),
        lambda n: _generate_inhibition(n, config, rng),
        lambda n: _generate_lattice(n, config, rng),
    ]
    names = ("uniform", "cluster", "inhibition", "lattice")
    points: list[np.ndarray] = []
    process_counts: dict[str, int] = {}
    for name, count, gen in zip(names, counts, generators, strict=True):
        if count <= 0:
            process_counts[name] = 0
            continue
        samples = gen(int(count))
        process_counts[name] = samples.shape[0]
        points.append(samples)
    if not points:
        # Should not happen because seed_count > 0, but keep the uniform fallback handy.
        fallback = _generate_uniform(seed_count, config, rng)
        points.append(fallback)
        process_counts = {"uniform": fallback.shape[0], "cluster": 0, "inhibition": 0, "lattice": 0}
    stacked = np.vstack(points)
    (
        edited_points,
        hole_points,
        merge_pairs,
    ) = _apply_editing(stacked, rng, hole_target, merge_target)
    if edited_points.shape[0] != target_count:
        raise RuntimeError(
            "Editing pipeline failed to honour target count "
            f"(expected {target_count}, found {edited_points.shape[0]})."
        )
    polygons, degrees = _derive_voronoi_geometry(edited_points, config)
    metrics = _build_voronoi_metrics(polygons, degrees)
    return VoronoiSeedResult(
        points=edited_points,
        config=config,
        process_counts=process_counts,
        hole_points=hole_points,
        merge_pairs=merge_pairs,
        polygons=polygons,
        metrics=metrics,
    )


def _generate_uniform(
    count: int,
    config: VoronoiSeedConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    xs = rng.random(count) * config.aspect_ratio
    ys = rng.random(count)
    return np.column_stack((xs, ys))


def _generate_cluster(
    count: int,
    config: VoronoiSeedConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if count == 0:
        return np.zeros((0, 2))
    clusters = max(1, math.ceil(count / config.cluster.size))
    centers = _generate_uniform(clusters, config, rng)
    samples: list[np.ndarray] = []
    remaining = count
    scale = config.cluster.spread * min(config.aspect_ratio, 1.0)
    for center in centers:
        take = min(config.cluster.size, remaining)
        offsets = rng.normal(loc=0.0, scale=scale, size=(take, 2))
        pts = np.empty_like(offsets)
        pts[:, 0] = np.clip(center[0] + offsets[:, 0], 0.0, config.aspect_ratio)
        pts[:, 1] = np.clip(center[1] + offsets[:, 1], 0.0, 1.0)
        samples.append(pts)
        remaining -= take
        if remaining <= 0:
            break
    return np.vstack(samples)


def _generate_inhibition(
    count: int,
    config: VoronoiSeedConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if count == 0:
        return np.zeros((0, 2))
    min_distance = config.inhibition.resolved_distance(config.aspect_ratio, count)
    attempts_allowed = config.inhibition.max_attempts_per_point * count
    points = np.empty((count, 2), dtype=float)
    filled = 0
    attempts = 0
    while filled < count and attempts < attempts_allowed:
        candidate = _generate_uniform(1, config, rng)[0]
        if filled == 0:
            points[filled] = candidate
            filled += 1
            continue
        distances = np.linalg.norm(points[:filled] - candidate, axis=1)
        if np.all(distances >= min_distance):
            points[filled] = candidate
            filled += 1
        attempts += 1
    if filled < count:
        raise RuntimeError(
            "Failed to place all inhibition points; consider reducing the minimum distance."
        )
    return points


def _generate_lattice(
    count: int,
    config: VoronoiSeedConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    if count == 0:
        return np.zeros((0, 2))
    nx, ny = config.lattice.resolved_resolution(count, config.aspect_ratio)
    total_points = nx * ny
    if total_points < count:
        raise ValueError(
            "Lattice resolution does not provide enough points for the requested count."
        )
    xs = (np.arange(nx, dtype=float) + 0.5) * (config.aspect_ratio / nx)
    ys = (np.arange(ny, dtype=float) + 0.5) * (1.0 / ny)
    grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
    coordinates = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    indices = rng.choice(total_points, size=count, replace=False)
    selected = coordinates[indices]
    if config.lattice.jitter > 0:
        jitter_scale = config.lattice.jitter * min(config.aspect_ratio / nx, 1.0 / ny)
        jitter = rng.normal(scale=jitter_scale, size=selected.shape)
        selected = np.column_stack(
            (
                np.clip(selected[:, 0] + jitter[:, 0], 0.0, config.aspect_ratio),
                np.clip(selected[:, 1] + jitter[:, 1], 0.0, 1.0),
            )
        )
    return selected


def _resolve_edit_count(fraction: float, target_count: int) -> int:
    if fraction <= 0:
        return 0
    return max(0, int(round(target_count * fraction)))


def _apply_editing(
    points: np.ndarray,
    rng: np.random.Generator,
    hole_count: int,
    merge_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hole_count == 0 and merge_count == 0:
        return points, np.zeros((0, 2)), np.zeros((0, 2, 2))
    work = np.array(points, copy=True)
    hole_points = np.zeros((0, 2))
    if hole_count > 0:
        if hole_count >= work.shape[0]:
            raise ValueError(
                "Edit configuration removes all available seeds; reduce hole_fraction."
            )
        hole_indices = rng.choice(work.shape[0], size=hole_count, replace=False)
        hole_points = work[hole_indices]
        keep_mask = np.ones(work.shape[0], dtype=bool)
        keep_mask[hole_indices] = False
        work = work[keep_mask]
    merge_pairs = np.zeros((0, 2, 2))
    if merge_count > 0:
        available = work.shape[0]
        if merge_count > available // 2:
            merge_count = available // 2
        if merge_count > 0:
            indices = rng.choice(available, size=2 * merge_count, replace=False)
            merge_pairs = np.empty((merge_count, 2, 2), dtype=float)
            keep_mask = np.ones(available, dtype=bool)
            reshaped = indices.reshape(merge_count, 2)
            for pair_idx, pair in enumerate(reshaped):
                merge_pairs[pair_idx, 0] = work[pair[0]]
                merge_pairs[pair_idx, 1] = work[pair[1]]
                midpoint = work[pair].mean(axis=0)
                work[pair[0]] = midpoint
                keep_mask[pair[1]] = False
            work = work[keep_mask]
    return work, hole_points, merge_pairs


def _derive_voronoi_geometry(
    points: np.ndarray,
    config: VoronoiSeedConfig,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return clipped Voronoi polygons and per-seed vertex degrees."""

    if points.shape[0] == 0:
        return [], np.array([], dtype=float)
    try:
        vor = Voronoi(points)
        polygons = _clip_voronoi_regions(vor, config.aspect_ratio)
        degrees = _vertex_degrees_from_ridges(vor, points.shape[0])
        return polygons, degrees
    except (QhullError, ValueError):
        return _fallback_geometry(points, config.aspect_ratio)


def _clip_voronoi_regions(vor: Voronoi, aspect_ratio: float) -> list[np.ndarray]:
    bounds = (0.0, 0.0, aspect_ratio, 1.0)
    regions, vertices = _voronoi_finite_polygons_2d(vor)
    polygons: list[np.ndarray] = []
    for region in regions[: vor.points.shape[0]]:
        polygon = vertices[region]
        if polygon.size == 0:
            polygons.append(np.zeros((0, 2)))
            continue
        center = polygon.mean(axis=0)
        angles = np.arctan2(polygon[:, 1] - center[1], polygon[:, 0] - center[0])
        order = np.argsort(angles)
        polygon = polygon[order]
        clipped = _clip_polygon_to_bounds(polygon, bounds)
        if clipped.shape[0] < 3:
            clipped = np.zeros((0, 2))
        polygons.append(clipped)
    return polygons


def _vertex_degrees_from_ridges(vor: Voronoi, count: int) -> np.ndarray:
    adjacency: list[set[int]] = [set() for _ in range(count)]
    for p_a, p_b in vor.ridge_points:
        if p_a < count and p_b < count:
            adjacency[p_a].add(p_b)
            adjacency[p_b].add(p_a)
    return np.array([len(neighbors) for neighbors in adjacency], dtype=float)


def _fallback_geometry(
    points: np.ndarray, aspect_ratio: float
) -> tuple[list[np.ndarray], np.ndarray]:
    """Fallback partitioning when Qhull cannot build a Voronoi diagram (low counts, degeneracy)."""

    n = points.shape[0]
    if n == 0:
        return [], np.array([], dtype=float)
    order = np.argsort(points[:, 0], kind="mergesort")
    x_edges = np.linspace(0.0, aspect_ratio, n + 1)
    polygons: list[np.ndarray] = [np.zeros((0, 2)) for _ in range(n)]
    degrees = np.zeros(n, dtype=float)
    for idx, point_idx in enumerate(order):
        polygon = np.array(
            [
                [x_edges[idx], 0.0],
                [x_edges[idx + 1], 0.0],
                [x_edges[idx + 1], 1.0],
                [x_edges[idx], 1.0],
            ]
        )
        polygons[point_idx] = polygon
        if idx > 0:
            degrees[point_idx] += 1
        if idx < n - 1:
            degrees[point_idx] += 1
    return polygons, degrees


def _build_voronoi_metrics(polygons: list[np.ndarray], degrees: np.ndarray) -> VoronoiMetrics:
    if polygons:
        areas = np.array([_polygon_area(poly) for poly in polygons], dtype=float)
        area_mean = float(areas.mean()) if areas.size else 0.0
        area_cv = float(areas.std(ddof=0) / area_mean) if area_mean > 0 else 0.0
    else:
        area_mean = 0.0
        area_cv = 0.0
    if degrees.size:
        vertex_mean = float(degrees.mean())
        vertex_std = float(degrees.std(ddof=0))
    else:
        vertex_mean = 0.0
        vertex_std = 0.0
    return VoronoiMetrics(
        polygon_count=len(polygons),
        area_mean=area_mean,
        area_cv=area_cv,
        vertex_degree_mean=vertex_mean,
        vertex_degree_std=vertex_std,
    )


def _voronoi_finite_polygons_2d(
    vor: Voronoi, radius: float | None = None
) -> tuple[list[list[int]], np.ndarray]:
    """Reconstruct infinite Voronoi regions to finite polygons (adapted from SciPy docs)."""

    if vor.points.shape[1] != 2:
        raise ValueError("Voronoi input must be 2-D.")
    new_regions: list[list[int]] = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = float(np.ptp(vor.points, axis=0).max()) * 2

    all_ridges: dict[int, list[tuple[int, int, int]]] = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices, strict=True):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region_index in enumerate(vor.point_region):
        vertices = vor.regions[region_index]
        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges.get(p1, [])
        new_region = [v for v in vertices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue
            tangent = vor.points[p2] - vor.points[p1]
            tangent /= np.linalg.norm(tangent)
            normal = np.array([-tangent[1], tangent[0]])
            midpoint = (vor.points[p1] + vor.points[p2]) / 2
            direction = np.sign(np.dot(midpoint - center, normal)) * normal
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)
        new_regions.append(new_region)

    return new_regions, np.asarray(new_vertices)


def _clip_polygon_to_bounds(
    polygon: np.ndarray,
    bounds: tuple[float, float, float, float],
) -> np.ndarray:
    def _clip_halfspace(
        pts: list[list[float]],
        axis: int,
        value: float,
        keep_greater: bool,
    ) -> list[list[float]]:
        if not pts:
            return []
        result: list[list[float]] = []
        prev = pts[-1]
        prev_inside = (prev[axis] >= value) if keep_greater else (prev[axis] <= value)
        for curr in pts:
            curr_inside = (curr[axis] >= value) if keep_greater else (curr[axis] <= value)
            if curr_inside:
                if not prev_inside:
                    result.append(_halfspace_intersection(prev, curr, axis, value))
                result.append(curr)
            elif prev_inside:
                result.append(_halfspace_intersection(prev, curr, axis, value))
            prev = curr
            prev_inside = curr_inside
        return result

    def _halfspace_intersection(
        start: list[float] | np.ndarray,
        end: list[float] | np.ndarray,
        axis: int,
        value: float,
    ) -> list[float]:
        start_val = start[axis]
        end_val = end[axis]
        if start_val == end_val:
            intersection = end.copy() if isinstance(end, list) else end.tolist()
            intersection[axis] = value
            return intersection
        t = (value - start_val) / (end_val - start_val)
        intersection = [
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
        ]
        intersection[axis] = value
        return intersection

    pts = polygon.tolist()
    xmin, ymin, xmax, ymax = bounds
    for axis, value, keep_greater in (
        (0, xmin, True),
        (0, xmax, False),
        (1, ymin, True),
        (1, ymax, False),
    ):
        pts = _clip_halfspace(pts, axis, value, keep_greater)
        if not pts:
            break
    return np.asarray(pts, dtype=float)


def _polygon_area(polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))
