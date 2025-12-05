"""Seed-point configuration helpers for Voronoi-based landscape tiling.

The CJFR rlandscape paper describes a mixture of four point processes and two
editing knobs (hole/merge fractions) that collectively control the number of
management units, polygon area variation, and vertex-degree statistics. This
module captures those inputs so downstream synthesis code (Phase 1 of the plan)
can generate repeatable seed sets and share the configuration with docs/tests.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.spatial import QhullError, Voronoi

__all__ = [
    "ClusterConfig",
    "InhibitionConfig",
    "LatticeConfig",
    "PointProcessMix",
    "VoronoiEditConfig",
    "MaskGeometry",
    "MaskMode",
    "RasterMask",
    "RasterMode",
    "VoronoiSeedConfig",
    "SeedLayoutMode",
    "SeedLayoutConfig",
    "VoronoiMetrics",
    "VoronoiSeedResult",
    "load_mask_from_geojson",
    "load_polygons_from_geojson",
    "load_raster_mask",
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


class MaskMode(str, Enum):
    """Mask behaviour for polygon overlays."""

    CLIP = "clip"
    EXCLUDE = "exclude"


class RasterMode(str, Enum):
    """Raster modifier behaviour."""

    KEEP = "keep"
    EXCLUDE = "exclude"


@dataclass(slots=True)
class MaskGeometry:
    """Optional clipping geometry for Voronoi polygons."""

    polygons: list[np.ndarray]
    name: str | None = None
    mode: MaskMode = MaskMode.CLIP


@dataclass(slots=True)
class RasterMask:
    """Raster-based modifier that keeps or excludes polygons."""

    values: np.ndarray
    threshold: float = 0.0
    mode: RasterMode = RasterMode.KEEP
    name: str | None = None


class SeedLayoutMode(str, Enum):
    """Seed placement strategies available to callers/CLI."""

    RANDOM = "random"
    HEX = "hex"
    IMPORTED = "imported"
    GEOJSON = "geojson"


@dataclass(slots=True)
class SeedLayoutConfig:
    """Deterministic layout configuration (hex grid or imported points)."""

    mode: SeedLayoutMode = SeedLayoutMode.RANDOM
    points: np.ndarray | None = None
    source: str | None = None
    geojson_polygons: list[np.ndarray] | None = None


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
    mask: MaskGeometry | None = None
    mask_overlays: list[MaskGeometry] = field(default_factory=list)
    raster_masks: list[RasterMask] = field(default_factory=list)
    layout: SeedLayoutConfig = field(default_factory=SeedLayoutConfig)
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
            "mask": self._mask_metadata(),
            "rasters": self._raster_metadata(),
            "layout": {
                "mode": self.config.layout.mode.value,
                "points_provided": (
                    None
                    if self.config.layout.points is None
                    else int(self.config.layout.points.shape[0])
                ),
                "source": self.config.layout.source,
                "geojson_features": (
                    None
                    if not self.config.layout.geojson_polygons
                    else len(self.config.layout.geojson_polygons)
                ),
            },
        }

    def _mask_metadata(self) -> dict[str, object] | None:
        if self.config.mask is None and not self.config.mask_overlays:
            return None
        primary = None
        if self.config.mask is not None:
            primary = {
                "polygons": len(self.config.mask.polygons),
                "name": self.config.mask.name,
                "mode": self.config.mask.mode.value,
            }
        overlays = [
            {
                "polygons": len(mask.polygons),
                "name": mask.name,
                "mode": mask.mode.value,
            }
            for mask in self.config.mask_overlays
        ]
        return {"primary": primary, "overlays": overlays}

    def _raster_metadata(self) -> list[dict[str, object]] | None:
        if not self.config.raster_masks:
            return None
        return [
            {
                "name": raster.name,
                "threshold": raster.threshold,
                "mode": raster.mode.value,
                "shape": list(raster.values.shape),
            }
            for raster in self.config.raster_masks
        ]


def generate_seed_points(config: VoronoiSeedConfig) -> VoronoiSeedResult:
    """Generate provisional seed coordinates for Phase 1 prototypes."""

    rng = config.rng or np.random.default_rng()
    target_count = config.count
    hole_target = _resolve_edit_count(config.edit.hole_fraction, target_count)
    merge_target = _resolve_edit_count(config.edit.merge_fraction, target_count)
    seed_count = target_count + hole_target + merge_target
    base_points, process_counts = _generate_base_points(seed_count, config, rng)
    (
        edited_points,
        hole_points,
        merge_pairs,
    ) = _apply_editing(base_points, rng, hole_target, merge_target)
    if edited_points.shape[0] != target_count:
        raise RuntimeError(
            "Editing pipeline failed to honour target count "
            f"(expected {target_count}, found {edited_points.shape[0]})."
        )
    polygons, degrees = _derive_voronoi_geometry(edited_points, config)
    polygons = _apply_mask_modifiers(polygons, config, edited_points)
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


def _generate_base_points(
    seed_count: int,
    config: VoronoiSeedConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
    layout_mode = config.layout.mode
    if layout_mode == SeedLayoutMode.RANDOM:
        return _generate_random_mix(seed_count, config, rng)
    if layout_mode == SeedLayoutMode.HEX:
        hex_points = _generate_hex_packed(seed_count, config)
        counts = {
            "uniform": 0,
            "cluster": 0,
            "inhibition": 0,
            "lattice": 0,
            "layout_hex": hex_points.shape[0],
        }
        return hex_points, counts
    if layout_mode == SeedLayoutMode.IMPORTED:
        imported = _prepare_imported_layout(seed_count, config)
        counts = {
            "uniform": 0,
            "cluster": 0,
            "inhibition": 0,
            "lattice": 0,
            "layout_imported": imported.shape[0],
        }
        return imported, counts
    if layout_mode == SeedLayoutMode.GEOJSON:
        geojson_points = _generate_geojson_layout(seed_count, config)
        counts = {
            "uniform": 0,
            "cluster": 0,
            "inhibition": 0,
            "lattice": 0,
            "layout_geojson": geojson_points.shape[0],
        }
        return geojson_points, counts
    raise ValueError(f"Unsupported layout mode: {layout_mode}")


def _generate_random_mix(
    seed_count: int,
    config: VoronoiSeedConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, int]]:
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
        fallback = _generate_uniform(seed_count, config, rng)
        process_counts = {"uniform": fallback.shape[0], "cluster": 0, "inhibition": 0, "lattice": 0}
        return fallback, process_counts
    stacked = np.vstack(points)
    return stacked, process_counts


def _generate_hex_packed(
    seed_count: int,
    config: VoronoiSeedConfig,
) -> np.ndarray:
    if seed_count <= 0:
        return np.zeros((0, 2))
    width = config.aspect_ratio
    area = width * 1.0
    spacing = math.sqrt((2.0 / math.sqrt(3.0)) * area / seed_count)
    dx = spacing
    dy = spacing * math.sqrt(3.0) / 2.0
    attempt = 0
    points: np.ndarray | None = None
    while attempt < 8:
        points = _hex_grid_points(width, 1.0, dx, dy)
        if points.shape[0] >= seed_count:
            break
        dx *= 0.9
        dy *= 0.9
        attempt += 1
    if points is None or points.shape[0] < seed_count:
        raise RuntimeError(
            "Hex layout could not generate enough seeds; consider lowering count "
            "or adjusting aspect ratio."
        )
    return points[:seed_count]


def _hex_grid_points(width: float, height: float, dx: float, dy: float) -> np.ndarray:
    if dx <= 0 or dy <= 0:
        raise ValueError("Hex grid spacing must be positive.")
    points: list[list[float]] = []
    row = 0
    y = 0.0
    while y <= height + dy:
        offset = 0.0 if row % 2 == 0 else dx / 2.0
        col = 0
        x = offset
        while x <= width + 1e-9:
            if 0.0 <= x <= width and 0.0 <= y <= height:
                points.append([x, y])
            col += 1
            x = offset + col * dx
        row += 1
        y = row * dy
    return np.asarray(points, dtype=float)


def _prepare_imported_layout(
    seed_count: int,
    config: VoronoiSeedConfig,
) -> np.ndarray:
    points = config.layout.points
    if points is None:
        raise ValueError("Imported layout requires explicit coordinates.")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Layout point array must be shaped (n, 2).")
    if points.shape[0] < seed_count:
        raise ValueError(
            f"Imported layout provides {points.shape[0]} points but {seed_count} are required."
        )
    subset = np.array(points[:seed_count], dtype=float)
    if np.any(subset[:, 0] < -1e-9) or np.any(subset[:, 0] > config.aspect_ratio + 1e-9):
        raise ValueError("Imported layout contains x coordinates outside the bounding box.")
    if np.any(subset[:, 1] < -1e-9) or np.any(subset[:, 1] > 1.0 + 1e-9):
        raise ValueError("Imported layout contains y coordinates outside the bounding box.")
    return subset


def _generate_geojson_layout(
    seed_count: int,
    config: VoronoiSeedConfig,
) -> np.ndarray:
    polygons = config.layout.geojson_polygons
    if not polygons:
        raise ValueError("GeoJSON layout mode requires polygons via layout.geojson_polygons.")
    centroids = np.array([_polygon_centroid(poly) for poly in polygons], dtype=float)
    if centroids.shape[0] < seed_count:
        repeats = int(math.ceil(seed_count / centroids.shape[0]))
        centroids = np.vstack([centroids] * repeats)
    bounded = np.clip(centroids[:seed_count], [0.0, 0.0], [config.aspect_ratio, 1.0])
    return bounded


def _polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    if polygon.shape[0] < 3:
        return polygon.mean(axis=0) if polygon.size else np.zeros(2)
    x = polygon[:, 0]
    y = polygon[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-12:
        return polygon.mean(axis=0)
    factor = cross / (6.0 * area)
    cx = np.sum((x + np.roll(x, -1)) * factor)
    cy = np.sum((y + np.roll(y, -1)) * factor)
    return np.array([cx, cy], dtype=float)


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


def _apply_mask_modifiers(
    polygons: list[np.ndarray],
    config: VoronoiSeedConfig,
    points: np.ndarray,
) -> list[np.ndarray]:
    masks: list[MaskGeometry] = []
    if config.mask is not None:
        masks.append(config.mask)
    if config.mask_overlays:
        masks.extend(config.mask_overlays)
    working = polygons
    for mask in masks:
        working = _apply_single_mask(working, mask, points)
    if config.raster_masks:
        working = _apply_raster_masks(working, config.raster_masks, points, config.aspect_ratio)
    return working


def _apply_single_mask(
    polygons: list[np.ndarray],
    mask: MaskGeometry,
    points: np.ndarray,
) -> list[np.ndarray]:
    if mask.mode == MaskMode.EXCLUDE:
        return _exclude_polygons_with_mask(polygons, mask, points)
    return _clip_polygons_to_mask(polygons, mask, points)


def _exclude_polygons_with_mask(
    polygons: list[np.ndarray],
    mask: MaskGeometry,
    points: np.ndarray,
) -> list[np.ndarray]:
    if not mask.polygons:
        return polygons
    output: list[np.ndarray] = []
    for idx, polygon in enumerate(polygons):
        if polygon.size == 0:
            output.append(polygon)
            continue
        point = points[idx]
        inside = any(_point_in_polygon(point, poly) for poly in mask.polygons)
        output.append(np.zeros((0, 2)) if inside else polygon)
    return output


def _apply_raster_masks(
    polygons: list[np.ndarray],
    rasters: list[RasterMask],
    points: np.ndarray,
    aspect_ratio: float,
) -> list[np.ndarray]:
    if not rasters:
        return polygons
    working = list(polygons)
    for idx, polygon in enumerate(working):
        if polygon.size == 0:
            continue
        sample_point = points[idx]
        for raster in rasters:
            value = _sample_raster_value(raster.values, sample_point, aspect_ratio)
            if raster.mode == RasterMode.KEEP and value < raster.threshold:
                working[idx] = np.zeros((0, 2))
                break
            if raster.mode == RasterMode.EXCLUDE and value >= raster.threshold:
                working[idx] = np.zeros((0, 2))
                break
    return working


def _sample_raster_value(
    values: np.ndarray,
    point: np.ndarray,
    aspect_ratio: float,
) -> float:
    rows, cols = values.shape
    if rows == 0 or cols == 0:
        return 0.0
    x = float(np.clip(point[0], 0.0, aspect_ratio if aspect_ratio > 0 else 1.0))
    y = float(np.clip(point[1], 0.0, 1.0))
    if aspect_ratio <= 0:
        aspect_ratio = 1.0
    rel_x = x / aspect_ratio
    rel_y = 1.0 - y  # raster row 0 at top
    col = min(cols - 1, max(0, int(round(rel_x * (cols - 1)))))
    row = min(rows - 1, max(0, int(round(rel_y * (rows - 1)))))
    return float(values[row, col])


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
    return abs(_signed_polygon_area(polygon))


def _signed_polygon_area(polygon: np.ndarray) -> float:
    if polygon.shape[0] < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def load_mask_from_geojson(
    path: Path,
    *,
    name: str | None = None,
    mode: MaskMode = MaskMode.CLIP,
) -> MaskGeometry:
    """Load a simple polygon or multipolygon mask from GeoJSON."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    polygons = _extract_polygons_from_geojson(payload)
    if not polygons:
        raise ValueError(f"No Polygon/MultiPolygon geometries found in {path}")
    mask_name = name or payload.get("name") or path.stem
    return MaskGeometry(polygons=polygons, name=mask_name, mode=mode)


def load_polygons_from_geojson(path: Path) -> list[np.ndarray]:
    """Return polygon coordinates from a GeoJSON file."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    polygons = _extract_polygons_from_geojson(payload)
    if not polygons:
        raise ValueError(f"No Polygon/MultiPolygon geometries found in {path}")
    return polygons


def load_raster_mask(
    path: Path,
    *,
    threshold: float = 0.0,
    mode: RasterMode = RasterMode.KEEP,
    name: str | None = None,
) -> RasterMask:
    """Load a raster (NumPy .npy/.npz or CSV/txt) used to keep/exclude polygons."""

    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        loaded = np.load(path)
        values = loaded if isinstance(loaded, np.ndarray) else loaded[list(loaded.files)[0]]
    elif suffix in {".csv", ".txt"}:
        values = np.loadtxt(path, delimiter="," if suffix == ".csv" else None)
    else:
        raise ValueError(f"Unsupported raster format for {path}; use .npy, .npz, .csv, or .txt.")
    if values.ndim != 2:
        raise ValueError(f"Raster mask {path} must be two-dimensional.")
    raster_name = name or path.stem
    return RasterMask(
        values=np.asarray(values, dtype=float), threshold=threshold, mode=mode, name=raster_name
    )


def _extract_polygons_from_geojson(payload: dict[str, object]) -> list[np.ndarray]:
    geo_type = payload.get("type")
    if geo_type == "FeatureCollection":
        feature_polygons: list[np.ndarray] = []
        features = payload.get("features", [])
        if isinstance(features, list):
            for feature in features:
                if isinstance(feature, dict):
                    geometry = feature.get("geometry", {})
                    if isinstance(geometry, dict):
                        feature_polygons.extend(_extract_polygons_from_geojson(geometry))
        return feature_polygons
    if geo_type == "Feature":
        geometry = payload.get("geometry", {})
        if isinstance(geometry, dict):
            return _extract_polygons_from_geojson(geometry)
        return []
    coords = payload.get("coordinates")
    polygons: list[np.ndarray] = []
    if geo_type == "Polygon":
        if isinstance(coords, list) and coords and isinstance(coords[0], list):
            polygon = _coords_to_array(coords[0])
            if polygon.size >= 6:
                polygons.append(polygon)
    elif geo_type == "MultiPolygon":
        if isinstance(coords, list):
            for poly in coords:
                if isinstance(poly, list) and poly and isinstance(poly[0], list):
                    polygon = _coords_to_array(poly[0])
                    if polygon.size >= 6:
                        polygons.append(polygon)
    return polygons


def _coords_to_array(ring: list[list[float]]) -> np.ndarray:
    if not ring:
        return np.zeros((0, 2))
    points = np.array(ring, dtype=float)
    if points.shape[0] > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    return _ensure_ccw(points)


def _ensure_ccw(points: np.ndarray) -> np.ndarray:
    if _signed_polygon_area(points) < 0:
        return np.flip(points, axis=0)
    return points


def _clip_polygons_to_mask(
    polygons: list[np.ndarray],
    mask: MaskGeometry,
    points: np.ndarray,
) -> list[np.ndarray]:
    if not mask.polygons:
        return polygons
    clipped: list[np.ndarray] = []
    for idx, polygon in enumerate(polygons):
        mask_polygon = _select_mask_polygon(mask.polygons, points[idx])
        if mask_polygon is None or mask_polygon.size < 6:
            clipped.append(np.zeros((0, 2)))
            continue
        clipped_polygon = _sutherland_hodgman_clip(polygon, mask_polygon)
        clipped.append(clipped_polygon)
    return clipped


def _select_mask_polygon(polygons: list[np.ndarray], point: np.ndarray) -> np.ndarray | None:
    for poly in polygons:
        if _point_in_polygon(point, poly):
            return poly
    return polygons[0] if polygons else None


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    if polygon.shape[0] < 3:
        return False
    x, y = point
    inside = False
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    j = polygon.shape[0] - 1
    for i in range(polygon.shape[0]):
        xi = x_coords[i]
        yi = y_coords[i]
        xj = x_coords[j]
        yj = y_coords[j]
        intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside


def _sutherland_hodgman_clip(subject: np.ndarray, clip_polygon: np.ndarray) -> np.ndarray:
    if subject.size == 0 or clip_polygon.size == 0:
        return np.zeros((0, 2))
    output = subject.tolist()
    clip_points = clip_polygon.tolist()
    clip_points.append(clip_points[0])
    for i in range(len(clip_points) - 1):
        input_list = output
        output = []
        if not input_list:
            break
        edge_start = clip_points[i]
        edge_end = clip_points[i + 1]
        prev_point = input_list[-1]
        prev_inside = _is_inside(prev_point, edge_start, edge_end)
        for curr_point in input_list:
            curr_inside = _is_inside(curr_point, edge_start, edge_end)
            if curr_inside:
                if not prev_inside:
                    output.append(
                        _compute_intersection(prev_point, curr_point, edge_start, edge_end)
                    )
                output.append(curr_point)
            elif prev_inside:
                output.append(_compute_intersection(prev_point, curr_point, edge_start, edge_end))
            prev_point = curr_point
            prev_inside = curr_inside
    if not output:
        return np.zeros((0, 2))
    return np.asarray(output, dtype=float)


def _is_inside(point: list[float], edge_start: list[float], edge_end: list[float]) -> bool:
    cross = (edge_end[0] - edge_start[0]) * (point[1] - edge_start[1]) - (
        edge_end[1] - edge_start[1]
    ) * (point[0] - edge_start[0])
    return cross >= 0


def _compute_intersection(
    start: list[float],
    end: list[float],
    edge_start: list[float],
    edge_end: list[float],
) -> list[float]:
    x1, y1 = start
    x2, y2 = end
    x3, y3 = edge_start
    x4, y4 = edge_end
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return end
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
    return [px, py]
