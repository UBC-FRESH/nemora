"""Tree placement helpers for stand-level synthesis."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .helpers import StandDBHSampler

__all__ = [
    "TreePlacementMode",
    "TreeAttributeConfig",
    "DEFAULT_ATTRIBUTE_PROVENANCE",
    "load_attribute_config",
    "TreeAttributeConfig",
    "TreeAttributes",
    "TreePlacementConfig",
    "attach_tree_attributes",
    "place_trees",
    "place_trees_with_dbh",
]

TreePlacementMode = Literal["poisson", "stratified", "clustered"]

DEFAULT_ATTRIBUTE_PROVENANCE = "placeholder-v1"


@dataclass(slots=True)
class TreePlacementConfig:
    """Configuration for stochastic tree placement inside a polygon."""

    min_spacing: float = 0.0
    max_attempt_factor: int = 50
    mode: TreePlacementMode = "poisson"
    cluster_count: int | None = None
    cluster_spread: float = 0.05


@dataclass(slots=True)
class TreeAttributeConfig:
    """Simple scalars used to derive placeholder tree attributes."""

    height_a: float = 1.3
    height_b: float = 0.45
    crown_ratio: float = 0.45
    biomass_a: float = 0.05
    biomass_b: float = 2.35
    bark_thickness_a: float = 0.03
    bark_thickness_b: float = 1.05
    provenance: str = DEFAULT_ATTRIBUTE_PROVENANCE


@dataclass(slots=True)
class TreeAttributes:
    """Derived per-tree metrics."""

    dbh_cm: float
    height_m: float
    crown_ratio: float
    basal_area_m2: float
    biomass_tonnes: float
    bark_thickness_cm: float


def place_trees(
    polygon: np.ndarray,
    count: int,
    *,
    rng: np.random.Generator | None = None,
    config: TreePlacementConfig | None = None,
) -> np.ndarray:
    """Sample ``count`` points uniformly inside ``polygon`` using rejection sampling.

    ``min_spacing`` enforces a minimum Euclidean distance between accepted points. A
    ``ValueError`` is raised if placement cannot satisfy the spacing within
    ``max_attempt_factor * count`` attempts.
    """

    if count <= 0:
        return np.empty((0, 2), dtype=float)
    rng = rng or np.random.default_rng()
    cfg = config or TreePlacementConfig()
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError("Polygon must be an (n, 2) array with at least 3 vertices.")
    if cfg.mode == "stratified":
        points = _place_stratified(poly, count, cfg, rng)
    elif cfg.mode == "clustered":
        points = _place_clustered(poly, count, cfg, rng)
    else:
        points = _place_poisson(poly, count, cfg, rng)
    return points


def place_trees_with_dbh(
    polygon: np.ndarray,
    sampler: StandDBHSampler,
    *,
    count: int | None = None,
    rng: np.random.Generator | None = None,
    config: TreePlacementConfig | None = None,
) -> list[dict[str, object]]:
    """Draw DBH values and pair them with spatial coordinates inside ``polygon``."""

    rng = rng or np.random.default_rng()
    draws = sampler.draw(sample_size=count, rng=rng)
    placement = place_trees(
        polygon,
        draws.shape[0],
        rng=rng,
        config=config,
    )
    records: list[dict[str, object]] = []
    for coords, dbh in zip(placement, draws, strict=False):
        records.append(
            {
                "stand_id": sampler.assignment.stand_id,
                "bootstrap_id": sampler.bootstrap_id,
                "sampler_type": sampler.sampler_type,
                "x": float(coords[0]),
                "y": float(coords[1]),
                "dbh": float(dbh),
            }
        )
    return records


def attach_tree_attributes(
    records: list[dict[str, object]],
    *,
    config: TreeAttributeConfig | None = None,
) -> list[dict[str, object]]:
    """Return records augmented with derived tree attributes."""

    cfg = config or load_attribute_config()
    enriched: list[dict[str, object]] = []
    for record in records:
        dbh = _coerce_dbh(record.get("dbh"))
        basal_area = _basal_area_from_dbh(dbh)
        attrs = TreeAttributes(
            dbh_cm=dbh,
            height_m=max(cfg.height_a * (dbh**cfg.height_b), 0.0),
            crown_ratio=min(max(cfg.crown_ratio, 0.0), 1.0),
            basal_area_m2=basal_area,
            biomass_tonnes=max(cfg.biomass_a * (dbh**cfg.biomass_b), 0.0),
            bark_thickness_cm=max(cfg.bark_thickness_a * (dbh**cfg.bark_thickness_b), 0.0),
        )
        enriched_record = dict(record)
        enriched_record["attributes"] = attrs
        enriched_record["attributes_provenance"] = {
            "height_a": cfg.height_a,
            "height_b": cfg.height_b,
            "biomass_a": cfg.biomass_a,
            "biomass_b": cfg.biomass_b,
            "bark_thickness_a": cfg.bark_thickness_a,
            "bark_thickness_b": cfg.bark_thickness_b,
            "crown_ratio": cfg.crown_ratio,
            "provenance": cfg.provenance,
        }
        enriched.append(enriched_record)
    return enriched


def _point_in_polygon(x: float, y: float, polygon: np.ndarray) -> bool:
    """Ray-casting algorithm for point-in-polygon."""

    inside = False
    x_coords = polygon[:, 0]
    y_coords = polygon[:, 1]
    n = polygon.shape[0]
    j = n - 1
    for i in range(n):
        xi = x_coords[i]
        yi = y_coords[i]
        xj = x_coords[j]
        yj = y_coords[j]
        intersects = (yi > y) != (yj > y) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def _is_spaced(
    candidate: tuple[float, float], points: list[tuple[float, float]], min_dist: float
) -> bool:
    """Return True if ``candidate`` is at least ``min_dist`` away from all points."""

    cx, cy = candidate
    min_dist_sq = min_dist * min_dist
    for px, py in points:
        dx = cx - px
        dy = cy - py
        if dx * dx + dy * dy < min_dist_sq:
            return False
    return True


def _basal_area_from_dbh(dbh_cm: float) -> float:
    """Compute basal area (m2) from DBH (cm)."""

    radius_m = (dbh_cm / 100.0) / 2.0
    return float(np.pi * radius_m * radius_m)


def _coerce_dbh(value: object | None) -> float:
    if isinstance(value, int | float | np.floating):
        return float(value)
    if isinstance(value, str | bytes | bytearray):
        try:
            return float(value)
        except ValueError:
            return 0.0
    return 0.0


def load_attribute_config(path: Path | None = None) -> TreeAttributeConfig:
    """Load a tree attribute config from JSON or fall back to defaults.

    JSON schema keys mirror ``TreeAttributeConfig`` fields and may include
    ``provenance`` for versioning (defaults to ``placeholder-v1``).
    An environment variable ``NEMORA_TREE_ATTRIBUTE_CONFIG`` can point to a
    JSON file to override defaults across CLI runs.
    """

    config_path = path
    if config_path is None:
        env_path = os.environ.get("NEMORA_TREE_ATTRIBUTE_CONFIG")
        if env_path:
            config_path = Path(env_path)
    if config_path is None:
        return TreeAttributeConfig()
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    kwargs: dict[str, object] = {
        k: payload.get(k)
        for k in (
            "height_a",
            "height_b",
            "crown_ratio",
            "biomass_a",
            "biomass_b",
            "bark_thickness_a",
            "bark_thickness_b",
            "provenance",
        )
    }
    height_a_raw = kwargs.get("height_a")
    height_b_raw = kwargs.get("height_b")
    crown_ratio_raw = kwargs.get("crown_ratio")
    biomass_a_raw = kwargs.get("biomass_a")
    biomass_b_raw = kwargs.get("biomass_b")
    bark_a_raw = kwargs.get("bark_thickness_a")
    bark_b_raw = kwargs.get("bark_thickness_b")
    provenance_raw = kwargs.get("provenance")
    return TreeAttributeConfig(
        height_a=float(height_a_raw) if isinstance(height_a_raw, int | float | str) else 1.3,
        height_b=float(height_b_raw) if isinstance(height_b_raw, int | float | str) else 0.45,
        crown_ratio=float(crown_ratio_raw)
        if isinstance(crown_ratio_raw, int | float | str)
        else 0.45,
        biomass_a=float(biomass_a_raw) if isinstance(biomass_a_raw, int | float | str) else 0.05,
        biomass_b=float(biomass_b_raw) if isinstance(biomass_b_raw, int | float | str) else 2.35,
        bark_thickness_a=float(bark_a_raw) if isinstance(bark_a_raw, int | float | str) else 0.03,
        bark_thickness_b=float(bark_b_raw) if isinstance(bark_b_raw, int | float | str) else 1.05,
        provenance=str(provenance_raw)
        if provenance_raw is not None
        else DEFAULT_ATTRIBUTE_PROVENANCE,
    )


def _place_poisson(
    polygon: np.ndarray,
    count: int,
    cfg: TreePlacementConfig,
    rng: np.random.Generator,
    *,
    seed_points: list[tuple[float, float]] | None = None,
) -> np.ndarray:
    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)
    points: list[tuple[float, float]] = list(seed_points or [])
    attempts = 0
    max_attempts = max(cfg.max_attempt_factor * count, count)
    while len(points) < count and attempts < max_attempts:
        x = float(rng.uniform(x_min, x_max))
        y = float(rng.uniform(y_min, y_max))
        attempts += 1
        if not _point_in_polygon(x, y, polygon):
            continue
        if cfg.min_spacing > 0 and points:
            if not _is_spaced((x, y), points, cfg.min_spacing):
                continue
        points.append((x, y))
    if len(points) < count:
        raise ValueError(
            f"Unable to place {count} trees with min_spacing={cfg.min_spacing} "
            f"after {attempts} attempts."
        )
    return np.asarray(points, dtype=float)


def _place_stratified(
    polygon: np.ndarray,
    count: int,
    cfg: TreePlacementConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    x_min, y_min = np.min(polygon, axis=0)
    x_max, y_max = np.max(polygon, axis=0)
    grid_size = max(int(np.ceil(np.sqrt(count))), 1)
    step_x = (x_max - x_min) / grid_size
    step_y = (y_max - y_min) / grid_size
    points: list[tuple[float, float]] = []
    for i in range(grid_size):
        for j in range(grid_size):
            if len(points) >= count:
                break
            x = x_min + (i + 0.5) * step_x
            y = y_min + (j + 0.5) * step_y
            if not _point_in_polygon(x, y, polygon):
                continue
            if cfg.min_spacing > 0 and points:
                if not _is_spaced((x, y), points, cfg.min_spacing):
                    continue
            points.append((x, y))
    if len(points) < count:
        remaining = count - len(points)
        poisson_points = _place_poisson(
            polygon,
            remaining,
            cfg,
            rng,
            seed_points=points,
        )
        return poisson_points
    return np.asarray(points[:count], dtype=float)


def _place_clustered(
    polygon: np.ndarray,
    count: int,
    cfg: TreePlacementConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    extent = np.ptp(polygon, axis=0)
    spread_scale = cfg.cluster_spread * float(np.min(extent) or 1.0)
    cluster_count = cfg.cluster_count or max(1, int(np.ceil(np.sqrt(count) / 1.5)))
    centers: list[tuple[float, float]] = []
    while len(centers) < cluster_count:
        candidate = _place_poisson(polygon, 1, cfg, rng)
        centers.append((float(candidate[0, 0]), float(candidate[0, 1])))
    points: list[tuple[float, float]] = []
    attempts = 0
    max_attempts = max(cfg.max_attempt_factor * count, count)
    while len(points) < count and attempts < max_attempts:
        cx, cy = centers[int(rng.integers(0, cluster_count))]
        x = float(rng.normal(cx, spread_scale))
        y = float(rng.normal(cy, spread_scale))
        attempts += 1
        if not _point_in_polygon(x, y, polygon):
            continue
        if cfg.min_spacing > 0 and points:
            if not _is_spaced((x, y), points, cfg.min_spacing):
                continue
        points.append((x, y))
    if len(points) < count:
        remaining = count - len(points)
        poisson_points = _place_poisson(
            polygon,
            remaining,
            cfg,
            rng,
            seed_points=points,
        )
        return poisson_points
    return np.asarray(points, dtype=float)
