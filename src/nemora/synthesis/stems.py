"""Tree placement helpers for stand-level synthesis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .helpers import StandDBHSampler

__all__ = [
    "TreeAttributeConfig",
    "TreeAttributes",
    "TreePlacementConfig",
    "attach_tree_attributes",
    "place_trees",
    "place_trees_with_dbh",
]


@dataclass(slots=True)
class TreePlacementConfig:
    """Configuration for stochastic tree placement inside a polygon."""

    min_spacing: float = 0.0
    max_attempt_factor: int = 50


@dataclass(slots=True)
class TreeAttributeConfig:
    """Simple scalars used to derive placeholder tree attributes."""

    height_factor: float = 0.65  # metres per cm dbh (placeholder)
    crown_ratio: float = 0.4
    biomass_factor: float = 0.08  # tonnes per cm^2 dbh proxy
    bark_thickness_factor: float = 0.02  # cm bark per cm dbh


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
    x_min, y_min = np.min(poly, axis=0)
    x_max, y_max = np.max(poly, axis=0)
    points: list[tuple[float, float]] = []
    attempts = 0
    max_attempts = max(cfg.max_attempt_factor * count, count)
    while len(points) < count and attempts < max_attempts:
        x = float(rng.uniform(x_min, x_max))
        y = float(rng.uniform(y_min, y_max))
        attempts += 1
        if not _point_in_polygon(x, y, poly):
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

    cfg = config or TreeAttributeConfig()
    enriched: list[dict[str, object]] = []
    for record in records:
        dbh = _coerce_dbh(record.get("dbh"))
        basal_area = _basal_area_from_dbh(dbh)
        attrs = TreeAttributes(
            dbh_cm=dbh,
            height_m=max(dbh * cfg.height_factor, 0.0),
            crown_ratio=max(cfg.crown_ratio, 0.0),
            basal_area_m2=basal_area,
            biomass_tonnes=max(basal_area * cfg.biomass_factor, 0.0),
            bark_thickness_cm=max(dbh * cfg.bark_thickness_factor, 0.0),
        )
        enriched_record = dict(record)
        enriched_record["attributes"] = attrs
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
