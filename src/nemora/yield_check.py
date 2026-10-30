"""Independent inventory-volume checks for FHOPS handoff scenarios.

The estimator is deliberately simple and transparent. It is not a taper model
and must not be treated as an operational yield estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

__all__ = [
    "InventoryVolumeEstimate",
    "compare_volume_estimates",
    "estimate_inventory_volume",
    "estimate_tree_volume_m3",
]

DEFAULT_MERCHANTABLE_HEIGHT_RATIO = 0.70
DEFAULT_FORM_FACTOR = 0.50


@dataclass(frozen=True, slots=True)
class InventoryVolumeEstimate:
    """Independent volume estimate for a tree inventory."""

    tree_count: int
    total_volume_m3: float
    mean_volume_m3: float
    merchantable_height_ratio: float
    form_factor: float
    formula: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary."""
        return {
            "tree_count": self.tree_count,
            "total_volume_m3": self.total_volume_m3,
            "mean_volume_m3": self.mean_volume_m3,
            "merchantable_height_ratio": self.merchantable_height_ratio,
            "form_factor": self.form_factor,
            "formula": self.formula,
        }


def estimate_tree_volume_m3(
    dbh_cm: float,
    height_m: float,
    *,
    merchantable_height_m: float | None = None,
    merchantable_height_ratio: float = DEFAULT_MERCHANTABLE_HEIGHT_RATIO,
    form_factor: float = DEFAULT_FORM_FACTOR,
) -> float:
    """Estimate one tree's volume with a paraboloid-style formula.

    ``V = pi/4 * D^2 * H_m * form_factor`` with D in metres. The default
    merchantable height is 70% of total height and the default form factor is
    0.5. This is an independent check, not a taper model.

    Raises
    ------
    ValueError
        If inputs are non-finite, non-positive, or outside ``(0, 1]``.
    """
    if not math.isfinite(dbh_cm) or dbh_cm <= 0.0:
        raise ValueError("dbh_cm must be finite and positive")
    if not math.isfinite(height_m) or height_m <= 0.0:
        raise ValueError("height_m must be finite and positive")
    if not 0.0 < merchantable_height_ratio <= 1.0:
        raise ValueError("merchantable_height_ratio must be in (0, 1]")
    if not 0.0 < form_factor <= 1.0:
        raise ValueError("form_factor must be in (0, 1]")
    if merchantable_height_m is not None:
        if not math.isfinite(merchantable_height_m) or merchantable_height_m <= 0.0:
            raise ValueError("merchantable_height_m must be finite and positive")
        if merchantable_height_m > height_m + 1e-9:
            raise ValueError("merchantable_height_m cannot exceed height_m")
        merchantable = merchantable_height_m
    else:
        merchantable = height_m * merchantable_height_ratio
    diameter_m = dbh_cm / 100.0
    return float((math.pi / 4.0) * diameter_m**2 * merchantable * form_factor)


def estimate_inventory_volume(
    trees: pd.DataFrame,
    *,
    merchantable_height_ratio: float = DEFAULT_MERCHANTABLE_HEIGHT_RATIO,
    form_factor: float = DEFAULT_FORM_FACTOR,
) -> InventoryVolumeEstimate:
    """Estimate total volume for a tree table with ``dbh_cm`` and ``height_m``."""
    required = ("dbh_cm", "height_m")
    missing = [column for column in required if column not in trees.columns]
    if missing:
        raise KeyError("trees is missing required columns: " + ", ".join(missing))
    if trees.empty:
        raise ValueError("cannot estimate an empty inventory")
    columns = list(trees.columns)
    dbh_index = columns.index("dbh_cm")
    height_index = columns.index("height_m")
    merchantable_index = (
        columns.index("merchantable_height_m") if "merchantable_height_m" in columns else None
    )
    volumes: list[float] = []
    for values in trees.itertuples(index=False, name=None):
        merchantable: float | None = None
        if merchantable_index is not None:
            raw = values[merchantable_index]
            if raw is not None and not pd.isna(raw):
                merchantable = float(raw)
        volumes.append(
            estimate_tree_volume_m3(
                float(values[dbh_index]),
                float(values[height_index]),
                merchantable_height_m=merchantable,
                merchantable_height_ratio=merchantable_height_ratio,
                form_factor=form_factor,
            )
        )
    total = float(sum(volumes))
    return InventoryVolumeEstimate(
        tree_count=len(volumes),
        total_volume_m3=total,
        mean_volume_m3=total / len(volumes),
        merchantable_height_ratio=merchantable_height_ratio,
        form_factor=form_factor,
        formula="V = pi/4 * D^2 * H_m * form_factor; H_m defaults to 0.70 * total height",
    )


def compare_volume_estimates(
    estimate_m3: float,
    *,
    registered_work_required_m3: float | None = None,
    fhops_processed_m3: float | None = None,
) -> dict[str, Any]:
    """Compare independent, registered, and FHOPS-processed volumes."""
    result: dict[str, Any] = {"estimate_m3": float(estimate_m3)}
    if registered_work_required_m3 is not None:
        result["registered_work_required_m3"] = float(registered_work_required_m3)
        result["estimate_vs_registered_relative"] = _relative(
            estimate_m3, registered_work_required_m3
        )
    if fhops_processed_m3 is not None:
        result["fhops_processed_m3"] = float(fhops_processed_m3)
        result["estimate_vs_processed_relative"] = _relative(estimate_m3, fhops_processed_m3)
    if registered_work_required_m3 is not None and fhops_processed_m3 is not None:
        result["registered_vs_processed_relative"] = _relative(
            registered_work_required_m3, fhops_processed_m3
        )
    return result


def _relative(value: float, reference: float) -> float | None:
    if reference == 0.0:
        return None
    return float((value - reference) / reference)
