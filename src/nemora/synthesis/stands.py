"""Stand-attribute templates inspired by the FLG workflow."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

__all__ = [
    "StandAttributeTemplate",
    "StandAttributeSample",
    "build_templates",
    "sample_stand_attributes",
    "load_templates_from_json",
]


@dataclass(slots=True)
class StandAttributeTemplate:
    """Reusable patch recipe derived from vegetation/age-class tables."""

    vegetation_type: str
    age_class_cdf: Sequence[tuple[str, float]]
    patch_weibull: tuple[float, float, float]

    def sample_age_class(self, rng: np.random.Generator | None = None) -> str:
        """Sample an age-class label using the stored cumulative probabilities."""

        rng = rng or np.random.default_rng()
        draw = float(rng.random())
        for label, cumulative in self.age_class_cdf:
            if draw <= cumulative:
                return label
        return self.age_class_cdf[-1][0]

    def sample_patch_size(
        self,
        rng: np.random.Generator | None = None,
        size: int = 1,
    ) -> np.ndarray:
        """Sample patch areas via a shifted/scaled Weibull distribution."""

        rng = rng or np.random.default_rng()
        shape, scale, shift = self.patch_weibull
        samples = rng.weibull(shape, size=size) * scale + shift
        return samples


@dataclass(slots=True)
class StandAttributeSample:
    """Concrete stand attribute sampled from a template."""

    vegetation_type: str
    age_class: str
    area: float


def sample_stand_attributes(
    templates: Sequence[StandAttributeTemplate],
    *,
    total_area: float,
    rng: np.random.Generator | None = None,
    weights: Sequence[float] | None = None,
) -> list[StandAttributeSample]:
    """Fill ``total_area`` with sampled stand attributes."""

    if total_area <= 0:
        raise ValueError("total_area must be positive.")
    if not templates:
        raise ValueError("At least one template is required.")
    if weights is not None and len(weights) != len(templates):
        raise ValueError("Length of weights must match the length of templates.")
    rng = rng or np.random.default_rng()
    if weights is not None:
        probs = np.asarray(weights, dtype=float)
        if np.any(probs < 0):
            raise ValueError("Weights must be non-negative.")
        if probs.sum() == 0:
            raise ValueError("At least one weight must be positive.")
        probs = probs / probs.sum()
    else:
        probs = None
    remaining = float(total_area)
    samples: list[StandAttributeSample] = []
    iterations = 0
    max_iterations = 10000
    while remaining > 1e-6 and iterations < max_iterations:
        if probs is None:
            index = int(rng.integers(0, len(templates)))
        else:
            index = int(rng.choice(len(templates), p=probs))
        template = templates[index]
        patch_area = float(template.sample_patch_size(rng, size=1)[0])
        if patch_area <= 0:
            iterations += 1
            continue
        age_class = template.sample_age_class(rng)
        area = min(patch_area, remaining)
        samples.append(
            StandAttributeSample(
                vegetation_type=template.vegetation_type,
                age_class=age_class,
                area=area,
            )
        )
        remaining -= area
        iterations += 1
    return samples


def build_templates(
    records: Iterable[Mapping[str, object]],
) -> list[StandAttributeTemplate]:
    """Convert raw vegetation-type tables into reusable templates.

    Each *record* is expected to expose:

    - ``vegetation_type``: identifier string
    - ``age_classes``: sequence of ``(label, cumulative_probability)``
    - ``patch_weibull``: ``(shape, scale, shift)`` tuple describing patch areas
    """

    templates: list[StandAttributeTemplate] = []
    for record in records:
        veg = str(record["vegetation_type"])
        raw_age_classes = cast(Sequence[tuple[Any, Any]], record["age_classes"])
        age_classes = tuple(
            (str(label), float(cumulative)) for label, cumulative in raw_age_classes
        )
        if not age_classes or age_classes[-1][1] < 1.0:
            raise ValueError(
                f"Age-class CDF for vegetation type '{veg}' must end at probability 1.0."
            )
        weibull_seq = tuple(float(value) for value in cast(Sequence[Any], record["patch_weibull"]))
        if len(weibull_seq) != 3:
            raise ValueError("patch_weibull must contain (shape, scale, shift).")
        weibull_params = cast(tuple[float, float, float], tuple(weibull_seq))
        templates.append(
            StandAttributeTemplate(
                vegetation_type=veg,
                age_class_cdf=age_classes,
                patch_weibull=weibull_params,  # type: ignore[arg-type]
            )
        )
    return templates


def load_templates_from_json(path: Path) -> list[StandAttributeTemplate]:
    """Load stand templates from a JSON file (list of record dictionaries)."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Template JSON must be a list of records.")
    return build_templates(cast(Iterable[Mapping[str, object]], payload))
