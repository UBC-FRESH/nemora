"""Stand-attribute templates inspired by the FLG workflow."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

__all__ = ["StandAttributeTemplate", "build_templates"]


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
