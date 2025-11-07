"""Core dataclasses and shared type aliases for nemora modules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

ArrayLike: TypeAlias = np.ndarray | Sequence[float]
TableLike: TypeAlias = pd.DataFrame | Mapping[str, Sequence[float]]


@dataclass(slots=True)
class InventorySpec:
    """Describe a single inventory tally source."""

    name: str
    sampling: str  # e.g. "hps", "fixed-area"
    bins: ArrayLike
    tallies: ArrayLike
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FitResult:
    """Container for a single distribution fit."""

    distribution: str
    parameters: dict[str, float]
    covariance: np.ndarray | None = None
    gof: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MixtureComponentFit:
    """Store the result of a single mixture component."""

    name: str
    weight: float
    parameters: dict[str, float]
    gof: dict[str, float] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MixtureFitResult:
    """Group the outputs of a finite-mixture fit."""

    distribution: str
    components: list[MixtureComponentFit]
    log_likelihood: float
    iterations: int
    converged: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FitSummary:
    """Aggregate fit outputs over candidate distributions."""

    inventory: InventorySpec
    results: list[FitResult]
    best: FitResult | None = None
    notes: str | None = None

    def to_frame(self) -> pd.DataFrame:
        """Return a tidy data frame summarising candidate results."""
        records: list[dict[str, Any]] = []
        for result in self.results:
            record: dict[str, Any] = {"distribution": result.distribution}
            record.update(result.parameters)
            record.update({f"gof_{k}": v for k, v in result.gof.items()})
            records.append(record)
        return pd.DataFrame.from_records(records)


__all__ = [
    "ArrayLike",
    "TableLike",
    "InventorySpec",
    "FitResult",
    "MixtureComponentFit",
    "MixtureFitResult",
    "FitSummary",
]
