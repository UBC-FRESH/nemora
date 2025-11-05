"""Core distribution registry infrastructure."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np

Pdf = Callable[[np.ndarray, Mapping[str, float]], np.ndarray]


@dataclass(slots=True)
class Distribution:
    """Describe a candidate PDF with metadata."""

    name: str
    parameters: tuple[str, ...]
    pdf: Pdf
    cdf: Pdf | None = None
    bounds: dict[str, tuple[float | None, float | None]] | None = None
    notes: str | None = None


_REGISTRY: dict[str, Distribution] = {}


def list_distributions() -> Iterable[str]:
    """Return registered distribution names."""
    return sorted(_REGISTRY.keys())


def get_distribution(name: str) -> Distribution:
    """Retrieve a distribution by name."""
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(f"Unknown distribution '{name}'.")
    return _REGISTRY[key]


def register_distribution(distribution: Distribution, *, overwrite: bool = False) -> None:
    """Register a distribution in the global registry."""
    key = distribution.name.lower()
    if key in _REGISTRY and not overwrite:
        raise ValueError(f"Distribution '{distribution.name}' already registered.")
    _REGISTRY[key] = distribution


def clear_registry() -> None:
    """Reset the registry (primarily for testing)."""
    _REGISTRY.clear()


__all__ = [
    "Distribution",
    "Pdf",
    "list_distributions",
    "get_distribution",
    "register_distribution",
    "clear_registry",
]
