"""Core distribution registry infrastructure."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

Pdf = Callable[[np.ndarray, Mapping[str, float]], np.ndarray]

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "nemora.distributions"


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


def _iter_distributions(candidate: Any) -> Iterable[Distribution]:
    if isinstance(candidate, Distribution):
        yield candidate
    elif isinstance(candidate, Mapping) and "name" in candidate and "pdf" in candidate:
        pdf_callable = _load_object(candidate["pdf"])
        cdf_callable = (
            _load_object(candidate["cdf"]) if "cdf" in candidate and candidate["cdf"] else None
        )
        bounds = candidate.get("bounds")
        raw_parameters = candidate.get("parameters", [])
        parameters = tuple(str(param) for param in raw_parameters)
        yield Distribution(
            name=str(candidate["name"]),
            parameters=parameters,
            pdf=pdf_callable,
            cdf=cdf_callable,
            bounds=bounds,
            notes=candidate.get("notes"),
        )
    elif isinstance(candidate, Iterable) and not isinstance(candidate, str | bytes):
        for item in candidate:
            yield from _iter_distributions(item)
    elif callable(candidate):
        result = candidate()
        yield from _iter_distributions(result)
    else:
        raise TypeError(
            "Unsupported distribution specification. Expected Distribution, iterable of "
            "Distribution instances, a callable returning them, or a mapping with name/pdf keys."
        )


def _load_object(path: str) -> Any:
    module_name, _, attribute = path.partition(":")
    if not attribute:
        module_name, _, attribute = path.rpartition(".")
    if not module_name or not attribute:
        raise ValueError(f"Invalid import path '{path}'. Expected 'module:callable'.")
    module = import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise AttributeError(f"Module '{module_name}' has no attribute '{attribute}'.") from exc


def load_entry_points(group: str = ENTRY_POINT_GROUP) -> list[str]:
    """Discover third-party distributions via entry points."""
    loaded: list[str] = []
    try:
        eps = metadata.entry_points()
        candidates: Iterable[Any]
        if hasattr(eps, "select"):  # Python 3.10+
            candidates = eps.select(group=group)
        else:  # pragma: no cover - legacy interface
            candidates = eps.get(group, [])  # type: ignore[call-arg]
    except Exception as exc:  # pragma: no cover - discovery failure
        logger.debug("Entry point discovery failed: %s", exc)
        return loaded

    for ep in candidates:
        try:
            obj = ep.load()
            for dist in _iter_distributions(obj):
                register_distribution(dist, overwrite=True)
                loaded.append(dist.name)
        except Exception as exc:  # pragma: no cover - plugin failure
            logger.warning("Failed to load distribution entry point '%s': %s", ep.name, exc)
    return loaded


def load_yaml_config(path: str | os.PathLike[str]) -> list[str]:
    """Load additional distributions from a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        logger.debug("Skipping distribution config %s (file not found)", path)
        return []

    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:  # pragma: no cover - parse failure
        logger.warning("Failed to parse distribution config %s: %s", path, exc)
        return []

    registered: list[str] = []
    for item in data.get("distributions", []):
        try:
            if "callable" in item:
                factory = _load_object(item["callable"])
                args = item.get("args", [])
                kwargs = item.get("kwargs", {})
                for dist in _iter_distributions(factory(*args, **kwargs)):
                    register_distribution(dist, overwrite=item.get("overwrite", True))
                    registered.append(dist.name)
            else:
                for dist in _iter_distributions(item):
                    register_distribution(dist, overwrite=item.get("overwrite", True))
                    registered.append(dist.name)
        except Exception as exc:
            logger.warning(
                "Failed to register distribution from %s (spec=%s): %s",
                path,
                item,
                exc,
            )
    return registered


__all__ = [
    "Distribution",
    "ENTRY_POINT_GROUP",
    "Pdf",
    "list_distributions",
    "get_distribution",
    "register_distribution",
    "clear_registry",
    "load_entry_points",
    "load_yaml_config",
]
