"""Core package exports for nemora."""

from __future__ import annotations

from importlib import metadata

__version__ = "0.0.1"

try:
    __version__ = metadata.version("nemora")
except metadata.PackageNotFoundError:  # pragma: no cover - local dev fallback
    pass

from .typing import FitResult, FitSummary, InventorySpec  # noqa: F401
from .workflows.censoring import fit_censored_inventory  # noqa: F401
from .workflows.hps import fit_hps_inventory  # noqa: F401

__all__ = [
    "__version__",
    "FitResult",
    "FitSummary",
    "InventorySpec",
    "fit_hps_inventory",
    "fit_censored_inventory",
]
