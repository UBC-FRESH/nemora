"""Top-level package exports for Nemora."""

from __future__ import annotations

from importlib import metadata

__version__ = "0.0.1-alpha"

try:
    __version__ = metadata.version("nemora")
except metadata.PackageNotFoundError:  # pragma: no cover - local dev fallback
    pass

from . import core as core  # noqa: F401
from . import distfit as distfit  # noqa: F401
from . import distributions as distributions  # noqa: F401
from .core import FitResult, FitSummary, InventorySpec  # noqa: F401
from .workflows.censoring import fit_censored_inventory  # noqa: F401
from .workflows.hps import fit_hps_inventory  # noqa: F401

__all__ = [
    "__version__",
    "core",
    "distributions",
    "distfit",
    "FitResult",
    "FitSummary",
    "InventorySpec",
    "fit_hps_inventory",
    "fit_censored_inventory",
]
