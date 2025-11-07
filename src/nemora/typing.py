"""Compatibility shim re-exporting core dataclasses."""

from __future__ import annotations

from warnings import warn

from .core import (  # noqa: F401
    ArrayLike,
    FitResult,
    FitSummary,
    InventorySpec,
    MixtureComponentFit,
    MixtureFitResult,
    TableLike,
)

warn(
    "`nemora.typing` is deprecated; import from `nemora.core` instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "ArrayLike",
    "TableLike",
    "InventorySpec",
    "FitResult",
    "MixtureComponentFit",
    "MixtureFitResult",
    "FitSummary",
]
