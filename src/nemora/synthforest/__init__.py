"""Compatibility shim for the renamed :mod:`nemora.synthesis` package."""

from __future__ import annotations

import warnings

from ..synthesis import *  # noqa: F401,F403
from ..synthesis import __all__ as _synthesis_all

warnings.warn(
    "`nemora.synthforest` has been renamed to `nemora.synthesis`; import from the new namespace.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = _synthesis_all
