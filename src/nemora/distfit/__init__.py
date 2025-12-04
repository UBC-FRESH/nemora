"""Compatibility shim for the renamed :mod:`nemora.fit` package."""

from __future__ import annotations

import warnings

from ..fit import *  # noqa: F401,F403
from ..fit import __all__ as _fit_all

warnings.warn(
    "`nemora.distfit` has been renamed to `nemora.fit`; import from the new namespace.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = _fit_all
