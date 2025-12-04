"""Compatibility wrapper forwarding to ``nemora.fit``."""

from __future__ import annotations

from warnings import warn

from ..fit import *  # noqa: F401,F403
from ..fit import __all__ as _fit_all

warn(
    "`nemora.fitting` has moved to `nemora.fit`. Please update imports accordingly.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = _fit_all
