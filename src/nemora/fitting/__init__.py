"""Compatibility wrapper forwarding to ``nemora.distfit``."""

from __future__ import annotations

from warnings import warn

from ..distfit import *  # noqa: F401,F403

warn(
    "`nemora.fitting` has moved to `nemora.distfit`. Please update imports accordingly.",
    DeprecationWarning,
    stacklevel=2,
)
