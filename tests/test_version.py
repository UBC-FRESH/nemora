"""Version regression for the release candidate."""

from __future__ import annotations

from nemora import __version__


def test_candidate_version_is_frozen() -> None:
    assert __version__ == "0.1.0a1"
