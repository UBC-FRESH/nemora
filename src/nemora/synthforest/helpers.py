"""Bootstrap helpers for synthforest consumers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..sampling import BootstrapResult

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

_METADATA_ATTR = "nemora_bootstrap"

__all__ = ["BootstrapPayload", "bootstrap_to_dataframe", "bootstrap_payload"]


@dataclass(slots=True)
class BootstrapPayload:
    """Structured payload for synthforest bootstrap consumers."""

    frame: pd.DataFrame
    stacked: np.ndarray
    metadata: dict[str, object]


def _build_metadata(result: BootstrapResult) -> dict[str, object]:
    return {
        "distribution": result.distribution,
        "parameters": dict(result.parameters),
        "bins": np.asarray(result.bins, dtype=float).copy(),
        "tallies": np.asarray(result.tallies, dtype=float).copy(),
        "resamples": result.resamples,
        "sample_size": result.sample_size,
        "rng_seed": result.rng_seed,
    }


def bootstrap_to_dataframe(
    result: BootstrapResult,
    *,
    attach_metadata: bool = True,
) -> pd.DataFrame:
    """Return a DataFrame view of a bootstrap result with optional metadata."""

    frame = result.to_dataframe()
    if attach_metadata:
        frame.attrs[_METADATA_ATTR] = _build_metadata(result)
    return frame


def bootstrap_payload(result: BootstrapResult) -> BootstrapPayload:
    """Return a structured payload with stacked samples + metadata."""

    frame = bootstrap_to_dataframe(result, attach_metadata=True)
    stacked = result.stacked()
    metadata = frame.attrs[_METADATA_ATTR]
    return BootstrapPayload(frame=frame, stacked=stacked, metadata=metadata)
