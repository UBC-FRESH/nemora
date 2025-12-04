"""Helper utilities built on top of bootstrap sampling results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from . import BootstrapResult

__all__ = ["DBHBootstrap", "bootstrap_dbh_vectors"]


@dataclass(slots=True)
class DBHBootstrap:
    """Structured representation of bootstrap DBH samples per stand."""

    stand_id: str | int | None
    dbh_vectors: dict[int, np.ndarray]
    metadata: dict[str, Any]
    frame: pd.DataFrame | None


def bootstrap_dbh_vectors(
    result: BootstrapResult,
    *,
    stand_id: str | int | None = None,
    include_frame: bool = True,
    extra_metadata: Mapping[str, Any] | None = None,
) -> DBHBootstrap:
    """Convert a :class:`BootstrapResult` into per-resample DBH vectors."""

    metadata: dict[str, Any] = {
        "stand_id": stand_id,
        "distribution": result.distribution,
        "parameters": dict(result.parameters),
        "resamples": result.resamples,
        "sample_size": result.sample_size,
        "rng_seed": result.rng_seed,
        "bins": np.asarray(result.bins, dtype=float),
        "tallies": np.asarray(result.tallies, dtype=float),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    dbh_vectors: dict[int, np.ndarray] = {}
    for idx, sample in enumerate(result.samples):
        if sample.size == 0:
            dbh_vectors[idx] = np.empty(0, dtype=float)
            continue
        dbh_vectors[idx] = np.asarray(sample[:, 1], dtype=float)

    frame = None
    if include_frame:
        import pandas as pd

        tally_total = float(result.tallies.sum()) if result.tallies.size else 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            tally_weights = (
                result.tallies / tally_total if tally_total > 0 else np.zeros_like(result.tallies)
            )
        bin_weights = {
            float(bin_value): float(weight)
            for bin_value, weight in zip(result.bins, tally_weights, strict=False)
        }
        rows: list[dict[str, float | int | str | None]] = []
        for idx, sample in enumerate(result.samples):
            if sample.size == 0:
                continue
            for bin_value, draw_value in sample:
                weight = bin_weights.get(float(bin_value))
                rows.append(
                    {
                        "stand_id": stand_id,
                        "resample": idx,
                        "bin": float(bin_value),
                        "draw": float(draw_value),
                        "dbh": float(draw_value),
                        "weight": weight,
                    }
                )
        frame = pd.DataFrame(rows, columns=["stand_id", "resample", "bin", "draw", "dbh", "weight"])

    return DBHBootstrap(
        stand_id=stand_id,
        dbh_vectors=dbh_vectors,
        metadata=metadata,
        frame=frame,
    )
