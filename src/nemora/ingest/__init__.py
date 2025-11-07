"""Ingestion/ETL scaffolding for Nemora.

This module defines lightweight abstractions that describe raw inventory
datasets (`DatasetSource`) and the transformation pipelines (`TransformPipeline`)
that convert them into the tidy stand tables consumed by other Nemora modules.
Concrete connectors for BC FAIB, FIA, and other inventories will extend these
primitives in upcoming revisions.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd


class DatasetFetcher(Protocol):
    """Callable that retrieves one or more artifacts for a dataset source."""

    def __call__(self, source: DatasetSource) -> Iterable[Path]:
        """Download or locate artifacts backing the dataset."""
        ...


@dataclass(slots=True)
class DatasetSource:
    """Describe a raw inventory dataset that can be ingested by Nemora.

    Parameters
    ----------
    name:
        Human-readable identifier for the dataset.
    description:
        Short summary of the dataset contents (region, sampling design, etc.).
    uri:
        Optional canonical URI (open data portal link, DataLad URL, etc.).
    metadata:
        Arbitrary extra fields (licensing, citation info, cache preferences).
    fetcher:
        Optional callable able to retrieve the dataset artifacts when invoked.
    """

    name: str
    description: str
    uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    fetcher: DatasetFetcher | None = None

    def fetch(self) -> Iterable[Path]:
        """Return artifacts for this dataset via the configured fetcher."""
        if self.fetcher is None:
            raise RuntimeError(f"No fetcher configured for dataset '{self.name}'.")
        return self.fetcher(self)


@dataclass(slots=True)
class TransformPipeline:
    """A sequence of callables that transform raw dataframes into Nemora tables."""

    name: str
    steps: list[Callable[[pd.DataFrame], pd.DataFrame]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(self, step: Callable[[pd.DataFrame], pd.DataFrame]) -> None:
        """Append a transformation step to the pipeline."""
        self.steps.append(step)

    def run(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Apply every transformation step to the supplied dataframe."""
        result = frame.copy()
        for step in self.steps:
            result = step(result)
        return result


from . import faib as faib  # noqa: E402,F401
from . import fia as fia  # noqa: E402,F401

__all__ = [
    "DatasetFetcher",
    "DatasetSource",
    "TransformPipeline",
    "faib",
    "fia",
]
