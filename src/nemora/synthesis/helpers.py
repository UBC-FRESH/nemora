"""Bootstrap helpers for synthesis (forest/stand/tree) consumers."""

from __future__ import annotations

import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from ..sampling import BootstrapResult, SamplingConfig, sample_distribution
from . import stands

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

_METADATA_ATTR = "nemora_bootstrap"

__all__ = [
    "BootstrapPayload",
    "StandDBHSampler",
    "bootstrap_payload",
    "bootstrap_to_dataframe",
    "build_dbh_samplers",
]


@dataclass(slots=True)
class BootstrapPayload:
    """Structured payload for synthesis bootstrap consumers."""

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


class StandDBHSampler:
    """Sampler that draws DBH values for a single stand."""

    def __init__(
        self,
        assignment: stands.StandBootstrapAssignment,
        entry: stands.StandBootstrapLibraryEntry,
        *,
        sampling_config: SamplingConfig | None = None,
    ) -> None:
        self.assignment = assignment
        self.bootstrap_id = assignment.bootstrap_id
        self.metadata = dict(entry.metadata)
        self.sampling_config = sampling_config or SamplingConfig()
        self._bootstrap_vectors = _coerce_bootstrap_vectors(entry.dbh_vectors)
        if self._bootstrap_vectors:
            self.sampler_type: Literal["bootstrap", "analytic"] = "bootstrap"
            self._bootstrap_pool = (
                np.concatenate(list(self._bootstrap_vectors.values()))
                if self._bootstrap_vectors
                else np.empty(0, dtype=float)
            )
            self._analytic_distribution = None
            self._analytic_parameters = None
        else:
            self.sampler_type = "analytic"
            self._bootstrap_pool = np.empty(0, dtype=float)
            self._analytic_distribution, self._analytic_parameters = _extract_analytic_spec(
                entry.metadata,
                bootstrap_id=self.bootstrap_id,
            )

    def draw(
        self,
        *,
        rng: np.random.Generator | None = None,
        sample_size: int | None = None,
        resample: int | None = None,
    ) -> np.ndarray:
        """Draw DBH values using the configured sampler."""

        rng = rng or np.random.default_rng()
        if self.sampler_type == "bootstrap":
            if resample is not None:
                vector = self._bootstrap_vectors.get(resample)
                if vector is None:
                    raise ValueError(
                        f"Resample {resample} not found for stand {self.assignment.stand_id}."
                    )
                return _maybe_resample_vector(vector, rng=rng, sample_size=sample_size)
            pool = self._bootstrap_pool
            if pool.size == 0:
                return np.empty(0, dtype=float)
            size = sample_size or pool.size
            if size == pool.size and sample_size is None:
                return pool.copy()
            indices = rng.choice(pool.size, size=size, replace=True)
            return pool[indices]
        # Analytic sampler
        distribution = self._analytic_distribution
        params = self._analytic_parameters or {}
        if not distribution:
            raise ValueError(
                "Analytic sampler for stand "
                f"{self.assignment.stand_id} lacks distribution metadata."
            )
        size = sample_size or _coerce_positive_int(self.metadata.get("sample_size"))
        if size <= 0:
            raise ValueError(
                "Analytic sampler requires a positive sample size "
                f"(stand={self.assignment.stand_id})."
            )
        draws = sample_distribution(
            distribution,
            params,
            size,
            random_state=rng,
            config=self.sampling_config,
        )
        return draws


def build_dbh_samplers(
    manifest: stands.StandBootstrapManifest,
    *,
    sampling_config: SamplingConfig | None = None,
) -> list[StandDBHSampler]:
    """Construct DBH samplers for each stand assignment in a manifest."""

    config = sampling_config or SamplingConfig()
    samplers: list[StandDBHSampler] = []
    for assignment in manifest.assignments:
        entry = manifest.bootstraps.get(assignment.bootstrap_id)
        if entry is None:
            raise ValueError(
                f"Stand '{assignment.stand_id}' references unknown bootstrap_id "
                f"'{assignment.bootstrap_id}'."
            )
        samplers.append(
            StandDBHSampler(
                assignment=assignment,
                entry=entry,
                sampling_config=config,
            )
        )
    return samplers


def _coerce_bootstrap_vectors(
    dbh_vectors: Mapping[str, list[float]],
) -> dict[int, np.ndarray]:
    converted: dict[int, np.ndarray] = {}
    for key, values in dbh_vectors.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            idx = len(converted)
        converted[idx] = np.asarray(values, dtype=float)
    return converted


def _extract_analytic_spec(
    metadata: Mapping[str, object],
    *,
    bootstrap_id: str,
) -> tuple[str, dict[str, float]]:
    distribution_raw = metadata.get("distribution")
    params_raw = metadata.get("parameters")
    if not isinstance(distribution_raw, str) or not distribution_raw.strip():
        raise ValueError(
            f"Analytic payload '{bootstrap_id}' requires a non-empty 'distribution' string."
        )
    if not isinstance(params_raw, Mapping):
        raise ValueError(f"Analytic payload '{bootstrap_id}' requires a 'parameters' mapping.")
    params: dict[str, float] = {}
    for key, value in params_raw.items():
        params[str(key)] = _coerce_float(value, field=str(key), bootstrap_id=bootstrap_id)
    distribution = distribution_raw.strip()
    normalized = _normalize_analytic_parameters(
        distribution,
        params,
        bootstrap_id=bootstrap_id,
    )
    return distribution, normalized


def _maybe_resample_vector(
    vector: np.ndarray,
    *,
    rng: np.random.Generator,
    sample_size: int | None,
) -> np.ndarray:
    if sample_size is None or sample_size >= vector.size:
        return vector.copy()
    indices = rng.choice(vector.size, size=sample_size, replace=True)
    return vector[indices]


def _coerce_positive_int(value: object | None, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, numbers.Integral):
        return max(int(value), 0)
    if isinstance(value, numbers.Real):
        return max(int(float(value)), 0)
    if isinstance(value, str | bytes | bytearray):
        try:
            converted = float(value)
        except ValueError:
            return default
        return max(int(converted), 0)
    return default


def _coerce_float(value: object, *, field: str, bootstrap_id: str) -> float:
    if isinstance(value, numbers.Real):
        return float(value)
    if isinstance(value, str | bytes | bytearray):
        try:
            return float(value)
        except ValueError as exc:  # pragma: no cover - error path
            raise ValueError(
                f"Analytic payload '{bootstrap_id}' requires numeric parameter '{field}'."
            ) from exc
    raise ValueError(f"Analytic payload '{bootstrap_id}' requires numeric parameter '{field}'.")


def _normalize_analytic_parameters(
    distribution: str,
    params: dict[str, float],
    *,
    bootstrap_id: str,
) -> dict[str, float]:
    name = distribution.lower()
    if name in {"lognormal", "ln"}:
        return _coerce_lognormal_params(params, bootstrap_id=bootstrap_id)
    return params


def _coerce_lognormal_params(
    params: dict[str, float],
    *,
    bootstrap_id: str,
) -> dict[str, float]:
    normalized = dict(params)
    if "mu" not in normalized:
        mean_val = normalized.pop("mean", None)
        if mean_val is None:
            raise ValueError(
                "Analytic payload "
                f"'{bootstrap_id}' requires either 'mu' or 'mean' for lognormal samplers."
            )
        normalized["mu"] = mean_val
    if "sigma2" not in normalized:
        sigma_val = normalized.pop("sigma", None)
        if sigma_val is None:
            raise ValueError(
                "Analytic payload "
                f"'{bootstrap_id}' requires either 'sigma2' or 'sigma' for lognormal samplers."
            )
        normalized["sigma2"] = sigma_val**2
    return normalized
