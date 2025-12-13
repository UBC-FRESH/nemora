from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas.testing as pdt

from nemora.core import FitResult
from nemora.sampling import BootstrapResult, SamplingConfig, bootstrap_inventory
from nemora.sampling.helpers import bootstrap_dbh_vectors
from nemora.synthesis import stands
from nemora.synthesis.helpers import (
    StandDBHSampler,
    bootstrap_payload,
    bootstrap_to_dataframe,
    build_dbh_samplers,
)


def _bootstrap_result() -> tuple[BootstrapResult, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(512)
    bins = np.array([5.0, 15.0, 25.0, 35.0])
    tallies = np.array([3.0, 4.0, 2.0, 1.0])
    fit = FitResult(
        distribution="weibull",
        parameters={"a": 2.4, "beta": 11.0, "s": 1.0},
    )
    result = cast(
        BootstrapResult,
        bootstrap_inventory(
            fit,
            bins,
            tallies,
            resamples=3,
            sample_size=6,
            random_state=rng,
            return_result=True,
        ),
    )
    return result, bins, tallies


def test_bootstrap_to_dataframe_includes_metadata() -> None:
    result, bins, tallies = _bootstrap_result()
    frame = bootstrap_to_dataframe(result)
    assert list(frame.columns) == ["resample", "bin", "draw"]
    metadata = frame.attrs["nemora_bootstrap"]
    assert metadata["distribution"] == result.distribution
    assert metadata["parameters"] == dict(result.parameters)
    np.testing.assert_allclose(metadata["bins"], bins)
    np.testing.assert_allclose(metadata["tallies"], tallies)
    assert metadata["resamples"] == result.resamples
    assert metadata["sample_size"] == result.sample_size
    assert metadata["rng_seed"] == result.rng_seed


def test_bootstrap_payload_matches_dataframe() -> None:
    result, _, _ = _bootstrap_result()
    payload = bootstrap_payload(result)
    frame = bootstrap_to_dataframe(result)
    pdt.assert_frame_equal(payload.frame, frame)
    assert payload.stacked.shape[1] == 2
    assert payload.stacked.shape[0] == result.resamples * result.sample_size
    other = frame.attrs["nemora_bootstrap"]
    assert payload.metadata.keys() == other.keys()
    for key, value in payload.metadata.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_allclose(value, other[key])
        else:
            assert value == other[key]


def test_build_dbh_samplers_handles_bootstrap_and_analytic() -> None:
    result, _, _ = _bootstrap_result()
    bootstrap_payload_struct = bootstrap_dbh_vectors(
        result, stand_id="stand-0001", include_frame=False
    )
    bootstrap_entry = stands.StandBootstrapLibraryEntry(
        identifier="bootstrap-1",
        source="bootstrap.json",
        metadata=_normalize_metadata(bootstrap_payload_struct.metadata),
        dbh_vectors={
            str(idx): values.tolist()
            for idx, values in bootstrap_payload_struct.dbh_vectors.items()
        },
    )
    analytic_entry = stands.StandBootstrapLibraryEntry(
        identifier="analytic-1",
        source="analytic",
        metadata={
            "distribution": "lognormal",
            "parameters": {"mean": 2.1, "sigma": 0.45},
            "sample_size": 4,
            "mode": "analytic",
        },
        dbh_vectors={},
    )
    manifest = stands.StandBootstrapManifest(
        attributes_source=None,
        plan_source=None,
        assignments=(
            stands.StandBootstrapAssignment(
                stand_id="stand-0001",
                vegetation_type="fir",
                age_class="60-80",
                area=3.0,
                bootstrap_id="bootstrap-1",
            ),
            stands.StandBootstrapAssignment(
                stand_id="stand-0002",
                vegetation_type="pine",
                age_class="20-40",
                area=2.5,
                bootstrap_id="analytic-1",
            ),
        ),
        bootstraps={
            "bootstrap-1": bootstrap_entry,
            "analytic-1": analytic_entry,
        },
    )
    samplers = build_dbh_samplers(manifest, sampling_config=SamplingConfig(grid_points=256))
    assert len(samplers) == 2
    bootstrap_sampler = samplers[0]
    analytic_sampler = samplers[1]
    assert isinstance(bootstrap_sampler, StandDBHSampler)
    assert bootstrap_sampler.sampler_type == "bootstrap"
    assert analytic_sampler.sampler_type == "analytic"
    resample_vector = bootstrap_sampler.draw(resample=0)
    assert resample_vector.shape[0] == bootstrap_payload_struct.metadata["sample_size"]
    draw_subset = bootstrap_sampler.draw(sample_size=4, rng=np.random.default_rng(2025))
    assert draw_subset.shape == (4,)
    np.testing.assert_array_less(0.0, draw_subset)
    analytic_draws = analytic_sampler.draw(sample_size=5, rng=np.random.default_rng(123))
    assert analytic_draws.shape == (5,)
    np.testing.assert_array_less(0.0, analytic_draws)


def _normalize_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            normalized[key] = value.tolist()
        else:
            normalized[key] = value
    return normalized
