from __future__ import annotations

from typing import cast

import numpy as np
import pandas.testing as pdt

from nemora.core import FitResult
from nemora.sampling import BootstrapResult, bootstrap_inventory
from nemora.synthforest.helpers import bootstrap_payload, bootstrap_to_dataframe


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
