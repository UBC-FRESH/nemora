from pathlib import Path

import numpy as np
import pytest

from nemora.workflows.hps import fit_hps_inventory


def load_example(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def test_psp_plot_weibull_matches_expected() -> None:
    dbh, tally = load_example(Path("examples/hps_baf12/4000002_PSP1_v1_p1.csv"))

    results = fit_hps_inventory(dbh, tally, baf=12.0)
    best = min(results, key=lambda result: result.gof["rss"])

    assert best.distribution == "weibull"
    assert best.gof["rss"] == pytest.approx(41847702.91568501, rel=1e-6)

    expected_params = {
        "a": 2.762844978640213,
        "beta": 13.778112123083137,
        "s": 69732.71124303175,
    }
    for key, value in expected_params.items():
        assert best.parameters[key] == pytest.approx(value, rel=1e-6)
