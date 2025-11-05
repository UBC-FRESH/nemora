from pathlib import Path

import pandas as pd
import pytest

from dbhdistfit.workflows.censoring import fit_censored_inventory


def test_censored_meta_plot_prefers_gamma_distribution() -> None:
    data = pd.read_csv(Path("tests/fixtures/hps/meta_censored.csv"))
    dbh = data["dbh_cm"].to_numpy()
    stand_table = data["stand_table"].to_numpy()

    results = fit_censored_inventory(dbh, stand_table, support=(9.0, float("inf")))
    best = min(results, key=lambda result: result.gof["rss"])

    assert best.distribution == "gamma"
    assert best.gof["rss"] == pytest.approx(204741183.21815377, rel=1e-6)

    expected_params = {
        "beta": 5.2110659713603775,
        "p": 2.5648573532685375,
        "s": 277655.3865602434,
    }

    for key, value in expected_params.items():
        assert best.parameters[key] == pytest.approx(value, rel=1e-6)
