import pandas as pd
import pytest

from nemora.workflows.censoring import fit_censored_inventory


def test_censored_meta_plot_prefers_gamma_distribution() -> None:
    data = pd.read_csv("examples/data/reference_hps/binned_meta_plots.csv")
    meta = (
        data[data["dbh_cm"] >= 20.0]
        .groupby("dbh_cm", as_index=False)
        .agg({"tally": "sum", "expansion_factor": "mean"})
    )

    dbh = meta["dbh_cm"].to_numpy()
    stand_table = meta["tally"].to_numpy() * meta["expansion_factor"].to_numpy()

    results = fit_censored_inventory(dbh, stand_table, support=(20.0, float("inf")))
    best = min(results, key=lambda result: result.gof["rss"])

    assert best.distribution == "gamma"
    assert best.gof["rss"] == pytest.approx(19749.56202467215, rel=1e-6)

    expected_params = {
        "beta": 15.535343492036668,
        "p": 2.7040655499433637,
        "s": 58396.63266805388,
    }

    for key, value in expected_params.items():
        assert best.parameters[key] == pytest.approx(value, rel=1e-6)
