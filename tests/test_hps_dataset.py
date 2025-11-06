from pathlib import Path

from nemora.dataprep import (
    aggregate_hps_tallies,
    load_plot_selections,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "hps"


def test_load_plot_selections_filters_first_measurement_only() -> None:
    plot_header = FIXTURE_DIR / "plot_header.csv"
    sample_byvisit = FIXTURE_DIR / "sample_byvisit.csv"

    selections = load_plot_selections(
        plot_header,
        sample_byvisit,
        baf=12.0,
    )

    assert len(selections) == 2
    plot_ids = sorted(sel.plot_id for sel in selections)
    assert plot_ids == ["4000002_PSP1_v1_p1", "4000002_PSP1_v1_p2"]
    assert all(sel.baf == 12.0 for sel in selections)
    assert all(sel.visit_number == 1 for sel in selections)


def test_aggregate_hps_tallies_counts_live_trees_only() -> None:
    plot_header = FIXTURE_DIR / "plot_header.csv"
    sample_byvisit = FIXTURE_DIR / "sample_byvisit.csv"
    tree_detail = FIXTURE_DIR / "tree_detail.csv"

    selections = load_plot_selections(
        plot_header,
        sample_byvisit,
        baf=12.0,
    )

    tallies, metadata = aggregate_hps_tallies(
        tree_detail,
        selections,
        bin_width=1.0,
        bin_origin=0.0,
        live_status=("L",),
        chunk_size=2,
    )

    assert set(tallies) == {"4000002_PSP1_v1_p1", "4000002_PSP1_v1_p2"}

    df1 = tallies["4000002_PSP1_v1_p1"]
    assert list(df1["dbh_cm"]) == [17.5, 18.5, 20.5]
    assert list(df1["tally"]) == [1, 1, 1]

    df2 = tallies["4000002_PSP1_v1_p2"]
    assert list(df2["dbh_cm"]) == [15.5]
    assert list(df2["tally"]) == [1]

    assert metadata.sort_values("plot")["trees"].tolist() == [3, 1]
    assert metadata["cluster_id"].unique().tolist() == ["4000002-PSP1"]
