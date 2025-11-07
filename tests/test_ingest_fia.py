from pathlib import Path

import pandas as pd
import pytest

from nemora.ingest.fia import (
    FIATables,
    aggregate_plot_stand_table,
    build_fia_dataset_source,
    build_stand_table_from_csvs,
    download_fia_tables,
)

FIXTURES = Path("tests/fixtures/fia")


def test_aggregate_plot_stand_table_basic() -> None:
    tree = pd.DataFrame(
        {
            "PLT_CN": [1, 1, 1],
            "SUBP": [1, 1, 1],
            "TREE": [1, 2, 3],
            "CONDID": [1, 1, 2],
            "STATUSCD": [1, 1, 2],
            "SPCD": [100, 101, 102],
            "DIA": [10.0, 20.0, 12.0],
            "TPA_UNADJ": [5.0, 3.0, 2.0],
        }
    )
    cond = pd.DataFrame(
        {
            "PLT_CN": [1, 1],
            "CONDID": [1, 2],
            "COND_STATUS_CD": [1, 1],
            "CONDPROP_UNADJ": [0.5, 0.5],
            "SUBPPROP_UNADJ": [None, None],
            "MICRPROP_UNADJ": [None, None],
            "MACRPROP_UNADJ": [None, None],
        }
    )
    plot = pd.DataFrame({"CN": [1], "PLOT": [42]})
    tables = FIATables(tree=tree, condition=cond, plot=plot)

    result = aggregate_plot_stand_table(tables)
    assert list(result["plot_cn"]) == [1, 1]
    assert list(result["plot"]) == [42, 42]
    assert list(result["dbh_cm"]) == [25.0, 51.0]  # rounded to nearest cm
    assert list(result["tally"]) == [2.5, 1.5]


@pytest.mark.parametrize(
    "plot_cn, expected_dbh, expected_tally",
    [
        (47825261010497, 3.0, 28.405695),
        (47825253010497, 20.0, 12.036092),
    ],
)
def test_build_stand_table_from_fixtures(
    plot_cn: int, expected_dbh: float, expected_tally: float
) -> None:
    result = build_stand_table_from_csvs(
        FIXTURES,
        plot_cn=plot_cn,
        tree_file="tree_small.csv",
        cond_file="cond_small.csv",
        plot_file="plot_small.csv",
    )
    assert not result.empty
    assert {"plot_cn", "plot", "dbh_cm", "tally"} == set(result.columns)
    assert plot_cn in result["plot_cn"].unique()
    observed = result.loc[
        (result["plot_cn"] == plot_cn) & (result["dbh_cm"] == expected_dbh),
        "tally",
    ].iloc[0]
    assert observed == pytest.approx(expected_tally, rel=1e-6)


def test_dead_trees_excluded_from_stand_table() -> None:
    tables = load_fixture_tables()
    result = aggregate_plot_stand_table(tables, plot_cn=47825253010497)
    assert result["tally"].sum() == pytest.approx(249.488616, rel=1e-6)


def load_fixture_tables() -> FIATables:
    tree = pd.read_csv(FIXTURES / "tree_small.csv")
    cond = pd.read_csv(FIXTURES / "cond_small.csv")
    plot = pd.read_csv(FIXTURES / "plot_small.csv")
    return FIATables(tree=tree, condition=cond, plot=plot)


def test_download_fia_tables(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def fake_retrieve(url: str, filename: str) -> tuple[str, None]:
        calls.append(url)
        Path(filename).write_text("demo", encoding="utf-8")
        return filename, None

    monkeypatch.setattr("nemora.ingest.fia.urlretrieve", fake_retrieve)
    paths = download_fia_tables(tmp_path, state="hi", tables=("TREE",))
    assert len(paths) == 1
    assert paths[0].name == "HI_TREE.csv"
    assert paths[0].exists()
    assert any(url.endswith("HI_TREE.csv") for url in calls)


def test_build_fia_dataset_source(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_download(
        dest: Path,
        state: str,
        tables: tuple[str, ...],
        overwrite: bool = False,
    ) -> list[Path]:
        calls.append((state, tables))
        target = dest / f"{state}_TREE.csv"
        target.write_text("demo", encoding="utf-8")
        return [target]

    monkeypatch.setattr("nemora.ingest.fia.download_fia_tables", fake_download)

    dataset = build_fia_dataset_source("hi", destination=tmp_path, tables=("TREE",), overwrite=True)
    paths = list(dataset.fetch())
    assert len(paths) == 1
    assert calls == [("HI", ("TREE",))]
    assert dataset.metadata["state"] == "HI"
