from pathlib import Path

import pandas as pd

from nemora.ingest.fia import (
    FIATables,
    aggregate_plot_stand_table,
    build_stand_table_from_csvs,
)


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


def test_build_stand_table_from_csvs(tmp_path: Path) -> None:
    tree = pd.DataFrame(
        {
            "PLT_CN": [100, 100],
            "SUBP": [1, 1],
            "TREE": [1, 2],
            "CONDID": [1, 1],
            "STATUSCD": [1, 1],
            "SPCD": [200, 201],
            "DIA": [8.0, 16.0],
            "TPA_UNADJ": [10.0, 5.0],
        }
    )
    cond = pd.DataFrame(
        {
            "PLT_CN": [100],
            "CONDID": [1],
            "COND_STATUS_CD": [1],
            "CONDPROP_UNADJ": [1.0],
            "SUBPPROP_UNADJ": [None],
            "MICRPROP_UNADJ": [None],
            "MACRPROP_UNADJ": [None],
        }
    )
    plot = pd.DataFrame({"CN": [100], "PLOT": [3001]})
    tree.to_csv(tmp_path / "TREE.csv", index=False)
    cond.to_csv(tmp_path / "COND.csv", index=False)
    plot.to_csv(tmp_path / "PLOT.csv", index=False)

    result = build_stand_table_from_csvs(tmp_path, plot_cn=100)
    assert not result.empty
    assert set(result.columns) == {"plot_cn", "plot", "dbh_cm", "tally"}
    assert result.loc[result["dbh_cm"] == 20.0, "tally"].iloc[0] == 10.0


def test_aggregate_plot_stand_table_empty_when_no_live_records() -> None:
    tree = pd.DataFrame(
        {
            "PLT_CN": [1],
            "SUBP": [1],
            "TREE": [1],
            "CONDID": [1],
            "STATUSCD": [2],  # removed because not live
            "SPCD": [100],
            "DIA": [12.0],
            "TPA_UNADJ": [3.0],
        }
    )
    cond = pd.DataFrame(
        {
            "PLT_CN": [1],
            "CONDID": [1],
            "COND_STATUS_CD": [1],
            "CONDPROP_UNADJ": [1.0],
            "SUBPPROP_UNADJ": [None],
            "MICRPROP_UNADJ": [None],
            "MACRPROP_UNADJ": [None],
        }
    )
    plot = pd.DataFrame({"CN": [1], "PLOT": [1]})
    tables = FIATables(tree=tree, condition=cond, plot=plot)
    result = aggregate_plot_stand_table(tables)
    assert result.empty
