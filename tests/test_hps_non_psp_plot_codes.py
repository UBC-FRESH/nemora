"""FAIB non-PSP plot codes are strings, not integers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nemora.dataprep import aggregate_hps_tallies, load_plot_selections


def _write_inputs(root: Path) -> tuple[Path, Path, Path]:
    plot_header = root / "plot_header.csv"
    sample = root / "sample_byvisit.csv"
    tree = root / "tree_detail.csv"
    pd.DataFrame(
        {
            "CLSTR_ID": ["1266066-M1"],
            "VISIT_NUMBER": [1],
            "PLOT": ["I"],
            "SITE_IDENTIFIER": ["1266066"],
        }
    ).to_csv(plot_header, index=False)
    pd.DataFrame(
        {
            "CLSTR_ID": ["1266066-M1"],
            "VISIT_NUMBER": [1],
            "FIRST_MSMT": ["Y"],
            "MEAS_DT": ["2023-06-25"],
            "SAMP_TYP": ["F"],
            "SAMPLE_ESTABLISHMENT_TYPE": ["CMI"],
        }
    ).to_csv(sample, index=False)
    pd.DataFrame(
        {
            "CLSTR_ID": ["1266066-M1"],
            "VISIT_NUMBER": [1],
            "PLOT": ["I"],
            "DBH": [16.0],
            "LV_D": ["L"],
        }
    ).to_csv(tree, index=False)
    return plot_header, sample, tree


def test_non_psp_plot_code_round_trips(tmp_path: Path) -> None:
    """A non-PSP code such as I is preserved through selection and tallying."""
    plot_header, sample, tree = _write_inputs(tmp_path)
    selections = load_plot_selections(plot_header, sample, baf=12.0)
    assert len(selections) == 1
    assert selections[0].plot == "I"
    assert selections[0].plot_id == "1266066_M1_v1_pI"

    tallies, metadata = aggregate_hps_tallies(
        tree,
        selections,
        bin_width=1.0,
        bin_origin=0.0,
        live_status=("L",),
        chunk_size=1,
    )
    assert set(tallies) == {"1266066_M1_v1_pI"}
    assert metadata.iloc[0]["plot"] == "I"
    assert metadata.iloc[0]["trees"] == 1
