"""Tests for the optional sgsR R bridge."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from nemora.cli import app
from nemora.sampling.sgsr import SgsRError, run_sgsr

FIXTURE = Path("examples/sgsr_sampling/structure.asc")
runner = CliRunner()


def _sgsr_available() -> bool:
    rscript = shutil.which("Rscript")
    if rscript is None:
        return False
    # Only the discovered Rscript is executed.
    completed = subprocess.run(  # noqa: S603
        [  # noqa: S603
            rscript,
            "-e",
            "quit(status=!requireNamespace('sgsR', quietly=TRUE))",
        ],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
        check=False,
    )
    return completed.returncode == 0


SGSR_AVAILABLE = _sgsr_available()
requires_sgsr = pytest.mark.skipif(
    not SGSR_AVAILABLE,
    reason="Rscript with sgsR is not installed; install sgsR to run the bridge integration test.",
)


def test_missing_metric_raises(tmp_path: Path) -> None:
    with pytest.raises(SgsRError, match="does not exist"):
        run_sgsr(tmp_path / "missing.asc", tmp_path / "out", n_samples=10, seed=1)


def test_bad_arguments_raise(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_samples"):
        run_sgsr(FIXTURE, tmp_path / "out", n_samples=0, seed=1)
    with pytest.raises(ValueError, match="n_strata"):
        run_sgsr(FIXTURE, tmp_path / "out", n_samples=10, seed=1, n_strata=1)
    with pytest.raises(ValueError, match="allocation"):
        run_sgsr(FIXTURE, tmp_path / "out", n_samples=10, seed=1, allocation="bogus")
    with pytest.raises(ValueError, match="method"):
        run_sgsr(FIXTURE, tmp_path / "out", n_samples=10, seed=1, method="bogus")
    with pytest.raises(ValueError, match="mindist"):
        run_sgsr(FIXTURE, tmp_path / "out", n_samples=10, seed=1, mindist=-1.0)


def test_missing_rscript_raises(tmp_path: Path) -> None:
    with pytest.raises(SgsRError, match="Rscript not found"):
        run_sgsr(
            FIXTURE,
            tmp_path / "out",
            n_samples=10,
            seed=1,
            rscript="definitely-not-a-real-rscript",
        )


def test_seed_is_required(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        run_sgsr(FIXTURE, tmp_path / "out", n_samples=10)  # type: ignore[call-arg]


@requires_sgsr
def test_sgsr_bridge_is_reproducible_and_exposes_strata(tmp_path: Path) -> None:
    first = run_sgsr(FIXTURE, tmp_path / "first", n_samples=24, seed=42, n_strata=3)
    second = run_sgsr(FIXTURE, tmp_path / "second", n_samples=24, seed=42, n_strata=3)

    pd.testing.assert_frame_equal(first.allocation, second.allocation)
    pd.testing.assert_frame_equal(first.samples, second.samples)
    assert first.sample_count == 24
    assert set(first.allocation.columns) >= {"strata", "total"}
    assert set(first.samples.columns) >= {"x", "y", "strata", "type", "rule"}
    assert set(first.strata.columns) >= {"strata", "cells", "min", "max", "mean"}
    assert not first.strata.empty
    assert first.strata_path.is_file()
    assert first.metadata["seed"] == "42"
    assert first.metadata["n_strata"] == "3"
    assert first.metadata["sgsr_licence"] == "GPL (>= 3)"
    assert first.metadata["n_samples_returned"] == "24"


@requires_sgsr
def test_constant_metric_is_rejected(tmp_path: Path) -> None:
    constant = tmp_path / "constant.asc"
    lines = [
        "ncols 5",
        "nrows 5",
        "xllcorner 0",
        "yllcorner 0",
        "cellsize 10",
        "NODATA_value -9999",
    ]
    lines.extend(["5 5 5 5 5"] * 5)
    constant.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(SgsRError, match="fewer than two strata"):
        run_sgsr(constant, tmp_path / "out", n_samples=10, seed=1, n_strata=3)


@requires_sgsr
def test_sgsr_cli(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "sampling-sgsr",
            str(FIXTURE),
            "--output",
            str(tmp_path / "out"),
            "--n-samples",
            "12",
            "--seed",
            "42",
            "--n-strata",
            "3",
        ],
    )

    assert result.exit_code == 0, result.stdout + result.stderr
    assert (tmp_path / "out" / "allocation.csv").is_file()
    assert (tmp_path / "out" / "samples.csv").is_file()
    assert (tmp_path / "out" / "strata.tif").is_file()
