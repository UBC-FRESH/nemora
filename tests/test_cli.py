import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from typer.testing import CliRunner

from nemora import __version__
from nemora.cli import _check_datalad_results, app
from nemora.workflows import fit_hps_inventory

runner = CliRunner()

_TABLE_PREFIXES = tuple("┏┓┛┗┃┡┠┨┬┴┼┽┾┿╀╁╂╃╄╅╆╇╈╉╊╋│")


def _parse_table_lines(lines: list[str]) -> tuple[list[str], dict[str, dict[str, str]]]:
    header: list[str] = []
    rows: dict[str, dict[str, str]] = {}
    for line in lines:
        if line.startswith("┃"):
            header = [cell.strip() for cell in line.split("┃")[1:-1]]
        elif line.startswith("│") and header:
            cells = [cell.strip() for cell in line.split("│")[1:-1]]
            if len(cells) != len(header):
                continue
            distribution = cells[0]
            rows[distribution] = dict(zip(header[1:], cells[1:], strict=False))
    return header, rows


def _parse_cli_tables(output: str) -> list[tuple[list[str], dict[str, dict[str, str]]]]:
    tables: list[tuple[list[str], dict[str, dict[str, str]]]] = []
    current: list[str] = []
    for line in output.splitlines():
        if line and line[0] in _TABLE_PREFIXES:
            current.append(line)
        else:
            if current:
                tables.append(_parse_table_lines(current))
                current = []
    if current:
        tables.append(_parse_table_lines(current))
    return tables


def _as_float(value: str) -> float:
    return float(value)


def test_registry_command_lists_distributions() -> None:
    result = runner.invoke(app, ["registry"])
    assert result.exit_code == 0
    assert "weibull" in result.stdout.lower()
    assert "Complete-form Weibull" in result.stdout


def test_version_option() -> None:
    result = runner.invoke(app, ["--verbose"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_fit_hps_command_outputs_metrics() -> None:
    result = runner.invoke(
        app,
        [
            "fit-hps",
            "examples/hps_baf12/4000002_PSP1_v1_p1.csv",
            "--baf",
            "12",
        ],
    )
    assert result.exit_code == 0
    tables = _parse_cli_tables(result.stdout)
    header, observed_rows = tables[0]
    assert set(observed_rows) == {"weibull", "gamma"}
    assert {"RSS", "AICc", "Chi^2", "Max |Res|"} <= set(header)

    data = pd.read_csv("examples/hps_baf12/4000002_PSP1_v1_p1.csv")
    expected_results = fit_hps_inventory(
        data["dbh_cm"].to_numpy(),
        data["tally"].to_numpy(),
        baf=12.0,
        distributions=("weibull", "gamma"),
    )
    expected = {res.distribution: res for res in expected_results}

    metric_map = {"RSS": "rss", "AICc": "aicc", "Chi^2": "chisq"}

    for dist, fit_result in expected.items():
        row = observed_rows[dist]
        for column, key in metric_map.items():
            if column in row and key in fit_result.gof:
                assert _as_float(row[column]) == pytest.approx(fit_result.gof[key], rel=1e-6)
        residual_summary = fit_result.diagnostics.get("residual_summary", {})
        if "Max |Res|" in row and residual_summary:
            assert _as_float(row["Max |Res|"]) == pytest.approx(
                residual_summary["max_abs"], rel=1e-6
            )


def test_fit_hps_distribution_filter() -> None:
    result = runner.invoke(
        app,
        [
            "fit-hps",
            "examples/hps_baf12/4000002_PSP1_v1_p1.csv",
            "--baf",
            "12",
            "--distribution",
            "gamma",
        ],
    )
    assert result.exit_code == 0
    tables = _parse_cli_tables(result.stdout)
    _, rows = tables[0]
    assert set(rows) == {"gamma"}


def test_fit_hps_unknown_distribution_errors() -> None:
    result = runner.invoke(
        app,
        [
            "fit-hps",
            "examples/hps_baf12/4000002_PSP1_v1_p1.csv",
            "--baf",
            "12",
            "--distribution",
            "not-a-dist",
        ],
    )
    assert result.exit_code != 0
    assert "not-a-dist" in result.stdout.lower()


def test_fit_hps_parameter_preview_includes_columns() -> None:
    result = runner.invoke(
        app,
        [
            "fit-hps",
            "examples/hps_baf12/4000002_PSP1_v1_p1.csv",
            "--baf",
            "12",
            "--show-parameters",
        ],
    )
    assert result.exit_code == 0
    tables = _parse_cli_tables(result.stdout)
    _, metric_rows = tables[0]
    param_header, param_rows = tables[1]
    assert set(metric_rows) == {"weibull", "gamma"}
    assert set(param_rows) == {"weibull", "gamma"}
    assert {"a", "beta", "s", "p"} <= set(param_header)

    data = pd.read_csv("examples/hps_baf12/4000002_PSP1_v1_p1.csv")
    expected_results = fit_hps_inventory(
        data["dbh_cm"].to_numpy(),
        data["tally"].to_numpy(),
        baf=12.0,
        distributions=("weibull", "gamma"),
    )
    expected = {res.distribution: res for res in expected_results}

    for dist, fit_result in expected.items():
        row = param_rows[dist]
        for param, value in fit_result.parameters.items():
            if param in row:
                assert _as_float(row[param]) == pytest.approx(value, rel=1e-4)


def test_ingest_faib_command(tmp_path: Path) -> None:
    fixtures = Path("tests/fixtures/faib")
    output = tmp_path / "stand_table.csv"

    result = runner.invoke(
        app,
        [
            "ingest-faib",
            str(fixtures),
            "--baf",
            "12",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    assert output.exists()
    df = pd.read_csv(output)
    assert "dbh_cm" in df.columns
    assert "tally" in df.columns


def test_ingest_faib_command_with_fetch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    output = tmp_path / "out.csv"

    def fake_download(
        destination: Path,
        dataset: str,
        *,
        overwrite: bool = False,
        filenames: list[str] | None = None,
    ) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "faib_tree_detail.csv").write_text(
            "CLSTR_ID,VISIT_NUMBER,PLOT,DBH_CM,TREE_EXP\nA,1,1,12.0,2\n",
            encoding="utf-8",
        )
        (destination / "faib_sample_byvisit.csv").write_text(
            "CLSTR_ID,VISIT_NUMBER,PLOT,BAF\nA,1,1,12\n",
            encoding="utf-8",
        )
        return [destination / "faib_tree_detail.csv", destination / "faib_sample_byvisit.csv"]

    monkeypatch.setattr("nemora.cli.download_faib_csvs", fake_download)

    result = runner.invoke(
        app,
        [
            "ingest-faib",
            str(tmp_path),
            "--baf",
            "12",
            "--fetch",
            "--dataset",
            "psp",
            "--cache-dir",
            str(cache_dir),
            "--overwrite",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    assert output.exists()
    df = pd.read_csv(output)
    assert not df.empty


def test_ingest_faib_auto_bafs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_auto(root: Path) -> list[float]:
        return [4.0, 8.0, 12.0]

    monkeypatch.setattr("nemora.cli.auto_select_bafs", fake_auto)
    result = runner.invoke(
        app,
        [
            "ingest-faib",
            str(tmp_path),
            "--auto-bafs",
        ],
    )
    assert result.exit_code == 0
    assert "Suggested BAFs" in result.stdout


def test_faib_manifest_command(tmp_path: Path) -> None:
    fixtures = Path("tests/fixtures/faib")
    destination = tmp_path / "manifest"

    result = runner.invoke(
        app,
        [
            "faib-manifest",
            str(destination),
            "--dataset",
            "psp",
            "--source",
            str(fixtures),
            "--no-fetch",
            "--baf",
            "12",
            "--max-rows",
            "10",
        ],
    )
    assert result.exit_code == 0
    manifest = destination / "faib_manifest.csv"
    assert manifest.exists()
    df = pd.read_csv(manifest)
    assert not df.empty


def test_fetch_reference_data_dry_run_message() -> None:
    result = runner.invoke(app, ["fetch-reference-data"])  # default dry-run
    assert result.exit_code == 0
    assert "reference dataset" in result.stdout.lower()
    assert "nemora-data" in result.stdout
    assert "dry-run" in result.stdout.lower()


def test_check_datalad_results_handles_dataset_objects() -> None:
    class DummyDataset:
        pass

    records = [
        DummyDataset(),
        {"status": "impossible", "message": "broken"},
        {"status": "ok", "message": "ignored"},
    ]
    issues = _check_datalad_results("install", records, fatal=False)
    assert issues == ["impossible: broken"]


def test_fetch_reference_data_falls_back_to_git(monkeypatch, tmp_path) -> None:
    from nemora import cli as cli_module

    calls: dict[str, Any] = {}

    def fake_clone(url: str, dest: Path) -> None:
        calls["url"] = url
        calls["dest"] = dest

    monkeypatch.setattr(cli_module, "_git_clone_dataset", fake_clone)

    def _install(**_: object) -> list[dict[str, str]]:
        return [{"status": "impossible", "message": "boom"}]

    dummy_api = SimpleNamespace(install=_install)
    dummy_module = cast(Any, ModuleType("datalad"))
    dummy_module.api = dummy_api

    support_module = cast(Any, ModuleType("datalad.support"))
    exceptions_module = cast(Any, ModuleType("datalad.support.exceptions"))

    class DummyIncompleteError(Exception):
        pass

    exceptions_module.IncompleteResultsError = DummyIncompleteError
    support_module.exceptions = exceptions_module
    dummy_module.support = support_module

    monkeypatch.setitem(sys.modules, "datalad", dummy_module)
    monkeypatch.setitem(sys.modules, "datalad.api", dummy_api)
    monkeypatch.setitem(sys.modules, "datalad.support", support_module)
    monkeypatch.setitem(sys.modules, "datalad.support.exceptions", exceptions_module)

    result = runner.invoke(
        app,
        [
            "fetch-reference-data",
            "--no-dry-run",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert result.exit_code == 0
    assert calls["url"] == cli_module.REFERENCE_DATA_URL
    assert "git clone" in result.stdout.lower()
