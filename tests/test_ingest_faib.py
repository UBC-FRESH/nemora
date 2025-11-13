import os
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pytest

from nemora.ingest.faib import (
    NON_PSP_DICTIONARY_URL,
    PSP_DICTIONARY_URL,
    DataDictionary,
    aggregate_stand_table,
    auto_select_bafs,
    build_faib_dataset_source,
    build_faib_stand_table_pipeline,
    build_stand_table_from_csvs,
    download_faib_csvs,
    generate_faib_manifest,
    load_data_dictionary,
    load_non_psp_dictionary,
    load_psp_dictionary,
)
from nemora.ingest.hps import (
    SelectionCriteria as HPSelectionCriteria,
)
from nemora.ingest.hps import (
    export_hps_outputs,
    run_hps_pipeline,
)
from nemora.ingest.hps import (
    load_plot_selections as load_hps_plot_selections,
)


def test_manifest_records_exist() -> None:
    manifest_path = Path("examples/faib_manifest/faib_manifest.csv")
    manifest = pd.read_csv(manifest_path)
    assert not manifest.empty
    assert {"dataset", "baf", "rows", "path", "truncated"} <= set(manifest.columns)
    for rel_path in manifest["path"]:
        assert (manifest_path.parent / rel_path).exists()


def test_auto_select_bafs_prefers_plot_header(tmp_path: Path) -> None:
    root = tmp_path
    plot_header = pd.DataFrame(
        {
            "CLSTR_ID": ["A", "B", "C", "D"],
            "VISIT_NUMBER": [1, 1, 1, 1],
            "PLOT": [1, 1, 1, 1],
            "BLOWUP_MAIN": [10.0, 12.0, 20.0, 40.0],
        }
    )
    plot_header.to_csv(root / "faib_plot_header.csv", index=False)
    bafs = auto_select_bafs(root, count=3)
    assert len(bafs) == 3
    assert min(bafs) >= 10.0
    assert max(bafs) <= 40.0


def test_load_data_dictionary_from_local(tmp_path: Path) -> None:
    sheet = pd.DataFrame({"Attribute": ["col1", "col2"], "Description": ["One", "Two"]})
    path = tmp_path / "demo.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        sheet.to_excel(writer, sheet_name="faib_plot_header", index=False)

    dictionary = load_data_dictionary(str(path))
    assert isinstance(dictionary, DataDictionary)
    schema = dictionary.get_table_schema("faib_plot_header")
    assert list(schema["Attribute"]) == ["col1", "col2"]


@pytest.mark.skip(reason="FTP download requires network and may be rate limited.")
def test_load_data_dictionary_from_ftp() -> None:
    dictionary = load_data_dictionary(
        "ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/psp/"
        "PSP_data_dictionary_20250514.xlsx"
    )
    assert "faib_plot_header" in dictionary.sheets


def test_dictionary_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[str] = []

    def fake_loader(url: str) -> DataDictionary:
        called.append(url)
        return DataDictionary({"demo": pd.DataFrame({"Attribute": ["col1"]})})

    monkeypatch.setattr("nemora.ingest.faib.load_data_dictionary", fake_loader)

    psp_dict = load_psp_dictionary()
    assert psp_dict.get_table_schema("demo").shape == (1, 1)
    non_psp_dict = load_non_psp_dictionary()
    assert non_psp_dict.get_table_schema("demo").shape == (1, 1)

    assert called == [PSP_DICTIONARY_URL, NON_PSP_DICTIONARY_URL]


def test_aggregate_stand_table() -> None:
    tree_detail = pd.DataFrame(
        {
            "CLSTR_ID": ["A", "A", "B"],
            "VISIT_NUMBER": [1, 1, 1],
            "PLOT": [1, 1, 1],
            "DBH_CM": [12.4, 24.8, 10.2],
            "TREE_EXP": [3.0, 2.0, 4.0],
        }
    )
    tree_detail = pd.concat(
        [
            tree_detail,
            pd.DataFrame(
                {
                    "CLSTR_ID": ["A"],
                    "VISIT_NUMBER": [1],
                    "PLOT": [1],
                    "DBH_CM": ["30.6"],
                    "TREE_EXP": ["1.5"],
                }
            ),
        ],
        ignore_index=True,
    )
    sample_byvisit = pd.DataFrame(
        {
            "CLSTR_ID": ["A", "B"],
            "VISIT_NUMBER": [1, 1],
            "PLOT": [1, 1],
            "BAF": [12.0, 8.0],
        }
    )

    stand_table = aggregate_stand_table(tree_detail, sample_byvisit, baf=12.0)
    assert list(stand_table["dbh_cm"]) == [12.0, 25.0, 31.0]
    assert list(stand_table["tally"]) == [3.0, 2.0, 1.5]

    empty = aggregate_stand_table(tree_detail, sample_byvisit, baf=20.0)
    assert empty.empty


def test_build_stand_table_from_csvs(tmp_path: Path) -> None:
    fixture_root = Path("tests/fixtures/faib")
    result = build_stand_table_from_csvs(fixture_root, baf=12.0)
    assert list(result["dbh_cm"]) == [12.0, 13.0, 25.0]


def test_build_faib_stand_table_pipeline() -> None:
    tree_detail = pd.DataFrame(
        {
            "CLSTR_ID": ["A", "A", "B"],
            "VISIT_NUMBER": [1, 1, 1],
            "PLOT": [1, 1, 1],
            "DBH_CM": [12.0, 25.0, 14.0],
            "TREE_EXP": [3.0, 2.0, 1.0],
        }
    )
    plot_info = pd.DataFrame(
        {
            "CLSTR_ID": ["A", "B"],
            "VISIT_NUMBER": [1, 1],
            "PLOT": [1, 1],
            "BAF": [12.0, 8.0],
        }
    )
    pipeline = build_faib_stand_table_pipeline(
        plot_info,
        baf=12.0,
        dbh_col="DBH_CM",
        expansion_col="TREE_EXP",
        baf_col="BAF",
    )
    result = pipeline.run(tree_detail)
    assert list(result.columns) == ["dbh_cm", "tally"]
    assert result["tally"].sum() == pytest.approx(5.0)


def _write_hps_csvs(tmp_path: Path) -> tuple[Path, Path, Path]:
    plot_header = pd.DataFrame(
        {
            "CLSTR_ID": ["A"],
            "VISIT_NUMBER": [1],
            "PLOT": [1],
            "SITE_IDENTIFIER": ["SITE-1"],
        }
    )
    sample_byvisit = pd.DataFrame(
        {
            "CLSTR_ID": ["A"],
            "VISIT_NUMBER": [1],
            "FIRST_MSMT": ["Y"],
            "MEAS_DT": ["2020-06-01"],
            "SAMP_TYP": ["P"],
            "SAMPLE_ESTABLISHMENT_TYPE": ["BASE"],
        }
    )
    tree_detail = pd.DataFrame(
        {
            "CLSTR_ID": ["A", "A", "A"],
            "VISIT_NUMBER": [1, 1, 1],
            "PLOT": [1, 1, 1],
            "DBH": [12.3, 24.9, 30.1],
            "LV_D": ["L", "L", "D"],
        }
    )
    plot_header_path = tmp_path / "faib_plot_header.csv"
    sample_byvisit_path = tmp_path / "faib_sample_byvisit.csv"
    tree_detail_path = tmp_path / "faib_tree_detail.csv"
    plot_header.to_csv(plot_header_path, index=False)
    sample_byvisit.to_csv(sample_byvisit_path, index=False)
    tree_detail.to_csv(tree_detail_path, index=False)
    return plot_header_path, sample_byvisit_path, tree_detail_path


def test_run_hps_pipeline(tmp_path: Path) -> None:
    plot_header_path, sample_byvisit_path, tree_detail_path = _write_hps_csvs(tmp_path)
    selections = load_hps_plot_selections(
        plot_header_path,
        sample_byvisit_path,
        baf=12.0,
        criteria=HPSelectionCriteria(),
        encoding="latin1",
    )
    result = run_hps_pipeline(
        tree_detail_path,
        selections,
        live_status=("L",),
        bin_width=1.0,
        bin_origin=0.0,
        chunk_size=10,
    )
    assert len(result.tallies) == 1
    plot_id, frame = next(iter(result.tallies.items()))
    assert plot_id.startswith("A_v1_p1")
    assert frame["tally"].sum() == 2
    assert not result.manifest.empty
    assert int(result.manifest.iloc[0]["trees"]) == 2


def test_export_hps_outputs(tmp_path: Path) -> None:
    plot_header_path, sample_byvisit_path, tree_detail_path = _write_hps_csvs(tmp_path)
    selections = load_hps_plot_selections(
        plot_header_path,
        sample_byvisit_path,
        baf=12.0,
        criteria=HPSelectionCriteria(),
    )
    result = run_hps_pipeline(tree_detail_path, selections)
    output_dir = tmp_path / "output"
    manifest_path = tmp_path / "manifest.csv"
    export_hps_outputs(
        result.tallies,
        result.manifest,
        output_dir=output_dir,
        manifest_path=manifest_path,
        quiet=True,
    )
    assert manifest_path.exists()
    assert any(output_dir.glob("*.csv"))


def test_download_faib_csvs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    files = ["faib_tree_detail.csv", "faib_sample_byvisit.csv", "readme.txt"]
    file_data = {
        "faib_tree_detail.csv": b"CLSTR_ID,VISIT_NUMBER,PLOT,DBH_CM,TREE_EXP\nA,1,1,12,2\n",
        "faib_sample_byvisit.csv": b"CLSTR_ID,VISIT_NUMBER,PLOT,BAF\nA,1,1,12\n",
    }

    class FakeFTP:
        def __init__(self, *args, **kwargs) -> None:
            self.cwd_path: str | None = None

        def connect(self, hostname: str, port: int = 21) -> None:
            return None

        def login(self) -> None:
            return None

        def cwd(self, path: str) -> None:
            self.cwd_path = path

        def nlst(self) -> list[str]:
            return files

        def retrbinary(self, command: str, callback) -> None:
            name = command.split()[-1]
            callback(file_data.get(name, b""))

        def quit(self) -> None:
            return None

    monkeypatch.setattr("nemora.ingest.faib.FTP", FakeFTP)

    downloaded = download_faib_csvs(tmp_path, dataset="psp")
    assert len(downloaded) == 2
    assert (tmp_path / "faib_tree_detail.csv").exists()


@pytest.mark.network
@pytest.mark.skipif(
    not os.environ.get("NEMORA_RUN_FAIB_INTEGRATION"),
    reason="Set NEMORA_RUN_FAIB_INTEGRATION=1 to exercise live FAIB FTP download.",
)
def test_download_faib_csvs_integration(tmp_path: Path) -> None:
    files = download_faib_csvs(
        tmp_path,
        dataset="psp",
        filenames=["faib_plot_header.csv"],
        overwrite=True,
    )
    assert any(path.name == "faib_plot_header.csv" for path in files)
    assert (tmp_path / "faib_plot_header.csv").exists()


def test_generate_faib_manifest(tmp_path: Path) -> None:
    destination = tmp_path / "manifest"
    fixtures = Path("tests/fixtures/faib")
    result = generate_faib_manifest(
        destination,
        dataset="psp",
        source=fixtures,
        fetch=False,
        bafs=[12.0],
        max_rows=10,
    )
    assert result.manifest_path.exists()
    assert len(result.tables) == 1
    assert result.tables[0].exists()
    manifest = pd.read_csv(result.manifest_path)
    assert set(manifest.columns) == {"dataset", "baf", "rows", "path", "truncated"}


def test_build_faib_dataset_source(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def fake_download(
        destination: Path,
        dataset: str,
        *,
        overwrite: bool = False,
        filenames: Iterable[str] | None = None,
    ) -> list[Path]:
        destination.mkdir(parents=True, exist_ok=True)
        captured.update(
            {
                "destination": destination,
                "dataset": dataset,
                "overwrite": overwrite,
                "filenames": filenames,
            }
        )
        generated = destination / "faib_tree_detail.csv"
        generated.write_text("CLSTR_ID,VISIT_NUMBER,PLOT,DBH_CM,TREE_EXP\n", encoding="utf-8")
        return [generated]

    monkeypatch.setattr("nemora.ingest.faib.download_faib_csvs", fake_download)

    source = build_faib_dataset_source(
        "psp",
        destination=tmp_path / "data",
        filenames=["faib_tree_detail.csv"],
        overwrite=True,
    )
    files = list(source.fetch())
    assert files and files[0].exists()
    assert captured["dataset"] == "psp"
    assert captured["overwrite"] is True
