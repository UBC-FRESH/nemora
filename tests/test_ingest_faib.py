from pathlib import Path

import pandas as pd
import pytest

from nemora.ingest.faib import (
    NON_PSP_DICTIONARY_URL,
    PSP_DICTIONARY_URL,
    DataDictionary,
    aggregate_stand_table,
    build_stand_table_from_csvs,
    download_faib_csvs,
    load_data_dictionary,
    load_non_psp_dictionary,
    load_psp_dictionary,
)


def test_manifest_records_exist() -> None:
    manifest_path = Path("examples/faib_manifest/faib_manifest.csv")
    manifest = pd.read_csv(manifest_path)
    assert not manifest.empty
    for rel_path in manifest["path"]:
        assert (manifest_path.parent / rel_path).exists()


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
    sample_byvisit = pd.DataFrame(
        {
            "CLSTR_ID": ["A", "B"],
            "VISIT_NUMBER": [1, 1],
            "PLOT": [1, 1],
            "BAF": [12.0, 8.0],
        }
    )

    stand_table = aggregate_stand_table(tree_detail, sample_byvisit, baf=12.0)
    assert list(stand_table["dbh_cm"]) == [12.0, 25.0]
    assert list(stand_table["tally"]) == [3.0, 2.0]

    empty = aggregate_stand_table(tree_detail, sample_byvisit, baf=20.0)
    assert empty.empty


def test_build_stand_table_from_csvs(tmp_path: Path) -> None:
    fixture_root = Path("tests/fixtures/faib")
    result = build_stand_table_from_csvs(fixture_root, baf=12.0)
    assert list(result["dbh_cm"]) == [12.0, 13.0, 25.0]


def test_download_faib_csvs(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    listing = (
        b"05-01-25  09:00AM  12345 faib_tree_detail.csv\n"
        b"05-01-25  09:00AM  67890 faib_sample_byvisit.csv\n"
        b"05-01-25  09:00AM  11111 readme.txt\n"
    )
    file_data = {
        "faib_tree_detail.csv": b"CLSTR_ID,VISIT_NUMBER,PLOT,DBH_CM,TREE_EXP\nA,1,1,12,2\n",
        "faib_sample_byvisit.csv": b"CLSTR_ID,VISIT_NUMBER,PLOT,BAF\nA,1,1,12\n",
    }

    class FakeResponse:
        def __init__(self, data: bytes) -> None:
            self._data = data

        def read(self) -> bytes:
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    urls_requested: list[str] = []

    def fake_urlopen(url: str, *args, **kwargs):
        urls_requested.append(url)
        if url.endswith("psp/"):
            return FakeResponse(listing)
        name = url.split("/")[-1]
        return FakeResponse(file_data.get(name, b""))

    monkeypatch.setattr("nemora.ingest.faib.urlopen", fake_urlopen)

    downloaded = download_faib_csvs(tmp_path, dataset="psp")
    assert len(downloaded) == 2
    assert (tmp_path / "faib_tree_detail.csv").exists()
    assert urls_requested[0].endswith("psp/")
