from pathlib import Path

import pandas as pd
import pytest

from nemora.ingest.faib import DataDictionary, load_data_dictionary


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
