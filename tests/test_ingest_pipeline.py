import pandas as pd

from nemora.ingest import DatasetSource, TransformPipeline


def test_dataset_source_fetch_requires_fetcher(tmp_path):
    def _fetch(source: DatasetSource):
        path = tmp_path / f"{source.name}.csv"
        path.write_text("id,value\n1,10\n", encoding="utf-8")
        return [path]

    source = DatasetSource(
        name="demo",
        description="Demo dataset",
        uri="https://example.com/demo.csv",
        metadata={"license": "ODC"},
        fetcher=_fetch,
    )
    artifacts = list(source.fetch())
    assert len(artifacts) == 1
    assert artifacts[0].exists()


def test_transform_pipeline_applies_steps():
    frame = pd.DataFrame({"dbh_cm": [10.0, 20.0], "tally": [5, 7]})
    pipeline = TransformPipeline(name="demo")

    pipeline.add_step(lambda df: df.assign(stand_table=df["tally"] * 2.0))
    pipeline.add_step(lambda df: df.rename(columns={"dbh_cm": "dbh_mm"}))

    result = pipeline.run(frame)
    assert "stand_table" in result.columns
    assert list(result["stand_table"]) == [10.0, 14.0]
    assert "dbh_mm" in result.columns
    assert "dbh_cm" not in result.columns
