"""Pipelines for preparing HPS tallies from FAIB PSP compilations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..dataprep import PlotSelection, SelectionCriteria, aggregate_hps_tallies, load_plot_selections
from . import TransformPipeline

DEFAULT_PLOT_HEADER = "faib_plot_header.csv"
DEFAULT_SAMPLE_BYVISIT = "faib_sample_byvisit.csv"
DEFAULT_TREE_DETAIL = "faib_tree_detail.csv"


@dataclass(slots=True)
class HPSPipelineResult:
    """Container produced by :func:`run_hps_pipeline`."""

    tallies: dict[str, pd.DataFrame]
    manifest: pd.DataFrame
    tallies_frame: pd.DataFrame


def build_hps_pipeline(
    tree_detail_source: str | Path,
    selections: list[PlotSelection],
    *,
    dbh_column: str = "DBH",
    status_column: str | None = "LV_D",
    live_status: tuple[str, ...] = ("L",),
    bin_width: float = 1.0,
    bin_origin: float = 0.0,
    chunk_size: int = 200_000,
    encoding: str = "latin1",
) -> TransformPipeline:
    """Return a pipeline that aggregates HPS tallies for ``selections``."""

    metadata: dict[str, Any] = {
        "tree_detail_source": str(tree_detail_source),
        "dbh_column": dbh_column,
        "status_column": status_column,
        "live_status": live_status,
        "bin_width": bin_width,
        "bin_origin": bin_origin,
        "chunk_size": chunk_size,
        "encoding": encoding,
        "manifest": pd.DataFrame(),
        "tallies": {},
    }

    live_status = tuple(live_status) if live_status else ()

    def _aggregate(_: pd.DataFrame) -> pd.DataFrame:
        tallies, manifest = aggregate_hps_tallies(
            tree_detail_source,
            selections,
            dbh_column=dbh_column,
            status_column=status_column,
            live_status=live_status,
            bin_width=bin_width,
            bin_origin=bin_origin,
            chunk_size=chunk_size,
            encoding=encoding,
        )
        metadata["manifest"] = manifest
        metadata["tallies"] = tallies
        frames: list[pd.DataFrame] = []
        for plot_id, frame in tallies.items():
            if frame.empty:
                continue
            enriched = frame.copy()
            enriched.insert(0, "plot_id", plot_id)
            frames.append(enriched)
        if not frames:
            return pd.DataFrame(columns=["plot_id", "dbh_cm", "tally"])
        combined = pd.concat(frames, ignore_index=True)
        return combined.loc[:, ["plot_id", "dbh_cm", "tally"]]

    def _sort(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        return frame.sort_values(["plot_id", "dbh_cm"]).reset_index(drop=True)

    pipeline = TransformPipeline(name="faib-hps-tallies", metadata=metadata)
    pipeline.add_step(_aggregate)
    pipeline.add_step(_sort)
    return pipeline


def run_hps_pipeline(
    tree_detail_source: str | Path,
    selections: list[PlotSelection],
    *,
    dbh_column: str = "DBH",
    status_column: str | None = "LV_D",
    live_status: tuple[str, ...] = ("L",),
    bin_width: float = 1.0,
    bin_origin: float = 0.0,
    chunk_size: int = 200_000,
    encoding: str = "latin1",
) -> HPSPipelineResult:
    """Execute the HPS pipeline and return tallies/manifest dataframes."""

    pipeline = build_hps_pipeline(
        tree_detail_source,
        selections,
        dbh_column=dbh_column,
        status_column=status_column,
        live_status=live_status,
        bin_width=bin_width,
        bin_origin=bin_origin,
        chunk_size=chunk_size,
        encoding=encoding,
    )
    selections_frame = pd.DataFrame([sel.__dict__ for sel in selections])
    tallies_frame = pipeline.run(selections_frame if not selections_frame.empty else pd.DataFrame())
    tallies = pipeline.metadata.get("tallies", {})
    manifest = pipeline.metadata.get("manifest", pd.DataFrame())
    return HPSPipelineResult(
        tallies=tallies,
        manifest=manifest,
        tallies_frame=tallies_frame,
    )


def export_hps_outputs(
    tallies: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    *,
    output_dir: Path,
    manifest_path: Path,
    quiet: bool = False,
) -> None:
    """Write per-plot tallies and accompanying manifest to disk."""

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    for plot_id, frame in tallies.items():
        destination = output_dir / f"{plot_id}.csv"
        frame.to_csv(destination, index=False)
        if not quiet:
            print(f"[write] {destination}")

    manifest_copy = manifest.copy()
    if "measurement_date" in manifest_copy.columns:
        manifest_copy["measurement_date"] = manifest_copy["measurement_date"].apply(
            lambda value: value.isoformat() if value else ""
        )
    manifest_copy.to_csv(manifest_path, index=False)
    if not quiet:
        print(f"[manifest] {manifest_path}")


__all__ = [
    "DEFAULT_PLOT_HEADER",
    "DEFAULT_SAMPLE_BYVISIT",
    "DEFAULT_TREE_DETAIL",
    "HPSPipelineResult",
    "build_hps_pipeline",
    "export_hps_outputs",
    "run_hps_pipeline",
    "load_plot_selections",
    "SelectionCriteria",
]
