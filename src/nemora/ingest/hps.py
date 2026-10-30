"""Pipelines for preparing HPS tallies from FAIB PSP compilations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ..dataprep import PlotSelection, SelectionCriteria, aggregate_hps_tallies, load_plot_selections
from . import TransformPipeline
from .schema import (
    PLOT_COLUMNS,
    SPECIES_COLUMNS,
    TREE_COLUMNS,
    InventoryProvenance,
    InventoryRelease,
    InventoryUnits,
    validate_inventory_release,
    write_inventory_release,
)

DEFAULT_PLOT_HEADER = "faib_plot_header.csv"
DEFAULT_SAMPLE_BYVISIT = "faib_sample_byvisit.csv"
DEFAULT_TREE_DETAIL = "faib_tree_detail.csv"


@dataclass(slots=True)
class HPSPipelineResult:
    """Container produced by :func:`run_hps_pipeline`."""

    tallies: dict[str, pd.DataFrame]
    manifest: pd.DataFrame
    tallies_frame: pd.DataFrame

    def to_inventory_release(
        self,
        *,
        release_id: str,
        provenance: InventoryProvenance | None = None,
        units: InventoryUnits | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> InventoryRelease:
        """Return a schema-validated ``nemora.inventory.v1`` release."""
        return inventory_release_from_hps(
            self,
            release_id=release_id,
            provenance=provenance,
            units=units,
            metadata=metadata,
        )


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


def inventory_release_from_hps(
    result: HPSPipelineResult,
    *,
    release_id: str,
    provenance: InventoryProvenance | None = None,
    units: InventoryUnits | None = None,
    metadata: dict[str, Any] | None = None,
) -> InventoryRelease:
    """Adapt an HPS pipeline result to the shared inventory schema.

    HPS tallies are DBH-class aggregates, not individual trees. The canonical
    ``trees`` table therefore uses a null ``tree_id``, a null ``species_code``,
    and the tally count for each DBH class. The ``plots`` table carries the HPS
    manifest fields.

    Parameters
    ----------
    result:
        Output of :func:`run_hps_pipeline`.
    release_id:
        Stable identifier for the release.
    provenance:
        Release provenance. Defaults to the public FAIB PSP source with a
        current UTC retrieval timestamp.
    units:
        Canonical units. Defaults to centimetres, metres, counts, and
        ``m2_per_ha``.
    metadata:
        Additional JSON-serializable release metadata.
    """
    plots = result.manifest.copy()
    plots = plots.rename(columns={"plot": "plot_number", "trees": "tree_count"})
    if "measurement_date" in plots.columns:
        plots["measurement_date"] = plots["measurement_date"].map(
            lambda value: value.isoformat() if hasattr(value, "isoformat") else value
        )
    plots = plots.reindex(columns=list(PLOT_COLUMNS))

    trees = result.tallies_frame.copy()
    trees = trees.assign(
        tree_id=pd.Series(pd.NA, index=trees.index, dtype="string"),
        species_code=pd.Series(pd.NA, index=trees.index, dtype="string"),
        height_m=float("nan"),
        status="live",
    )
    trees = trees.reindex(columns=list(TREE_COLUMNS))

    release_metadata = {"tally_grain": "dbh_class", "tally_units": "count"}
    if metadata is not None:
        release_metadata.update(metadata)

    release = InventoryRelease(
        release_id=release_id,
        provenance=provenance
        or InventoryProvenance(
            source_id="bc-faib-psp",
            source_version="unspecified",
            source_uri="https://bcgov-env.shinyapps.io/FAIB_GROUND_SAMPLE/",
            retrieved_utc=datetime.now(timezone.utc).isoformat(),
            licence="Public",
            notes="HPS tallies derived from FAIB PSP compilations.",
        ),
        plots=plots,
        trees=trees,
        species=pd.DataFrame(columns=list(SPECIES_COLUMNS)),
        units=units or InventoryUnits(),
        metadata=release_metadata,
    )
    validate_inventory_release(release)
    return release


def export_hps_inventory(
    result: HPSPipelineResult,
    output_dir: str | Path,
    *,
    release_id: str,
    provenance: InventoryProvenance | None = None,
    units: InventoryUnits | None = None,
    metadata: dict[str, Any] | None = None,
) -> InventoryRelease:
    """Write a schema-validated HPS release directory and return it.

    This is additive to :func:`export_hps_outputs`: the legacy per-plot CSVs and
    manifest keep their historical bytes, while the canonical release adds
    ``release.json``, ``plots.csv``, ``trees.csv``, and ``species.csv``.
    """
    release = inventory_release_from_hps(
        result,
        release_id=release_id,
        provenance=provenance,
        units=units,
        metadata=metadata,
    )
    write_inventory_release(release, output_dir)
    return release


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
    "export_hps_inventory",
    "export_hps_outputs",
    "inventory_release_from_hps",
    "run_hps_pipeline",
    "load_plot_selections",
    "SelectionCriteria",
]
