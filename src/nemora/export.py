"""Export adapters for external consumers.

The FHOPS adapter writes a standalone individual-tree inventory. It does not
import FHOPS or create a FHOPS scenario; FHOPS remains a consumer of the
exported contract.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .synthesis.stems import TreePlacementConfig, place_trees

__all__ = ["FhopsTreeExport", "export_fhops_tree_inventory"]

_TREE_REQUIRED = ("tree_id", "species_code", "dbh_cm", "height_m")
_FHOPS_COLUMNS = (
    "tree_id",
    "block_id",
    "x_m",
    "y_m",
    "elevation_m",
    "species_code",
    "dbh_cm",
    "height_m",
    "crown_radius_m",
    "status",
    "confidence",
    "source_id",
    "source_tile_id",
    "merchantable_height_m",
    "defect",
    "grade_hint",
)
_FHOPS_UNITS = {
    "x_y": "m",
    "elevation": "m",
    "dbh": "cm",
    "height": "m",
    "crown_radius": "m",
}


@dataclass(frozen=True, slots=True)
class FhopsTreeExport:
    """Paths and provenance for one FHOPS tree-inventory export."""

    inventory_path: Path
    metadata_path: Path
    inventory_sha256: str
    tree_count: int
    block_id: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable export summary."""
        return {
            "inventory_path": str(self.inventory_path),
            "metadata_path": str(self.metadata_path),
            "inventory_sha256": self.inventory_sha256,
            "tree_count": self.tree_count,
            "block_id": self.block_id,
        }


def export_fhops_tree_inventory(
    trees: pd.DataFrame,
    output_dir: str | Path,
    *,
    block_id: str = "GAVIN_EMPIRICAL",
    inventory_id: str = "nemora-phase9-gavin-empirical",
    crs: str = "EPSG:3005",
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    size_m: float = 100.0,
    seed: int = 20261030,
    producer: str = "Nemora",
    method: str = "Plot-conditioned empirical resampling; synthetic placement.",
    source_id: str = "bc-faib-psp",
    caveats: tuple[str, ...] = (
        "Synthetic plot-conditioned tree inventory; not a surveyed stand or real tree map.",
        "Coordinates are deterministic synthetic placement inside a one-hectare block context.",
    ),
) -> FhopsTreeExport:
    """Write a FHOPS ``fhops.tree.inventory.v1`` CSV and metadata file.

    Parameters
    ----------
    trees:
        Synthetic tree rows with ``tree_id``, ``species_code``, ``dbh_cm``, and
        ``height_m``. Extra columns are ignored.
    output_dir:
        Directory for ``tree_inventory.csv`` and ``tree_inventory.meta.json``.
    block_id:
        FHOPS block identifier.
    inventory_id:
        Stable inventory identifier.
    crs:
        CRS label for the synthetic coordinates.
    origin_x, origin_y, size_m:
        Synthetic square block context in the export CRS.
    seed:
        Deterministic placement seed.
    producer, method, source_id, caveats:
        Provenance fields written to the metadata.

    Raises
    ------
    KeyError
        If a required tree column is missing.
    ValueError
        If the input is empty or ``size_m`` is not positive.
    """
    missing = [column for column in _TREE_REQUIRED if column not in trees.columns]
    if missing:
        raise KeyError("trees is missing required columns: " + ", ".join(missing))
    if trees.empty:
        raise ValueError("cannot export an empty tree inventory")
    if size_m <= 0.0 or not np.isfinite(size_m):
        raise ValueError("size_m must be finite and positive")
    if not block_id.strip():
        raise ValueError("block_id must not be blank")

    frame = trees.loc[:, list(_TREE_REQUIRED)].copy()
    frame["tree_id"] = frame["tree_id"].astype(str)
    if not frame["tree_id"].is_unique:
        raise ValueError("tree_id values must be unique")
    frame["species_code"] = frame["species_code"].astype(str)
    frame["dbh_cm"] = pd.to_numeric(frame["dbh_cm"], errors="coerce")
    frame["height_m"] = pd.to_numeric(frame["height_m"], errors="coerce")
    if not np.isfinite(frame["dbh_cm"]).all() or not np.isfinite(frame["height_m"]).all():
        raise ValueError("dbh_cm and height_m must be finite")
    if (frame["dbh_cm"] <= 0).any() or (frame["height_m"] <= 0).any():
        raise ValueError("dbh_cm and height_m must be positive")

    polygon = np.asarray(
        [
            [origin_x, origin_y],
            [origin_x + size_m, origin_y],
            [origin_x + size_m, origin_y + size_m],
            [origin_x, origin_y + size_m],
        ],
        dtype=float,
    )
    points = place_trees(
        polygon,
        len(frame),
        rng=np.random.default_rng(seed),
        config=TreePlacementConfig(mode="poisson", min_spacing=0.0),
    )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    inventory_path = target / "tree_inventory.csv"
    metadata_path = target / "tree_inventory.meta.json"

    with inventory_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(_FHOPS_COLUMNS), lineterminator="\n")
        writer.writeheader()
        rows = frame.reset_index(drop=True).itertuples(index=False, name=None)
        for index, values in enumerate(rows):
            point = points[index]
            writer.writerow(
                {
                    "tree_id": values[0],
                    "block_id": block_id,
                    "x_m": f"{float(point[0]):.6f}",
                    "y_m": f"{float(point[1]):.6f}",
                    "elevation_m": "",
                    "species_code": values[1],
                    "dbh_cm": f"{float(values[2]):.6f}",
                    "height_m": f"{float(values[3]):.6f}",
                    "crown_radius_m": "",
                    "status": "live",
                    "confidence": "1.0",
                    "source_id": source_id,
                    "source_tile_id": "",
                    "merchantable_height_m": "",
                    "defect": "",
                    "grade_hint": "",
                }
            )
    digest = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    metadata = {
        "schema_version": "fhops.tree.inventory.v1",
        "inventory_id": inventory_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_id": source_id,
        "producer": producer,
        "method": method,
        "crs": crs,
        "units": dict(_FHOPS_UNITS),
        "review_status": "not_reviewed",
        "reviewed_by": None,
        "block_ids": [block_id],
        "caveats": list(caveats),
        "tree_count": len(frame),
        "inventory_sha256": digest,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return FhopsTreeExport(
        inventory_path=inventory_path,
        metadata_path=metadata_path,
        inventory_sha256=digest,
        tree_count=len(frame),
        block_id=block_id,
    )
