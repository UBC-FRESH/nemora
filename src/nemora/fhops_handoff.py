"""Register a FHOPS scenario against an exported tree inventory.

This is an external handoff adapter. It writes FHOPS scenario metadata and
checksums without importing FHOPS or changing its core.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml

__all__ = [
    "FhopsScenarioRegistration",
    "register_fhops_scenario",
    "validate_fhops_scenario_pin",
]


@dataclass(frozen=True, slots=True)
class FhopsScenarioRegistration:
    """Paths and checksums for a registered FHOPS scenario."""

    scenario_path: Path
    inventory_path: Path
    metadata_path: Path
    inventory_sha256: str
    metadata_sha256: str
    block_id: str
    work_required_m3: float | None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary."""
        return {
            "scenario_path": str(self.scenario_path),
            "inventory_path": str(self.inventory_path),
            "metadata_path": str(self.metadata_path),
            "inventory_sha256": self.inventory_sha256,
            "metadata_sha256": self.metadata_sha256,
            "block_id": self.block_id,
            "work_required_m3": self.work_required_m3,
        }


def register_fhops_scenario(
    scenario_template: str | Path,
    output_dir: str | Path,
    inventory_dir: str | Path,
    *,
    block_id: str = "B01",
    work_required_m3: float | None = None,
) -> FhopsScenarioRegistration:
    """Copy a scenario template and pin an exported tree inventory.

    Parameters
    ----------
    scenario_template:
        Directory containing ``scenario.yaml`` and its ``data/`` files.
    output_dir:
        Destination scenario directory. It is replaced if it exists.
    inventory_dir:
        Directory containing ``tree_inventory.csv`` and
        ``tree_inventory.meta.json``.
    block_id:
        FHOPS block to reconcile in ``data/blocks.csv``.
    work_required_m3:
        Optional replacement for the block's ``work_required`` value.

    Returns
    -------
    FhopsScenarioRegistration
        Registered paths and checksums.

    Raises
    ------
    FileNotFoundError
        If an expected input file is missing.
    ValueError
        If the inventory metadata is inconsistent or the block is absent.
    """
    template = Path(scenario_template)
    target = Path(output_dir)
    inventory_dir = Path(inventory_dir)
    inventory_path = inventory_dir / "tree_inventory.csv"
    metadata_path = inventory_dir / "tree_inventory.meta.json"
    scenario_path = target / "scenario.yaml"
    for path in (template / "scenario.yaml", inventory_path, metadata_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(template, target)
    inventory_sha = _sha256(inventory_path)
    metadata_sha = _sha256(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("inventory_sha256") not in (None, inventory_sha):
        raise ValueError("inventory metadata checksum does not match the CSV")
    tree_count = metadata.get("tree_count")
    block_ids = metadata.get("block_ids") or [block_id]
    if block_id not in block_ids:
        raise ValueError(f"block_id {block_id!r} is not declared in inventory metadata")
    payload = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("scenario.yaml must contain a mapping")
    inventory_rel = os.path.relpath(inventory_path.resolve(), target.resolve())
    metadata_rel = os.path.relpath(metadata_path.resolve(), target.resolve())
    payload["tree_inventory"] = {
        "inventory_id": metadata.get("inventory_id", "nemora-tree-inventory"),
        "inventory_sha256": inventory_sha,
        "metadata_sha256": metadata_sha,
        "inventory_path": inventory_rel,
        "metadata_path": metadata_rel,
        "crs": metadata.get("crs", "EPSG:3005"),
        "review_status": metadata.get("review_status", "not_reviewed"),
        "source_id": metadata.get("source_id", "nemora"),
        "producer": metadata.get("producer", "Nemora"),
        "method": metadata.get("method", ""),
        "tree_count": tree_count,
        "block_ids": block_ids,
        "caveats": metadata.get("caveats", []),
    }
    scenario_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    if work_required_m3 is not None:
        _update_block_work_required(target / "data" / "blocks.csv", block_id, work_required_m3)
    return FhopsScenarioRegistration(
        scenario_path=scenario_path,
        inventory_path=inventory_path,
        metadata_path=metadata_path,
        inventory_sha256=inventory_sha,
        metadata_sha256=metadata_sha,
        block_id=block_id,
        work_required_m3=work_required_m3,
    )


def validate_fhops_scenario_pin(scenario_path: str | Path) -> bool:
    """Return whether a scenario's tree inventory pin matches its files."""
    path = Path(scenario_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    pin = payload.get("tree_inventory") if isinstance(payload, dict) else None
    if not isinstance(pin, dict):
        return False
    inventory_path = path.parent / str(pin.get("inventory_path", ""))
    metadata_path = path.parent / str(pin.get("metadata_path", ""))
    if not inventory_path.exists() or not metadata_path.exists():
        return False
    return _sha256(inventory_path) == pin.get("inventory_sha256") and _sha256(
        metadata_path
    ) == pin.get("metadata_sha256")


def _update_block_work_required(path: Path, block_id: str, work_required: float) -> None:
    if work_required < 0 or not isinstance(work_required, int | float):
        raise ValueError("work_required_m3 must be non-negative")
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if not rows:
        raise ValueError("blocks.csv is empty")
    if "work_required" not in rows[0]:
        raise ValueError("blocks.csv has no work_required column")
    found = False
    for row in rows:
        if row.get("id") == block_id:
            row["work_required"] = f"{float(work_required):.6f}"
            row["volume_per_ha_m3"] = f"{float(work_required):.6f}"
            found = True
    if not found:
        raise ValueError(f"block {block_id!r} not found in blocks.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
