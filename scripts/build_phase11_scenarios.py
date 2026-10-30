#!/usr/bin/env python3
"""Build the Phase 11 Gavin and Malcolm Knapp FHOPS scenario set."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nemora.export import export_fhops_tree_inventory
from nemora.fhops_handoff import register_fhops_scenario, validate_fhops_scenario_pin
from nemora.ingest.selection import (
    RESEARCH_FOREST_TARGETS,
    resolve_tree_detail,
    select_forest_plots,
)
from nemora.validation import generate_empirical_stand


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faib-root", type=Path, required=True)
    parser.add_argument("--scenario-template", type=Path, required=True)
    parser.add_argument("--gavin-inventory-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("build/phase11"))
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261030)
    return parser.parse_args()


def main() -> None:
    """Build both scenario directories and the catalog."""
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    scenarios = []

    gavin_scenario = args.output_root / "gavin" / "scenario"
    gavin_registration = register_fhops_scenario(
        args.scenario_template,
        gavin_scenario,
        args.gavin_inventory_dir,
        block_id="B01",
        work_required_m3=530.5433400362283,
    )
    scenarios.append(
        _catalog_entry(
            "gavin-empirical",
            "B01",
            "L1",
            gavin_registration.scenario_path,
            gavin_registration.inventory_path,
            gavin_registration.inventory_sha256,
            gavin_registration.work_required_m3,
            "Gavin Lake P8 empirical stand; one-hectare synthetic placement.",
        )
    )

    site = pd.read_csv(args.faib_root / "faib_header.csv", low_memory=False)
    plot = pd.read_csv(args.faib_root / "faib_plot_header.csv", low_memory=False)
    sample = pd.read_csv(args.faib_root / "faib_sample_byvisit.csv", low_memory=False)
    local = select_forest_plots(site, plot, sample, target=RESEARCH_FOREST_TARGETS["mkrf"])
    local_trees = resolve_tree_detail(args.faib_root / "faib_tree_detail.csv", local)
    mkrf_trees = generate_empirical_stand(local_trees, seed=args.seed)
    mkrf_inventory = args.output_root / "mkrf" / "inventory"
    export_fhops_tree_inventory(
        mkrf_trees,
        mkrf_inventory,
        block_id="B02",
        inventory_id="nemora-phase11-mkrf-local-empirical",
        seed=args.seed,
        caveats=(
            "Synthetic plot-conditioned tree inventory; not a surveyed stand or real tree map.",
            "Malcolm Knapp local six-plot empirical scenario; no independent hold-out validation.",
        ),
    )
    mkrf_template = args.output_root / "mkrf" / "template"
    _clone_template(args.scenario_template, mkrf_template, "B02", "L2")
    mkrf_scenario = args.output_root / "mkrf" / "scenario"
    mkrf_registration = register_fhops_scenario(
        mkrf_template,
        mkrf_scenario,
        mkrf_inventory,
        block_id="B02",
        work_required_m3=None,
    )
    scenarios.append(
        _catalog_entry(
            "mkrf-local-empirical",
            "B02",
            "L2",
            mkrf_registration.scenario_path,
            mkrf_registration.inventory_path,
            mkrf_registration.inventory_sha256,
            None,
            "Malcolm Knapp local six-plot empirical scenario; synthetic transfer case.",
        )
    )

    payload = {
        "schema_version": "nemora.scenario_catalog.v1",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scenarios": scenarios,
        "notes": [
            "FHOPS ground system runs one block/one landing at a time.",
            "Gavin is the primary validation block; Malcolm Knapp is a synthetic transfer case.",
            "Scenario and inventory bytes are local build artifacts; checksums identify them.",
        ],
    }
    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    args.catalog.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


def _catalog_entry(
    scenario_id: str,
    block_id: str,
    landing_id: str,
    scenario_path: Path,
    inventory_path: Path,
    inventory_sha256: str,
    work_required_m3: float | None,
    notes: str,
) -> dict[str, object]:
    return {
        "scenario_id": scenario_id,
        "block_id": block_id,
        "landing_id": landing_id,
        "scenario_path": str(scenario_path),
        "inventory_path": str(inventory_path),
        "inventory_sha256": inventory_sha256,
        "work_required_m3": work_required_m3,
        "pin_valid": validate_fhops_scenario_pin(scenario_path),
        "notes": notes,
    }


def _clone_template(template: Path, output: Path, block_id: str, landing_id: str) -> None:
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(template, output)
    _rename_block(output / "data" / "blocks.csv", block_id, landing_id)
    _rename_landing(output / "data" / "landings.csv", landing_id)
    _rename_prod_rate_block(output / "data" / "prod_rates.csv", block_id)


def _rename_block(path: Path, block_id: str, landing_id: str) -> None:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    for row in rows:
        row["id"] = block_id
        row["landing_id"] = landing_id
    _write_csv(path, rows)


def _rename_landing(path: Path, landing_id: str) -> None:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    for row in rows:
        row["id"] = landing_id
    _write_csv(path, rows)


def _rename_prod_rate_block(path: Path, block_id: str) -> None:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    for row in rows:
        row["block_id"] = block_id
    _write_csv(path, rows)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"{path} is empty")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
