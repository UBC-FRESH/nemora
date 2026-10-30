#!/usr/bin/env python3
"""Run independent yield checks and reconcile the Malcolm Knapp scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nemora.fhops_handoff import register_fhops_scenario
from nemora.yield_check import compare_volume_estimates, estimate_inventory_volume


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gavin-inventory", type=Path, required=True)
    parser.add_argument("--mkrf-inventory", type=Path, required=True)
    parser.add_argument("--mkrf-template", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Estimate volumes, reconcile MKRf, and update the catalog."""
    args = parse_args()
    gavin = estimate_inventory_volume(pd.read_csv(args.gavin_inventory))
    mkrf = estimate_inventory_volume(pd.read_csv(args.mkrf_inventory))
    gavin_comparison = compare_volume_estimates(
        gavin.total_volume_m3,
        registered_work_required_m3=530.5433400362283,
        fhops_processed_m3=530.5433400362283,
    )
    mkrf_comparison = compare_volume_estimates(mkrf.total_volume_m3)
    mkrf_registration = register_fhops_scenario(
        args.mkrf_template,
        args.mkrf_template.parent / "scenario",
        args.mkrf_inventory.parent,
        block_id="B02",
        work_required_m3=mkrf.total_volume_m3,
    )
    mkrf_comparison = compare_volume_estimates(
        mkrf.total_volume_m3,
        registered_work_required_m3=mkrf_registration.work_required_m3,
    )
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    for item in catalog["scenarios"]:
        if item["scenario_id"] == "mkrf-local-empirical":
            item["work_required_m3"] = mkrf.total_volume_m3
            item["work_required_source"] = "independent paraboloid estimate (P11.2)"
    catalog["yield_check"] = {
        "gavin_estimate_m3": gavin.total_volume_m3,
        "mkrf_estimate_m3": mkrf.total_volume_m3,
    }
    args.catalog.write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload = {
        "estimator": gavin.to_dict(),
        "gavin": gavin_comparison,
        "mkrf": mkrf_comparison,
        "mkrf_registration": mkrf_registration.to_dict(),
        "caveat": (
            "Paraboloid-style independent check; not a taper model or operational yield estimate."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
