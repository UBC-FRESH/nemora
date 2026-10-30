#!/usr/bin/env python3
"""Register a FHOPS scenario against the exported Phase 9 tree inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nemora.fhops_handoff import register_fhops_scenario, validate_fhops_scenario_pin


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario-template", type=Path, required=True)
    parser.add_argument("--inventory-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-id", default="B01")
    parser.add_argument("--work-required-m3", type=float, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """Register the scenario and write a summary."""
    args = parse_args()
    result = register_fhops_scenario(
        args.scenario_template,
        args.output_dir,
        args.inventory_dir,
        block_id=args.block_id,
        work_required_m3=args.work_required_m3,
    )
    valid = validate_fhops_scenario_pin(result.scenario_path)
    payload = result.to_dict()
    payload["pin_valid"] = valid
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
