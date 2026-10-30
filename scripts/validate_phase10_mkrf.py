#!/usr/bin/env python3
"""Validate the age-stratified CWHdm pool for Malcolm Knapp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nemora.ingest.selection import (
    RESEARCH_FOREST_TARGETS,
    resolve_tree_detail,
    select_forest_plots,
)
from nemora.validation import validate_empirical_stand


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faib-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stand-seed", type=int, default=20261030)
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    """Run the age-stratified transfer validation."""
    args = parse_args()
    site = pd.read_csv(args.faib_root / "faib_header.csv", low_memory=False)
    plot = pd.read_csv(args.faib_root / "faib_plot_header.csv", low_memory=False)
    sample = pd.read_csv(args.faib_root / "faib_sample_byvisit.csv", low_memory=False)
    local = select_forest_plots(site, plot, sample, target=RESEARCH_FOREST_TARGETS["mkrf"])
    regional = select_forest_plots(
        site, plot, sample, target=RESEARCH_FOREST_TARGETS["mkrf_age80_110"]
    )
    local_trees = resolve_tree_detail(args.faib_root / "faib_tree_detail.csv", local)
    regional_trees = resolve_tree_detail(args.faib_root / "faib_tree_detail.csv", regional)
    result = validate_empirical_stand(
        regional_trees,
        local_trees,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.stand_seed,
    )
    payload = result.to_dict()
    payload["forest"] = "Malcolm Knapp Research Forest"
    payload["pooling"] = {
        "local_radius_km": 50.0,
        "regional_radius_km": 100.0,
        "regional_ring_km": [50.0, 100.0],
        "regional_age_years": [80.0, 110.0],
        "local_plots": int(len(local)),
        "regional_plots": int(len(regional)),
        "local_trees": int(len(local_trees)),
        "regional_trees": int(len(regional_trees)),
        "rule": (
            "Train on CWHdm plots 50-100 km from the forest centre "
            "with PROJ_AGE_1 between 80 and 110; hold out the six local plots."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(result.comparison_frame().to_string(index=False))
    print(result.uncertainty)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
