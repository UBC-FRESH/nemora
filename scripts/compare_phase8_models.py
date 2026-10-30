#!/usr/bin/env python3
"""Compare pooled, species-stratified, and empirical stand models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nemora.ingest.selection import (
    RESEARCH_FOREST_TARGETS,
    assign_holdout_split,
    resolve_tree_detail,
    select_forest_plots,
)
from nemora.validation import compare_stand_models


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forest", choices=sorted(RESEARCH_FOREST_TARGETS), required=True)
    parser.add_argument("--faib-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=20261030)
    parser.add_argument("--stand-seed", type=int, default=20261030)
    return parser.parse_args()


def main() -> None:
    """Run the candidate comparison and write the JSON report."""
    args = parse_args()
    target = RESEARCH_FOREST_TARGETS[args.forest]
    site = pd.read_csv(args.faib_root / "faib_header.csv", low_memory=False)
    plot = pd.read_csv(args.faib_root / "faib_plot_header.csv", low_memory=False)
    sample = pd.read_csv(args.faib_root / "faib_sample_byvisit.csv", low_memory=False)
    selected = assign_holdout_split(
        select_forest_plots(site, plot, sample, target=target),
        holdout_fraction=args.holdout_fraction,
        seed=args.split_seed,
    )
    trees = resolve_tree_detail(args.faib_root / "faib_tree_detail.csv", selected)
    training = trees.loc[
        trees["plot_id"].isin(set(selected.loc[selected["split"] == "train", "plot_id"]))
    ]
    holdout = trees.loc[
        trees["plot_id"].isin(set(selected.loc[selected["split"] == "holdout", "plot_id"]))
    ]
    comparison = compare_stand_models(training, holdout, seed=args.stand_seed)
    payload = comparison.to_dict()
    payload["forest"] = target.name
    payload["counts"] = {
        "plots_train": int((selected["split"] == "train").sum()),
        "plots_holdout": int((selected["split"] == "holdout").sum()),
        "trees_train": int(len(training)),
        "trees_holdout": int(len(holdout)),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(comparison.comparison_frame().to_string(index=False))
    print(f"selected: {comparison.selected}")
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
