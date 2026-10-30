#!/usr/bin/env python3
"""Export the selected Gavin Lake empirical stand to a FHOPS tree inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nemora.export import export_fhops_tree_inventory
from nemora.ingest.selection import (
    RESEARCH_FOREST_TARGETS,
    assign_holdout_split,
    resolve_tree_detail,
    select_forest_plots,
)
from nemora.validation import validate_empirical_stand


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faib-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("build/phase9/fhops"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=20261030)
    parser.add_argument("--stand-seed", type=int, default=20261030)
    return parser.parse_args()


def main() -> None:
    """Build the empirical stand and export the FHOPS contract."""
    args = parse_args()
    target = RESEARCH_FOREST_TARGETS["gavin"]
    site = pd.read_csv(args.faib_root / "faib_header.csv", low_memory=False)
    plot = pd.read_csv(args.faib_root / "faib_plot_header.csv", low_memory=False)
    sample = pd.read_csv(args.faib_root / "faib_sample_byvisit.csv", low_memory=False)
    selected = assign_holdout_split(
        select_forest_plots(site, plot, sample, target=target),
        holdout_fraction=0.25,
        seed=args.split_seed,
    )
    trees = resolve_tree_detail(args.faib_root / "faib_tree_detail.csv", selected)
    training = trees.loc[
        trees["plot_id"].isin(set(selected.loc[selected["split"] == "train", "plot_id"]))
    ]
    holdout = trees.loc[
        trees["plot_id"].isin(set(selected.loc[selected["split"] == "holdout", "plot_id"]))
    ]
    stand = validate_empirical_stand(
        training,
        holdout,
        bootstrap_replicates=0,
        seed=args.stand_seed,
    )
    export = export_fhops_tree_inventory(
        stand.synthetic_trees,
        args.output_dir,
        block_id="B01",
        inventory_id="nemora-phase9-gavin-empirical",
        seed=args.stand_seed,
        caveats=(
            "Synthetic plot-conditioned tree inventory; not a surveyed stand or real tree map.",
            "Coordinates are deterministic synthetic placement inside a one-hectare block context.",
            "Empirical resampling from Gavin Lake FAIB training plots.",
        ),
    )
    payload = {
        "forest": target.name,
        "model": stand.model,
        "tree_count": export.tree_count,
        "block_id": export.block_id,
        "inventory_sha256": export.inventory_sha256,
        "inventory_relpath": "build/phase9/fhops/tree_inventory.csv",
        "metadata_relpath": "build/phase9/fhops/tree_inventory.meta.json",
        "training_plots": int((selected["split"] == "train").sum()),
        "holdout_plots": int((selected["split"] == "holdout").sum()),
        "stand_metrics": stand.synthetic.to_dict(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
