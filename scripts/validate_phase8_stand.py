#!/usr/bin/env python3
"""Rerun the Gavin Lake stand validation with the selected empirical model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from nemora.ingest.plots import PlotInventorySpec, build_inventory_release
from nemora.ingest.schema import InventoryProvenance, write_inventory_release
from nemora.ingest.selection import (
    RESEARCH_FOREST_TARGETS,
    assign_holdout_split,
    resolve_tree_detail,
    select_forest_plots,
)
from nemora.validation import validate_empirical_stand

FAIB_PSP_SOURCE_URI = "ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/psp/"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forest", choices=sorted(RESEARCH_FOREST_TARGETS), required=True)
    parser.add_argument("--faib-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=Path("build/phase8"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=20261030)
    parser.add_argument("--stand-seed", type=int, default=20261030)
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    parser.add_argument("--licence", default="OGL-BC")
    return parser.parse_args()


def main() -> None:
    """Run selection, release construction, and empirical validation."""
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
    species = pd.DataFrame(
        {"species_code": sorted(trees["species_code"].dropna().astype(str).unique())}
    )
    for split in ("train", "holdout"):
        split_plots = selected.loc[selected["split"] == split].copy()
        split_trees = trees.loc[trees["plot_id"].isin(set(split_plots["plot_id"]))].copy()
        release = build_inventory_release(
            split_plots,
            split_trees,
            species=species,
            spec=PlotInventorySpec(
                release_id=f"phase8-{args.forest}-{split}",
                provenance=InventoryProvenance(
                    source_id="bc-faib-psp",
                    source_version="public-faib-psp",
                    source_uri=FAIB_PSP_SOURCE_URI,
                    retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    licence=args.licence,
                    notes="Training/hold-out split for Phase 8 validation rerun.",
                ),
                crs="EPSG:3005",
                metadata={
                    "forest": target.name,
                    "split": split,
                    "split_seed": args.split_seed,
                    "holdout_fraction": args.holdout_fraction,
                    "model": "empirical",
                },
            ),
        )
        write_inventory_release(release, args.release_root / args.forest / split)

    training = trees.loc[
        trees["plot_id"].isin(set(selected.loc[selected["split"] == "train", "plot_id"]))
    ]
    holdout = trees.loc[
        trees["plot_id"].isin(set(selected.loc[selected["split"] == "holdout", "plot_id"]))
    ]
    result = validate_empirical_stand(
        training,
        holdout,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.stand_seed,
    )
    payload = result.to_dict()
    payload["forest"] = target.name
    payload["counts"] = {
        "plots_train": int((selected["split"] == "train").sum()),
        "plots_holdout": int((selected["split"] == "holdout").sum()),
        "trees_train": int(len(training)),
        "trees_holdout": int(len(holdout)),
    }
    payload["provenance"] = {
        "source_id": "bc-faib-psp",
        "source_uri": FAIB_PSP_SOURCE_URI,
        "licence": args.licence,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(result.comparison_frame().to_string(index=False))
    print("uncertainty:")
    for key, value in result.uncertainty.items():
        print(key, value)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
