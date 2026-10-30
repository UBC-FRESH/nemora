#!/usr/bin/env python3
"""Validate Malcolm Knapp using a CWHdm regional training ring."""

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
    resolve_tree_detail,
    select_forest_plots,
)
from nemora.validation import validate_empirical_stand

FAIB_PSP_SOURCE_URI = "ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/psp/"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faib-root", type=Path, required=True)
    parser.add_argument("--release-root", type=Path, default=Path("build/phase9"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--stand-seed", type=int, default=20261030)
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    parser.add_argument("--licence", default="OGL-BC")
    return parser.parse_args()


def main() -> None:
    """Run local/regional selection and validation."""
    args = parse_args()
    site = pd.read_csv(args.faib_root / "faib_header.csv", low_memory=False)
    plot = pd.read_csv(args.faib_root / "faib_plot_header.csv", low_memory=False)
    sample = pd.read_csv(args.faib_root / "faib_sample_byvisit.csv", low_memory=False)
    local = select_forest_plots(site, plot, sample, target=RESEARCH_FOREST_TARGETS["mkrf"])
    regional = select_forest_plots(
        site, plot, sample, target=RESEARCH_FOREST_TARGETS["mkrf_regional"]
    )
    local_trees = resolve_tree_detail(args.faib_root / "faib_tree_detail.csv", local)
    regional_trees = resolve_tree_detail(args.faib_root / "faib_tree_detail.csv", regional)
    all_trees = pd.concat([local_trees, regional_trees], ignore_index=True)
    species = pd.DataFrame(
        {"species_code": sorted(all_trees["species_code"].dropna().astype(str).unique())}
    )
    for name, plots, trees in (
        ("mkrf_local_holdout", local, local_trees),
        ("mkrf_regional_train", regional, regional_trees),
    ):
        release = build_inventory_release(
            plots,
            trees,
            species=species,
            spec=PlotInventorySpec(
                release_id=f"phase9-{name}",
                provenance=InventoryProvenance(
                    source_id="bc-faib-psp",
                    source_version="public-faib-psp",
                    source_uri=FAIB_PSP_SOURCE_URI,
                    retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    licence=args.licence,
                    notes="Phase 9 Malcolm Knapp local/regional validation.",
                ),
                crs="EPSG:3005",
                metadata={"phase": "P9", "sample": name, "model": "empirical"},
            ),
        )
        write_inventory_release(release, args.release_root / name)
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
        "local_plots": int(len(local)),
        "regional_plots": int(len(regional)),
        "local_trees": int(len(local_trees)),
        "regional_trees": int(len(regional_trees)),
        "rule": (
            "Train on CWHdm plots 50-100 km from the forest centre; "
            "hold out the six local mature plots."
        ),
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
    print(result.uncertainty)
    print(f"wrote {args.report}")


if __name__ == "__main__":
    main()
