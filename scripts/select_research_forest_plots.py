#!/usr/bin/env python3
"""Select public FAIB plots for a research-forest validation target.

The script reads FAIB site, plot, and visit tables, applies the named target's
BEC/age/radius filters, assigns a deterministic train/hold-out split, and writes
a location-light JSON manifest. It does not download data or write coordinates
to the output manifest.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nemora.ingest.selection import (
    ForestPlotTarget,
    assign_holdout_split,
    select_forest_plots,
    selection_manifest,
    write_selection_manifest,
)

FOREST_TARGETS: dict[str, ForestPlotTarget] = {
    "gavin": ForestPlotTarget(
        name="Alex Fraser Research Forest, Gavin Lake",
        latitude=52.462616,
        longitude=-121.763919,
        radius_km=25.0,
        bec_codes=("ICHmk3", "SBSdw1"),
    ),
    "mkrf": ForestPlotTarget(
        name="Malcolm Knapp Research Forest",
        latitude=49.30663,
        longitude=-122.551375,
        radius_km=50.0,
        bec_codes=("CWHdm",),
    ),
}

FAIB_PSP_SOURCE_URI = "ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/psp/"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forest", choices=sorted(FOREST_TARGETS), required=True)
    parser.add_argument("--site-header", type=Path, required=True)
    parser.add_argument("--plot-header", type=Path, required=True)
    parser.add_argument("--sample-byvisit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.25)
    parser.add_argument("--split-seed", type=int, default=20261030)
    parser.add_argument("--min-age", type=float, default=None)
    parser.add_argument("--licence", default="OGL-BC")
    return parser.parse_args()


def main() -> None:
    """Run the selection and write the manifest."""
    args = parse_args()
    base = FOREST_TARGETS[args.forest]
    target = ForestPlotTarget(
        name=base.name,
        latitude=base.latitude,
        longitude=base.longitude,
        radius_km=base.radius_km,
        bec_codes=base.bec_codes,
        min_projected_age=base.min_projected_age if args.min_age is None else args.min_age,
        sample_type=base.sample_type,
        first_visit_only=base.first_visit_only,
    )
    site_header = pd.read_csv(args.site_header, low_memory=False)
    plot_header = pd.read_csv(args.plot_header, low_memory=False)
    sample_by_visit = pd.read_csv(args.sample_byvisit, low_memory=False)
    selected = select_forest_plots(
        site_header,
        plot_header,
        sample_by_visit,
        target=target,
    )
    split = assign_holdout_split(
        selected,
        holdout_fraction=args.holdout_fraction,
        seed=args.split_seed,
    )
    manifest = selection_manifest(
        split,
        target=target,
        holdout_fraction=args.holdout_fraction,
        split_seed=args.split_seed,
    )
    manifest["provenance"] = {
        "source_id": "bc-faib-psp",
        "source_uri": FAIB_PSP_SOURCE_URI,
        "licence": args.licence,
        "retrieved_utc": datetime.now(UTC).isoformat(),
        "input_files": [
            _file_record(args.site_header),
            _file_record(args.plot_header),
            _file_record(args.sample_byvisit),
        ],
    }
    write_selection_manifest(manifest, args.output)
    print(
        f"{target.name}: {manifest['counts']['total']} plots "
        f"({manifest['counts']['train']} train, {manifest['counts']['holdout']} holdout)"
    )
    print(f"wrote {args.output}")


def _file_record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"name": path.name, "size_bytes": path.stat().st_size, "sha256": digest}


if __name__ == "__main__":
    main()
