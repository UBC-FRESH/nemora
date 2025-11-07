#!/usr/bin/env python3
"""Generate trimmed FAIB stand table manifests for testing/documentation."""

from __future__ import annotations

import argparse
from argparse import BooleanOptionalAction
from pathlib import Path

import pandas as pd

from nemora.ingest.faib import FAIBManifestResult, generate_faib_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="Output directory for samples and manifest.")
    parser.add_argument(
        "--dataset",
        default="psp",
        choices=["psp", "non_psp"],
        help="FAIB dataset to process.",
    )
    parser.add_argument(
        "--bafs",
        type=float,
        nargs="+",
        help="Explicit BAF values to generate stand tables for.",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-select representative BAF values from the dataset metadata.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Optional existing FAIB download directory (skip fetch when provided).",
    )
    parser.add_argument(
        "--fetch",
        action=BooleanOptionalAction,
        default=None,
        help=(
            "Download FAIB CSV files before building the manifest "
            "(defaults to fetch when no source dir)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action=BooleanOptionalAction,
        default=False,
        help="Force re-download of FAIB CSV files when --fetch is enabled.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit the number of rows retained for each generated stand table.",
    )
    parser.add_argument(
        "--auto-count",
        type=int,
        default=3,
        help="Number of representative BAFs to auto-select when --auto is passed.",
    )
    args = parser.parse_args()
    if args.auto and args.bafs:
        parser.error("--auto can not be combined with --bafs")

    output_dir = args.destination
    output_dir.mkdir(parents=True, exist_ok=True)

    fetch_flag = args.fetch
    if fetch_flag is None:
        fetch_flag = args.source is None

    manifest: FAIBManifestResult = generate_faib_manifest(
        output_dir,
        dataset=args.dataset,
        source=args.source,
        fetch=bool(fetch_flag),
        overwrite=args.overwrite,
        bafs=None if args.auto else args.bafs,
        auto_count=args.auto_count if args.auto else None,
        max_rows=args.max_rows,
    )

    df = pd.read_csv(manifest.manifest_path)
    print(f"Manifest written to {manifest.manifest_path} (rows={len(df)})")
    if manifest.downloaded:
        print(f"Downloaded {len(manifest.downloaded)} files into {manifest.downloaded[0].parent}")


if __name__ == "__main__":
    main()
