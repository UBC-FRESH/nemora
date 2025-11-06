#!/usr/bin/env python3
"""Prepare HPS tally datasets from the BC FAIB PSP compilations.

This script automates the steps described in ``docs/howto/hps_dataset.md``.
It downloads (or consumes) the PSP metadata tables, filters them to the desired
plot visits, and aggregates DBH tallies suitable for the HPS workflow demos.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from ftplib import FTP
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd

from nemora.dataprep import (
    SelectionCriteria,
    aggregate_hps_tallies,
    load_plot_selections,
)

DEFAULT_BASE_URL = "ftp://ftp.for.gov.bc.ca/HTS/external/!publish/ground_plot_compilations/psp"
DEFAULT_PLOT_HEADER = "faib_plot_header.csv"
DEFAULT_SAMPLE_BYVISIT = "faib_sample_byvisit.csv"
DEFAULT_TREE_DETAIL = "faib_tree_detail.csv"


def resolve_source(spec: str, base_url: str) -> str:
    """Resolve the supplied specification against the base URL."""
    parsed = urlparse(spec)
    if parsed.scheme:
        return spec
    path = Path(spec)
    if path.exists():
        return str(path)
    return f"{base_url.rstrip('/')}/{spec.lstrip('/')}"


def _download_ftp(url: str, destination: Path, *, quiet: bool) -> None:
    """Download an FTP resource using ftplib with latin-1 encoding."""
    parsed = urlparse(url)
    if parsed.scheme != "ftp":
        raise ValueError("Only FTP URLs may be passed to _download_ftp.")
    path_parts = Path(parsed.path)
    directories = path_parts.parent.as_posix().lstrip("/")
    filename = path_parts.name
    host = parsed.hostname
    if host is None:
        raise ValueError(f"FTP URL missing hostname: {url}")
    if not quiet:
        print(f"[ftp] Connecting to {parsed.hostname}", file=sys.stderr)
    with FTP() as ftp:  # noqa: S321 - FAIB publishes data via anonymous FTP only.
        ftp.encoding = "latin-1"
        ftp.connect(host, parsed.port or 21)
        user = parsed.username or "anonymous"
        password = parsed.password or "anonymous@"
        ftp.login(user, password)
        if directories:
            for part in directories.split("/"):
                if part:
                    ftp.cwd(part)
        if not quiet:
            print(f"[ftp] RETR {filename}", file=sys.stderr)
        with open(destination, "wb") as handle:
            ftp.retrbinary(f"RETR {filename}", handle.write)


def download_to_cache(
    url: str, cache_dir: Path, *, overwrite: bool = False, quiet: bool = False
) -> Path:
    """Download the given URL into the cache directory and return the local path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(urlparse(url).path).name
    destination = cache_dir / filename
    if destination.exists() and not overwrite:
        if not quiet:
            print(f"[cache] Using existing {destination}", file=sys.stderr)
        return destination

    if not quiet:
        print(f"[download] {url} -> {destination}", file=sys.stderr)
    parsed = urlparse(url)
    if parsed.scheme == "ftp":
        _download_ftp(url, destination, quiet=quiet)
    else:
        with urlopen(url) as response, open(destination, "wb") as handle:  # noqa: S310
            chunk_size = 1024 * 1024
            total = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                total += len(chunk)
                if not quiet:
                    progress_mib = total / (1024 * 1024)
                    progress_msg = f"\r[download] {progress_mib:.1f} MiB"
                    print(progress_msg, end="", file=sys.stderr)
        if not quiet:
            print(file=sys.stderr)
    return destination


def ensure_local_path(
    source: str,
    *,
    cache_dir: Path,
    overwrite: bool,
    quiet: bool,
    skip_download: bool,
) -> str:
    """Return a filesystem path for the source, downloading when required."""
    parsed = urlparse(source)
    if parsed.scheme in {"ftp", "http", "https"}:
        if skip_download:
            return source
        return str(download_to_cache(source, cache_dir, overwrite=overwrite, quiet=quiet))

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"{source} does not exist and is not a recognised URL.")
    return str(path)


def write_outputs(
    tallies: dict[str, pd.DataFrame],
    metadata: pd.DataFrame,
    *,
    output_dir: Path,
    manifest_path: Path,
    quiet: bool,
) -> None:
    """Persist per-plot tallies and manifest metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    for plot_id, frame in tallies.items():
        destination = output_dir / f"{plot_id}.csv"
        frame.to_csv(destination, index=False)
        if not quiet:
            print(f"[write] {destination}", file=sys.stderr)

    manifest = metadata.copy()
    manifest["measurement_date"] = manifest["measurement_date"].apply(
        lambda value: value.isoformat() if value else ""
    )
    manifest.to_csv(manifest_path, index=False)
    if not quiet:
        print(f"[manifest] {manifest_path}", file=sys.stderr)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="Base FTP/HTTP URL for PSP resources."
    )
    parser.add_argument(
        "--plot-header", default=DEFAULT_PLOT_HEADER, help="Path or name of the plot header CSV."
    )
    parser.add_argument(
        "--sample-byvisit",
        default=DEFAULT_SAMPLE_BYVISIT,
        help="Path or name of the sample-by-visit CSV.",
    )
    parser.add_argument(
        "--tree-detail", default=DEFAULT_TREE_DETAIL, help="Path or name of the tree detail CSV."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/examples/hps_baf12"),
        help="Directory for plot-level tallies.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Path to write the manifest CSV (defaults to <output>/manifest.csv).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/external/psp/raw"),
        help="Directory used for cached downloads.",
    )
    parser.add_argument(
        "--baf", type=float, default=12.0, help="Basal area factor assigned to the output tallies."
    )
    parser.add_argument("--bin-width", type=float, default=1.0, help="DBH bin width (cm).")
    parser.add_argument("--bin-origin", type=float, default=0.0, help="Origin for DBH bins (cm).")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Rows per chunk when streaming tree detail CSV.",
    )
    parser.add_argument(
        "--status",
        action="append",
        dest="status_codes",
        help="Tree status codes treated as live (default: L). Repeat for multiple codes.",
    )
    parser.add_argument(
        "--include-all-visits",
        action="store_true",
        help="Do not restrict to first-measurement visits.",
    )
    parser.add_argument(
        "--sample-type",
        action="append",
        dest="sample_types",
        help="Restrict plots to specific sample type codes (repeatable).",
    )
    parser.add_argument(
        "--max-plots",
        type=int,
        help="Limit the number of plots processed (useful for smoke tests).",
    )
    parser.add_argument(
        "--encoding", default="latin1", help="Encoding used when reading CSV files."
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Assume input paths are already local; never download.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download of inputs even if cached."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Do not write any files; report summary only."
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logs.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    plot_header_source = resolve_source(args.plot_header, args.base_url)
    sample_byvisit_source = resolve_source(args.sample_byvisit, args.base_url)
    tree_detail_source = resolve_source(args.tree_detail, args.base_url)

    cache_dir = args.cache_dir
    plot_header_path = ensure_local_path(
        plot_header_source,
        cache_dir=cache_dir,
        overwrite=args.force,
        quiet=args.quiet,
        skip_download=args.skip_download,
    )
    sample_byvisit_path = ensure_local_path(
        sample_byvisit_source,
        cache_dir=cache_dir,
        overwrite=args.force,
        quiet=args.quiet,
        skip_download=args.skip_download,
    )
    tree_detail_path = ensure_local_path(
        tree_detail_source,
        cache_dir=cache_dir,
        overwrite=args.force,
        quiet=args.quiet,
        skip_download=args.skip_download,
    )

    criteria = SelectionCriteria(
        first_visit_only=not args.include_all_visits,
        allowed_sample_types=tuple(args.sample_types) if args.sample_types else None,
        max_plots=args.max_plots,
    )
    selections = load_plot_selections(
        plot_header_path,
        sample_byvisit_path,
        baf=args.baf,
        criteria=criteria,
        encoding=args.encoding,
    )

    if not selections:
        print("No plots matched the selection criteria.", file=sys.stderr)
        return 1

    status_codes = args.status_codes or ["L"]

    tallies, metadata = aggregate_hps_tallies(
        tree_detail_path,
        selections,
        dbh_column="DBH",
        status_column="LV_D",
        live_status=status_codes,
        bin_width=args.bin_width,
        bin_origin=args.bin_origin,
        chunk_size=args.chunk_size,
        encoding=args.encoding,
    )

    if metadata.empty:
        print("Selected plots produced no tallies. Check filters or status codes.", file=sys.stderr)
        return 1

    if args.dry_run:
        print(
            f"[dry-run] {len(tallies)} plots would be written totaling "
            f"{metadata['trees'].sum()} trees.",
            file=sys.stderr,
        )
        return 0

    output_dir = args.output_dir
    manifest_path = args.manifest or (output_dir / "manifest.csv")
    write_outputs(
        tallies,
        metadata,
        output_dir=output_dir,
        manifest_path=manifest_path,
        quiet=args.quiet,
    )

    print(
        f"Prepared {len(tallies)} plots with {metadata['trees'].sum()} live trees → {output_dir}",
        file=sys.stderr if args.quiet else sys.stdout,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
