#!/usr/bin/env python3
"""Regenerate the deterministic synthetic landscape used by the WS4 smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

from nemora.synthesis.ws4_fixture import (
    DEFAULT_AREA_HA,
    DEFAULT_ASPECT_RATIO,
    DEFAULT_COUNT,
    DEFAULT_TREES_PER_HA,
    generate_ws4_fixture,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/ws4_synthetic"),
        help="Fixture directory (default: examples/ws4_synthetic).",
    )
    parser.add_argument("--seed", type=int, required=True, help="Deterministic RNG seed.")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Voronoi fragment count.")
    parser.add_argument(
        "--aspect-ratio",
        type=float,
        default=DEFAULT_ASPECT_RATIO,
        help="Landscape width / height.",
    )
    parser.add_argument(
        "--area-ha",
        type=float,
        default=DEFAULT_AREA_HA,
        help="Total landscape area in hectares.",
    )
    parser.add_argument(
        "--trees-per-ha",
        type=float,
        default=DEFAULT_TREES_PER_HA,
        help="Placeholder stem density.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing fixture's generated files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        if not args.force:
            raise SystemExit(
                f"{args.output} is not empty; pass --force to overwrite its generated files."
            )
    fixture = generate_ws4_fixture(
        args.output,
        seed=args.seed,
        count=args.count,
        aspect_ratio=args.aspect_ratio,
        area_ha=args.area_ha,
        trees_per_ha=args.trees_per_ha,
    )
    totals = fixture.metadata["totals"]
    print(
        f"Wrote {fixture.directory} "
        f"(seed={fixture.seed}, area_ha={totals['area_ha']}, "
        f"expected_volume_m3={totals['expected_volume_m3']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
