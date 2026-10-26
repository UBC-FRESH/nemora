"""Fit a multiplicative uncertainty band to an existing curve family.

Example
-------
    python scripts/curve_uncertainty.py \
      --validation tipsy_height_validation.tsa46.csv \
      --observed-column observed_height_m \
      --predicted-column predicted_height_m \
      --filter-column stratum_code \
      --filter-value CWH_HW \
      --holdout tipsy_height_validation_holdout.tsa46.csv \
      --age-column observed_age \
      --out-json /tmp/curve_band.json \
      --out-csv /tmp/curve_band.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from nemora.uncertainty import (
    band_coverage,
    fit_ratio_band,
    read_validation_csv,
    write_band_csv,
)


def main(argv: list[str] | None = None) -> int:
    """Run the curve-uncertainty workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--observed-column", required=True)
    parser.add_argument("--predicted-column", required=True)
    parser.add_argument("--filter-column")
    parser.add_argument("--filter-value")
    parser.add_argument("--holdout", type=Path)
    parser.add_argument("--holdout-filter-column")
    parser.add_argument("--holdout-filter-value")
    parser.add_argument("--age-column")
    parser.add_argument("--distribution", default="weibull")
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--level", type=float, default=0.90)
    parser.add_argument("--source")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args(argv)

    observed, predicted = read_validation_csv(
        args.validation,
        observed_column=args.observed_column,
        predicted_column=args.predicted_column,
        filter_column=args.filter_column,
        filter_value=args.filter_value,
    )
    band = fit_ratio_band(
        observed,
        predicted,
        distribution=args.distribution,
        bins=args.bins,
        level=args.level,
        source=args.source,
    )

    coverage = None
    if args.holdout is not None:
        holdout_observed, holdout_predicted = read_validation_csv(
            args.holdout,
            observed_column=args.observed_column,
            predicted_column=args.predicted_column,
            filter_column=args.holdout_filter_column,
            filter_value=args.holdout_filter_value,
        )
        coverage = band_coverage(holdout_observed, holdout_predicted, band)

    payload = band.to_dict()
    payload["coverage"] = coverage
    args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    frame = pd.read_csv(args.validation)
    if args.filter_column is not None:
        frame = frame[frame[args.filter_column].astype(str) == str(args.filter_value)]
    if frame.empty:
        raise ValueError("training table is empty after filtering")
    age = (
        frame[args.age_column].to_numpy(dtype=float)
        if args.age_column is not None
        else range(len(frame))
    )
    write_band_csv(
        args.out_csv,
        age=age,
        predicted=frame[args.predicted_column].to_numpy(dtype=float),
        band=band,
    )

    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"band JSON: {args.out_json}")
    print(f"band CSV:  {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
