# Curve uncertainty

## Purpose

This guide quantifies the uncertainty of an **existing** curve family. It does
not fit a new growth model. The input is a paired validation table: observed
values and the curve's prediction at the same stand. The ratio
`observed / predicted` is fitted with a positive-support distribution from the
Nemora registry, and the fitted interval becomes a multiplicative band.

## Method

- Compute `ratio = observed / predicted` for each validation pair.
- Turn the ratios into grouped tallies with a histogram.
- Fit a distribution from the Nemora registry. The default is `weibull`.
- Use the fitted inverse CDF to get the lower and upper ratio quantiles.
- Apply the band as `predicted * lower_ratio` and `predicted * upper_ratio`.
- Check the coverage of the band on a holdout table.

The band is **empirical**. It inherits the sample's bias, species mix, age
range, and measurement error. It is not a parameter covariance from the original
curve fit.

## Usage

```bash
python scripts/curve_uncertainty.py \
  --validation tipsy_height_validation.tsa46.csv \
  --observed-column observed_height_m \
  --predicted-column predicted_height_m \
  --filter-column stratum_code \
  --filter-value CWH_HW \
  --holdout tipsy_height_validation_holdout.tsa46.csv \
  --holdout-filter-column stratum_code \
  --holdout-filter-value CWH_HW \
  --age-column observed_age \
  --out-json /tmp/curve_band.json \
  --out-csv /tmp/curve_band.csv
```

Python API:

```python
from nemora.uncertainty import (
    apply_ratio_band,
    band_coverage,
    fit_ratio_band,
)

band = fit_ratio_band(observed, predicted, distribution="weibull", level=0.90)
lower, upper = apply_ratio_band(predicted, band)
coverage = band_coverage(holdout_observed, holdout_predicted, band)
```

## Output

The JSON payload records:

- distribution and fitted parameters;
- nominal level;
- sample size and source;
- observed ratio minimum, maximum, and median;
- lower and upper ratio quantiles;
- goodness-of-fit metrics;
- optional holdout coverage.

The CSV has one row per validation observation:

| Column | Meaning |
| --- | --- |
| `age` | Age or x-axis value |
| `predicted` | Central prediction |
| `lower` | Lower band |
| `upper` | Upper band |

## Real TSA46 CWH_HW pilot

Training table: `tipsy_height_validation.tsa46.csv`, stratum `CWH_HW`, 840
pairs. Holdout: `tipsy_height_validation_holdout.tsa46.csv`, 306 pairs.

| Item | Value |
| --- | ---: |
| Distribution | Weibull |
| 90% ratio interval | 0.686 - 1.989 |
| Median observed/predicted ratio | 1.397 |
| Holdout coverage | 87.6% |

The interval is wide because the existing curve systematically under-predicts
these stands. The holdout coverage is close to, but below, the nominal 90%.
That is the honest result; the band is not narrowed to make the number look
better.

## Caveats

- The band is sample-specific. It is not transferable to other species, BEC
  zones, or age ranges without re-fitting.
- The input predictions are treated as the central curve. If the curve changes,
  the band must be re-fitted.
- A coverage below the nominal level is a finding, not a reason to widen the
  band until it passes.
- The band is not a substitute for a new model or a field-validation campaign.

## See also

- {doc}`hps_workflow`
- {doc}`sampling`
- {doc}`../api/uncertainty`
