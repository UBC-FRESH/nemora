# Distribution Registry

`dbhdistfit` registers every probability density function (PDF) in the generalized beta family that
was validated in the legacy workflows, together with the canonical Weibull and Gamma forms. All
distributions are exposed through the registry functions in `dbhdistfit.distributions` and can be
used interchangeably by the fitting workflows and CLI.

## Parameter Conventions

- `s` denotes the scaling parameter introduced by the two-stage workflow. It is optional and only
  present when the underlying manuscript used a complete-form PDF with a free scale factor.
- `a`, `b`, `p`, `q` follow the notation from Ducey & Gove (2015) for the generalized beta families.
- `beta` represents the scale parameter of the generalized gamma specialisations.
- `mu` / `sigma2` correspond to the log-scale mean and variance for the lognormal construction.
- `u`, `v`, `d`, `df` retain their classical meanings for the `F`, `UG`, and half-`t` distributions.

All parameters are strictly positive unless otherwise noted; the registry populates conservative
starting values that can be overridden via `FitConfig`.

## Registered Distributions

```{list-table}
:header-rows: 1

* - Name
  - Parameters
  - Description
* - `weibull`
  - `a`, `beta`, `s`
  - Complete-form Weibull distribution.
* - `gamma`
  - `beta`, `p`, `s`
  - Gamma distribution with free scaling factor.
* - `gb1`
  - `a`, `b`, `p`, `q`, `s`
  - Generalized beta distribution of the first kind (GB1).
* - `gb2`
  - `a`, `b`, `p`, `q`, `s`
  - Generalized beta distribution of the second kind (GB2).
* - `gg`
  - `a`, `beta`, `p`, `s`
  - Generalized gamma parent distribution.
* - `ib1`
  - `b`, `p`, `q`, `s`
  - Inverted beta type I (GB1 with `a = -1`).
* - `ug`
  - `b`, `d`, `q`, `s`
  - Upper generalized distribution limit of GB1.
* - `b1`
  - `b`, `p`, `q`, `s`
  - Classical beta distribution on `(0, b)`.
* - `b2`
  - `b`, `p`, `q`, `s`
  - Beta distribution of the second kind.
* - `sm`
  - `a`, `b`, `q`, `s`
  - Singh–Maddala distribution.
* - `dagum`
  - `a`, `b`, `p`, `s`
  - Dagum (inverse Burr) distribution.
* - `pareto`
  - `b`, `p`, `s`
  - Pareto distribution.
* - `p`
  - `b`, `p`, `s`
  - Pearson type V distribution.
* - `ln`
  - `mu`, `sigma2`, `s`
  - Lognormal distribution derived from the generalized gamma limit.
* - `ga`
  - `beta`, `p`, `s`
  - Gamma distribution (alias to `gamma` without renaming parameters).
* - `w`
  - `a`, `beta`, `s`
  - Weibull distribution (alias to `weibull`).
* - `f`
  - `u`, `v`, `s`
  - Fisher–Snedecor `F` distribution.
* - `l`
  - `b`, `q`, `s`
  - Log-logistic (Type I) distribution.
* - `il`
  - `b`, `p`, `s`
  - Inverse log-logistic (Type II) distribution.
* - `fisk`
  - `a`, `b`, `s`
  - Fisk (log-logistic) distribution with explicit shape parameter.
* - `u`
  - `b`, `s`
  - Uniform distribution on `(0, b)`.
* - `halfn`
  - `sigma2`, `s`
  - Half-normal distribution.
* - `chisq`
  - `p`, `s`
  - Chi-square distribution.
* - `exp`
  - `beta`, `s`
  - Exponential distribution.
* - `r`
  - `beta`, `s`
  - Rayleigh distribution.
* - `halft`
  - `df`, `s`
  - Half-Student `t` distribution.
* - `ll`
  - `b`, `s`
  - Log-logistic distribution with equal shape parameters.
```
