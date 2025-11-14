# Sampling Inverse CDF Capability Matrix

Use this file to track which distributions expose analytic inverses.

| Distribution | Analytic Inverse? | Notes / Formula |
| --- | --- | --- |
| `exp` | ✅ | `icdf(u) = s - beta * ln(1-u)` |
| `pareto` | ✅ | `icdf(u) = s + beta * (1-u)^(-1/a) - beta` |
| `u` (uniform) | ✅ | `icdf(u) = low + (high-low) * u` |
| `weibull` | ✅ | `icdf(u) = s + beta * (-ln(1-u))**(1/a)` |
| `ln` (lognormal) | ✅ | `icdf(u) = exp(mu + sigma * Phi^{-1}(u))` |
| others (`b1`, `gamma`, `johnsonsb`, `birnbaum_saunders`, etc.) | ❌ | Numeric fallback; compare against `scipy.stats` `.ppf` for regression tests |

Action items:

1. Implement helper functions for the analytic cases above (guard against `u` in {0,1}).
2. For numeric-only distributions, add reference tests comparing our numeric inversion to SciPy quantiles.
3. Extend this table as additional distributions gain closed-form inverses or approximations.
