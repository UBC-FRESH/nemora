"""Scratch pad for grouped Weibull likelihood experiments.

Generates synthetic grouped tallies from a three-parameter Weibull
(alpha = 0.5, beta = 12, shape = 2.6) and runs the grouped likelihood
refinement implemented in src/dbhdistfit/fitting/grouped.py. Intended
for development only – this file is not imported by the package or tests.
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng
from scipy.optimize import minimize
from scipy.stats import weibull_min

from dbhdistfit.fitting.grouped import _weibull_probabilities

rng = default_rng(0)
alpha_true = 0.5
scale_true = 12.0
shape_true = 2.6
n = 50_000

# Generate raw samples and bin to mimic grouped tallies
samples = alpha_true + weibull_min.ppf(rng.random(n), shape_true, scale=scale_true)
bins = np.arange(0, 40, 1.0)
counts, edges = np.histogram(samples, bins=bins)
counts = counts.astype(float)

# Shift bin edges so support starts at zero after subtracting alpha
edges_shift = edges - alpha_true
edges_shift[edges_shift < 0.0] = 0.0


def grouped_neg_loglik(theta: np.ndarray) -> float:
    shape = float(np.exp(theta[0]))
    scale = float(np.exp(theta[1]))
    cdf = weibull_min.cdf(edges_shift, shape, scale=scale)
    diffs = np.diff(cdf)
    diffs[0] += cdf[0]
    diffs[-1] += 1.0 - cdf[-1]
    diffs = np.clip(diffs, 1e-12, None)
    return float(-np.sum(counts * np.log(diffs)))


x0 = np.log([shape_true, scale_true])
result = minimize(grouped_neg_loglik, x0, method="L-BFGS-B")
shape_hat, scale_hat = np.exp(result.x)

print("Converged:", result.success, result.message)
print("shape_hat", shape_hat, "scale_hat", scale_hat)
print("delta shape", shape_hat - shape_true)
print("delta scale", scale_hat - scale_true)

probs = _weibull_probabilities(shape_hat, scale_hat, edges_shift)
print("prob_sum", probs.sum())
