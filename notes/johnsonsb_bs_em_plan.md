# Johnson SB & Birnbaum–Saunders Grouped EM Plan

Date: 2025-11-06

## Objectives

- Replace the current `_grouped_mle` L-BFGS pathway (with SciPy `.fit` fallback) for Johnson SB and Birnbaum–Saunders grouped tallies with a dedicated expectation–maximisation (EM) routine that mirrors ForestFit’s conditional maximum-likelihood approach.
- Provide numerically stable updates for each parameter family, surface convergence diagnostics, and retire the sample-expansion fallback.
- Retain the existing regression tolerances for PSP and spruce–fir fixtures while opening the door to richer goodness-of-fit reporting (covariance from observed information).

## Johnson SB (Beta-transformed) distribution

### Parameterisation

- Random variable: `X = loc + scale * Z`,  with `Z ∈ (0, 1)` and `Z ~ Beta(a, b)`.
- Parameters: `θ = (a, b, loc, scale)`, with `a, b, scale > 0`.
- Grouped log-likelihood:
  \[
  \ell(θ) = \sum_{j=1}^k n_j \log\Big(F(x_{u,j}; θ) - F(x_{l,j}; θ)\Big),
  \]
  where \(F\) is the Johnson SB CDF and \(x_{l,j}, x_{u,j}\) are bin bounds.

### EM structure

1. **E-step** (work in the latent Beta space).
   - Convert bin bounds to \(z\)-space: \(z_{l,j} = \mathrm{clip}((x_{l,j} - \mathrm{loc}) / \mathrm{scale}, 0, 1)\), similarly for \(z_{u,j}\).
   - Probability mass per bin:
     \[
     w_j = I_{z_{u,j}}(a, b) - I_{z_{l,j}}(a, b),
     \]
     where \(I_x(a, b)\) is the regularised incomplete beta function.
   - Sufficient statistics (conditioned on the interval):
     \[
     S_1 = \sum_j n_j \mathbb{E}[\log Z \mid z_{l,j} < Z < z_{u,j}], \quad
     S_2 = \sum_j n_j \mathbb{E}[\log (1-Z) \mid z_{l,j} < Z < z_{u,j}].
     \]
     Each conditional expectation is:
     \[
     \mathbb{E}[\log Z \mid z_{l,j} < Z < z_{u,j}] =
     \frac{1}{w_j B(a, b)} \int_{z_{l,j}}^{z_{u,j}} z^{a-1} (1 - z)^{b-1} \log z \, dz,
     \]
     and analogously for \(\log (1-Z)\). These integrals do not have closed forms, but can be computed robustly using `scipy.integrate.quad` (log-space evaluation to avoid underflow).

2. **M-step for shape parameters (`a`, `b`)**.
   - Solve the score equations derived from the complete-data Beta log-likelihood:
     \[
     \psi(a) - \psi(a + b) = \frac{S_1}{N}, \quad
     \psi(b) - \psi(a + b) = \frac{S_2}{N},
     \]
     where `ψ` is the digamma function and \(N = \sum_j n_j\).
   - Use Newton iterations on the coupled system, leveraging the trigamma function `polygamma(1, ·)` for the Jacobian.

3. **M-step for location/scale**.
   - With updated `a`, `b`, maximise the grouped likelihood over `(loc, scale)` while holding the shape parameters fixed.
   - Gradients can be derived from the CDF differences:
     \[
     \frac{\partial \ell}{\partial \mathrm{loc}} = \sum_j \frac{n_j}{w_j}
       \left.\frac{\partial F(x; θ)}{\partial \mathrm{loc}}\right|_{x_{u,j}}^{x_{l,j}},
     \quad
     \frac{\partial \ell}{\partial \mathrm{scale}} = \sum_j \frac{n_j}{w_j}
       \left.\frac{\partial F(x; θ)}{\partial \mathrm{scale}}\right|_{x_{u,j}}^{x_{l,j}},
     \]
     where derivatives follow from the Beta PDF evaluated at the transformed edges.
   - Practically, employ a safeguarded Newton (line search) or L-BFGS in the two-parameter subspace, enforcing positivity on `scale` via a log transform.

4. **Convergence & diagnostics**.
   - Iterate until parameter updates fall below tolerance (`1e-6`) or likelihood improvement drops under threshold.
   - Surface iteration counts, delta metrics, Hessian-derived covariance (numerical) for reporting.
   - Guard against bins collapsing in z-space (probability ≈ 0); apply epsilon clipping (`1e-12`) and report warnings.

## Birnbaum–Saunders (Fatigue life) distribution

### Parameterisation

- PDF: \( f(x; α, β) = \frac{1}{2αβ\sqrt{2\pi}} \left(\sqrt{\frac{β}{x}} + \sqrt{\frac{x}{β}}\right)
  \exp\left(-\frac{1}{2α^2}\left(\sqrt{\frac{x}{β}} - \sqrt{\frac{β}{x}}\right)^2\right) \),
  for \( x > 0 \).
- Transformation to Normal: If \(Y \sim N(0, 1)\) then \(X = β \left(\frac{αY}{2} + \sqrt{\left(\frac{αY}{2}\right)^2 + 1}\right)^2\).
- Parameters: `α > 0` (shape), `β > 0` (scale).

### EM outline

1. **Latent variable**: work with `Y` (standard normal) for grouped bins:
   - For each bin `[l_j, u_j]`, convert to `y`-bounds via the inverse transformation \(Y = \frac{1}{α}\left(\sqrt{\frac{x}{β}} - \sqrt{\frac{β}{x}}\right)\).
   - Compute bin probability mass using the normal CDF.

2. **E-step statistics**.
   - Need \(\mathbb{E}[Y \mid Y \in (y_{l,j}, y_{u,j})]\) and \(\mathbb{E}[Y^2 \mid ...]\).
   - These conditional expectations have closed forms using the normal PDF/CDF:
     \[
     \mathbb{E}[Y \mid y_l < Y < y_u] = \frac{\phi(y_l) - \phi(y_u)}{\Phi(y_u) - \Phi(y_l)},
     \]
     \[
     \mathbb{E}[Y^2 \mid y_l < Y < y_u] = 1 + \frac{y_l \phi(y_l) - y_u \phi(y_u)}{\Phi(y_u) - \Phi(y_l)},
     \]
     where `φ`, `Φ` are the standard normal PDF/CDF.
   - Accumulate:
     \[
     T_1 = \sum_j n_j \mathbb{E}[Y \mid ...], \quad
     T_2 = \sum_j n_j \mathbb{E}[Y^2 \mid ...].
     \]

3. **M-step**.
   - Use the moment equations derived from the fatigue-life log-likelihood (following ForestFit documentation):
     \[
     α_{\text{new}} = \sqrt{\frac{T_2}{N} - 1 - \left(\frac{T_1}{N}\right)^2},
     \quad
     β_{\text{new}} = \frac{\sum_j n_j \sqrt{l_j u_j}}{\sum_j n_j},
     \]
     refined with Newton updates if necessary (ForestFit applies a single Newton correction after the moment estimate).
   - Alternatively, maximise the grouped log-likelihood over `(α, β)` using Newton with the conditional expectations pre-computed; the EM route will leverage the latent normal statistics.

4. **Convergence & guards**.
   - Ensure the argument under the square root for `α_new` stays positive; if not, shrink using damping.
   - Clip bin probability masses away from zero; if a bin collapses, merge with neighbours or inject a small epsilon count (report via diagnostics).

## Implementation roadmap

1. **Helper utilities** (module-local).
   - `_beta_interval_log_moments(a, b, lower, upper)` → returns conditional expectations for `log Z`, `log(1-Z)`, and optionally `Z`.
   - `_normal_interval_moments(lower, upper)` → closed-form conditional `E[Y]`, `E[Y^2]`.
   - `_solve_beta_mstep(S1, S2, N, initial_guess)` → Newton solver for `a`, `b`.
   - `_maximize_loc_scale(...)` → 2d Newton/L-BFGS wrapper using gradients from CDF differences. *(2025-11-06 update: Johnson SB EM now fixes `loc/scale` to the dataset support to keep the solver stable; revisit dedicated optimisation once shape updates are finalised.)*

2. **Johnson SB EM driver**.
   - Iterate: compute bin masses/conditional stats, update `(a, b)`, then `loc/scale`, until convergence.
   - Surface diagnostics (`iterations`, per-parameter deltas, fallback flags). Provide option to revert to previous iterate if likelihood decreases.

3. **Birnbaum–Saunders EM driver**.
   - Apply latent-normal moment updates, optionally refine via Newton. *(2025-11-06 — implemented an EMT-style loop using truncated normal moments with a bounded scalar search for `β`; when the variance term is non-positive the solver falls back to the legacy grouped MLE and records the chosen path in diagnostics.)*
   - Attach diagnostics and covariance (via numerical Hessian on `(α, β)`).

4. **Testing**.
   - Extend `tests/test_grouped.py` with synthetic Beta/Birnbaum grouped datasets where the true parameters are known. *(Johnson SB test now checks for the `grouped-em` diagnostic.)*
   - Update `tests/test_grouped_fixtures.py` to assert Johnson SB/Birnbaum estimators converge on PSP-derived grouped data (once fixtures are available or synthesised).

5. **Docs & change log**.
   - Document the new EM routines in `docs/howto/custom_distributions.md`, `docs/howto/hps_workflow.md`, and the changelog.
   - Note any residual caveats (e.g., need for bin-width metadata, damping heuristics).

This plan captures the implementation approach and references required to proceed. The next execution step is to implement the helper utilities and wire the Johnson SB EM driver.
