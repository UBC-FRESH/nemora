# Candidate Features Inspired by ForestFit

This working note records ForestFit capabilities that look directly applicable to `dbhdistfit`.
For each item we outline what the feature does in ForestFit, how it could map into the current Python
architecture, and why importing (or re-implementing) the idea would strengthen `dbhdistfit`. The goal
is to credit the original work while planning concrete ports into our workflow-driven toolkit.

## Finite mixtures for grouped and ungrouped tallies
- **ForestFit reference:** `fitmixture()`, `fitmixturegrouped()`, and the corresponding density / CDF
  helpers (`dmixture`, `pmixture`, `rmixture`).
- **What it provides:** EM-based estimation for two or more component distributions (log-normal,
  Weibull, Birnbaum–Saunders, exponential families) with support for both raw DBH vectors and grouped
  frequency tables. Built-in GOF metrics (AIC, BIC, Anderson–Darling, Cramér–von Mises, KS) are
  computed per component configuration.
- **dbhdistfit port plan:**
  - Add a `dbhdistfit.fitting.mixture` module that wraps a generic EM routine. Components would reuse
    the existing distribution registry (so any registered PDF/CDF can participate).
  - Provide grouped-data adapters mirroring our HPS workflow (bin midpoints + tallies) so mixtures
    plug directly into `fit_hps_inventory` once an `--mixture` option is introduced.
  - Surface results through the CLI (`fit-hps`) and API (`FitResult`) with explicit mixture metadata
    (weights, component parameters, covariance).
- **Why it helps:** Mixtures are a recurring ask for uneven-aged or bimodal stands (see Marshall et
  al. 2020; Vafaei et al. 2014). Matching ForestFit’s capability keeps our users from leaving the
  Python workflow when a second component is required.

## Mixture distribution utilities (density / CDF / simulation)
- **ForestFit reference:** `dmixture`, `pmixture`, `rmixture` for evaluating and simulating mixture
  PDFs.
- **Port plan:** expose analogous helpers under `dbhdistfit.distributions.mixture` so that fitted
  mixtures can be interrogated outside the optimisation context (e.g., for Monte Carlo stand-table
  generation or GOF visualisations).
- **Integration points:** wrappers would build on the component registry and accept arbitrary
  component names + weights; results get threaded into diagnostics tables and plotting utilities.

## Grouped-sample estimators with EM and analytical moments
- **ForestFit reference:** `fitgrouped1()` / `fitgrouped2()` implement EM and ML estimators tailored
  to binned frequencies (including correction terms for lower/upper bounds, covariance estimation,
  and a battery of GOF statistics).
- **Port plan:** integrate the grouped-data EM implementations for Weibull, Generalised Exponential,
  Birnbaum–Saunders, etc., as optional back-ends inside `fit_inventory`. We can encapsulate the math
  in dedicated strategy objects so HPS tallies (which are naturally grouped) benefit from the
  improved estimators.
- **Rationale:** improves parameter stability when only stand tables—not raw tree lists—are
  available, aligning with common provincial data releases.

## Johnson SB (JSB) and Generalised Secant (GSM) families
- **ForestFit reference:** density/CDF generators `djsb`, `pjsb`, `djsbb`, `dgsm`, with dedicated
  fitters (`fitJSB`, `fitgsm`).
- **Port plan:** add JSB/GSM parameterisations to our distribution registry (leveraging SciPy where
  available, or porting the closed forms). Provide sensible initialisation routines and bounds so
  they cooperate with our existing optimisers.
- **Value add:** JSB/GSM offer flexible bounded support and proven performance on irregular stand
  structures in the forestry literature; including them broadens our default distribution library to
  match ForestFit’s coverage.
- **Status:** JSB plus GSMN (for any `N ≥ 2`) now live in the registry with grouped maximum-likelihood
  estimators; next step is to surface documentation/tutorial coverage and benchmark convergence for
  higher-component fits.

## Bayesian fitting via TMB analogues
- **ForestFit reference:** `fitbayesJSB()` and `fitbayesWeibull()` use Template Model Builder (TMB)
  for Bayesian posterior sampling and credible intervals.
- **Port plan:** prototype a lightweight Bayesian interface using PyMC or NumPyro wrappers around the
  same distributions, outputting posterior summaries alongside our deterministic fits. The hook would
  live in a new `dbhdistfit.fitting.bayes` module with optional dependencies.
- **Benefit:** Provides uncertainty quantification paths familiar to ForestFit users while keeping the
  workflow inside the Python stack (important for students who want probabilistic assessments).

## Height–diameter curve library
- **ForestFit reference:** `fitcurve()` supports a suite of classical H–D models (Weibull,
  Chapman–Richards, Logistic, Gompertz, etc.) with automatic plotting.
- **Port plan:** create a companion `dbhdistfit.curves` subpackage housing these formulations, using
  SciPy’s `curve_fit` or lmfit. We can expose them through notebooks and optionally integrate with
  HPS expansion factors for coupled inventory analysis.
- **Why:** Many practitioners need consistent H–D models alongside DBH distributions; folding them in
  reduces context switching and mirrors the ForestFit experience.

## GOF diagnostics (AD, CvM, KS + grouped residuals)
- **ForestFit reference:** mixture/grouped estimators return Anderson–Darling, Cramér–von Mises, and
  Kolmogorov–Smirnov statistics tailored for grouped data, plus residual plots.
- **Port plan:** expand `FitResult.gof` to include these statistics (either via SciPy or manual
  computation) and emit grouped residual tables suitable for plotting. Shared code can live in a
  `dbhdistfit.metrics` module.
- **Rationale:** matches the diagnostic suite practitioners expect from ForestFit, and complements our
  existing RSS/AICc reports.

## Example datasets and vignettes
- **ForestFit reference:** packaged datasets (`DBH`, `HW`, `SW`) used in vignettes.
- **Port plan:** mirror these as optional DataLad resources (with attribution) or link to them from
  the docs, providing cross-language examples. When licensing permits, convert a subset into the
  `examples/` directory for parity testing.
- **Rationale:** ensures users can replicate ForestFit tutorials inside dbhdistfit, lowering the
  barrier to migration or side-by-side comparisons.

---
Tracking these items publicly keeps upstream credit clear and helps prioritise Phase 2 tasks. As we
implement a feature, move the corresponding bullet to a “completed imports” section so downstream
users can see which ForestFit ideas have been carried across.
