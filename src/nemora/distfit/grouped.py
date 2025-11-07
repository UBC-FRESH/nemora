"""Grouped-data estimators for selected distributions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.integrate import quad
from scipy.optimize import approx_fprime, brentq, curve_fit, minimize, minimize_scalar
from scipy.special import betainc, betaln, digamma, polygamma
from scipy.stats import fatiguelife, johnsonsb, norm, weibull_min
from scipy.stats import gamma as gamma_dist

from ..core import FitResult, InventorySpec
from ..distributions import weibull_pdf

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from . import FitConfig

GroupedEstimator = Callable[[InventorySpec, "FitConfig | None"], FitResult]

WEIBULL_CML_OFFSET = 0.5


def _numerical_hessian(
    func: Callable[[np.ndarray], float],
    theta: np.ndarray,
    *,
    step: float = 1e-5,
) -> np.ndarray | None:
    """Approximate the Hessian using central finite differences."""
    n_params = theta.size
    hessian = np.zeros((n_params, n_params), dtype=float)
    try:
        f0 = func(theta)
    except Exception:  # pragma: no cover - numerical failure
        return None

    for i in range(n_params):
        ei = np.zeros(n_params, dtype=float)
        ei[i] = step
        try:
            f_plus = func(theta + ei)
            f_minus = func(theta - ei)
        except Exception:
            return None
        hessian[i, i] = (f_plus - 2.0 * f0 + f_minus) / (step**2)
        for j in range(i + 1, n_params):
            ej = np.zeros(n_params, dtype=float)
            ej[j] = step
            try:
                f_pp = func(theta + ei + ej)
                f_pm = func(theta + ei - ej)
                f_mp = func(theta - ei + ej)
                f_mm = func(theta - ei - ej)
            except Exception:
                return None
            value = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step**2)
            hessian[i, j] = value
            hessian[j, i] = value
    return hessian


def _weibull_probabilities(
    a: float,
    beta: float,
    edges: np.ndarray,
    *,
    location_shift: float = 0.0,
) -> np.ndarray:
    """Return bin probabilities for a Weibull distribution across provided edges."""
    shifted = np.asarray(edges, dtype=float) - location_shift
    shifted = np.clip(shifted, 0.0, None)
    cdf_vals = weibull_min.cdf(shifted, a, scale=beta)
    probabilities = np.diff(cdf_vals)
    if probabilities.size == 0:
        return probabilities
    probabilities[0] += cdf_vals[0]
    probabilities[-1] += 1.0 - cdf_vals[-1]
    probabilities = np.clip(probabilities, 1e-12, None)
    total = float(np.sum(probabilities))
    if total <= 0:
        return np.full_like(probabilities, 1.0 / probabilities.size)
    return probabilities / total


def _prepare_grouped_data(
    inventory: InventorySpec,
    *,
    min_bins: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin centroids, counts) after removing empty bins."""
    x = np.asarray(inventory.bins, dtype=float)
    y = np.asarray(inventory.tallies, dtype=float)
    counts = np.clip(np.round(y).astype(int), 0, None)
    mask = counts > 0
    x = x[mask]
    counts = counts[mask]
    if x.size < min_bins or counts.sum() == 0:
        raise ValueError("Grouped estimator requires at least three populated bins.")
    return x, counts


def _bin_edges_from_centroids(x: np.ndarray) -> np.ndarray:
    """Approximate bin edges from centroids for grouped tallies."""
    edges = np.zeros(x.size + 1, dtype=float)
    if x.size == 1:
        width = max(x[0] * 0.1, 1.0)
        edges[0] = max(x[0] - width, 0.0)
        edges[1] = x[0] + width
        return edges
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    first_width = max(x[1] - x[0], 1.0)
    edges[0] = max(x[0] - first_width / 2.0, 0.0)
    last_width = max(x[-1] - x[-2], 1.0)
    edges[-1] = x[-1] + last_width / 2.0
    return edges


def _observed_expected_cdf(
    counts: np.ndarray,
    expected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    total = float(np.sum(counts))
    obs_cdf = np.cumsum(counts) / total
    exp_cdf = np.cumsum(expected) / total
    return obs_cdf, exp_cdf


def _build_initial_vector(
    param_names: tuple[str, ...],
    initial_map: dict[str, float],
    transform: dict[str, str],
) -> np.ndarray:
    theta0 = np.zeros(len(param_names), dtype=float)
    for idx, name in enumerate(param_names):
        value = float(initial_map.get(name, 1.0))
        if transform.get(name) == "log":
            theta0[idx] = np.log(max(value, 1e-6))
        else:
            theta0[idx] = value
    return theta0


def _convert_params(
    theta: np.ndarray,
    param_names: tuple[str, ...],
    transform: dict[str, str],
) -> dict[str, float]:
    params: dict[str, float] = {}
    for idx, name in enumerate(param_names):
        if transform.get(name) == "log":
            params[name] = float(np.exp(theta[idx]))
        else:
            params[name] = float(theta[idx])
    return params


def _approximate_covariance(
    result,
    params: dict[str, float],
    transform: dict[str, str],
    param_names: tuple[str, ...],
) -> np.ndarray | None:
    hess_inv = getattr(result, "hess_inv", None)
    if hess_inv is None:
        return None
    try:
        matrix = np.asarray(hess_inv.todense(), dtype=float)
    except AttributeError:
        matrix = np.asarray(hess_inv, dtype=float)
    if matrix.size != len(param_names) ** 2:
        return None
    jacobian = np.diag(
        [params[name] if transform.get(name) == "log" else 1.0 for name in param_names]
    )
    try:
        return jacobian @ matrix @ jacobian
    except Exception:  # pragma: no cover - numerical failure fallback
        return None


def _assemble_grouped_result(
    distribution: str,
    params: dict[str, float],
    counts: np.ndarray,
    edges: np.ndarray,
    probabilities: np.ndarray,
    *,
    covariance: np.ndarray | None,
    method: str,
    amplitude: float,
    extras: dict[str, float | int | bool | str | np.ndarray | None] | None = None,
) -> FitResult:
    expected = np.clip(amplitude * probabilities, 1e-12, None)
    residuals = counts - expected
    log_likelihood = float(np.sum(counts * np.log(expected) - expected))
    rss = float(np.sum(np.square(residuals)))
    chisq = float(np.sum(np.square(residuals) / np.clip(expected, 1e-12, None)))
    obs_cdf, exp_cdf = _observed_expected_cdf(counts, expected)
    ks = float(np.max(np.abs(obs_cdf - exp_cdf)))
    total = float(np.sum(counts))
    cvm = float(np.sum((obs_cdf - exp_cdf) ** 2 * counts / total)) if total > 0 else 0.0
    denom = np.clip(exp_cdf * (1.0 - exp_cdf), 1e-12, None)
    ad = float(np.sum(counts * np.square(obs_cdf - exp_cdf) / denom))

    k_params = len(params)
    aic = 2.0 * k_params - 2.0 * log_likelihood
    aicc = np.nan
    if total - k_params - 1.0 > 0.0:
        aicc = aic + (2.0 * k_params * (k_params + 1.0)) / (total - k_params - 1.0)
    bic = k_params * np.log(total) - 2.0 * log_likelihood

    gof = {
        "rss": rss,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "aicc": float(aicc) if not np.isnan(aicc) else np.nan,
        "bic": bic,
        "chisq": chisq,
        "ks": ks,
        "cvm": cvm,
        "ad": ad,
    }

    diagnostics: dict[str, float | int | bool | str | np.ndarray | None] = {
        "method": method,
        "distribution": distribution,
        "sample_size": int(total),
        "bin_edges": edges,
        "probabilities": probabilities,
        "amplitude": amplitude,
        "observed": counts,
        "expected": expected,
        "residuals": residuals,
        "observed_cdf": obs_cdf,
        "expected_cdf": exp_cdf,
    }
    if extras:
        diagnostics.update(extras)

    return FitResult(
        distribution=distribution,
        parameters=params,
        covariance=covariance,
        gof=gof,
        diagnostics=diagnostics,
    )


def _beta_interval_moments(
    a: float,
    b: float,
    lower: float,
    upper: float,
) -> tuple[float, float, float]:
    """Return (mass, E[log Z], E[log(1-Z)]) for Z ~ Beta(a, b) truncated to (lower, upper)."""
    epsilon = 1e-12
    lower_clipped = float(np.clip(lower, epsilon, 1.0 - epsilon))
    upper_clipped = float(np.clip(upper, epsilon, 1.0 - epsilon))
    if upper_clipped <= lower_clipped + 1e-12:
        mid = max(min(0.5 * (lower_clipped + upper_clipped), 1.0 - epsilon), epsilon)
        log_mid = float(np.log(mid))
        log_one_minus_mid = float(np.log1p(-mid))
        return epsilon, log_mid, log_one_minus_mid
    mass = float(
        np.clip(betainc(a, b, upper_clipped) - betainc(a, b, lower_clipped), epsilon, None)
    )
    log_norm = float(-betaln(a, b))

    def _integrand_log_z(z: float) -> float:
        return np.exp((a - 1.0) * np.log(z) + (b - 1.0) * np.log1p(-z) + log_norm) * np.log(z)

    def _integrand_log_one_minus(z: float) -> float:
        return np.exp((a - 1.0) * np.log(z) + (b - 1.0) * np.log1p(-z) + log_norm) * np.log1p(-z)

    log_z_int, _ = quad(
        _integrand_log_z,
        lower_clipped,
        upper_clipped,
        limit=100,
        epsabs=1e-10,
        epsrel=1e-8,
    )
    log_one_minus_int, _ = quad(
        _integrand_log_one_minus,
        lower_clipped,
        upper_clipped,
        limit=100,
        epsabs=1e-10,
        epsrel=1e-8,
    )
    return mass, float(log_z_int / mass), float(log_one_minus_int / mass)


def _normal_interval_moments(
    lower: float,
    upper: float,
) -> tuple[float, float, float]:
    """Return (mass, E[Y], E[Y^2]) for truncated standard normal Y in (lower, upper)."""
    epsilon = 1e-12

    if np.isneginf(lower):
        phi_lower = 0.0
        cdf_lower = 0.0
    else:
        phi_lower = float(norm.pdf(lower))
        cdf_lower = float(norm.cdf(lower))

    if np.isposinf(upper):
        phi_upper = 0.0
        cdf_upper = 1.0
    else:
        phi_upper = float(norm.pdf(upper))
        cdf_upper = float(norm.cdf(upper))

    mass = float(np.clip(cdf_upper - cdf_lower, epsilon, None))
    mean = (phi_lower - phi_upper) / mass
    mean_sq = 1.0 + (lower * phi_lower - upper * phi_upper) / mass
    return mass, mean, mean_sq


def _solve_beta_mstep(
    a: float,
    b: float,
    target_log_z: float,
    target_log_one_minus: float,
    *,
    max_iter: int = 15,
    tol: float = 1e-6,
) -> tuple[float, float, bool]:
    """Solve the Beta M-step equations via damped Newton iterations."""
    a_curr = max(a, 1e-6)
    b_curr = max(b, 1e-6)
    target1 = float(target_log_z)
    target2 = float(target_log_one_minus)
    for _ in range(max_iter):
        psi_sum = digamma(a_curr + b_curr)
        g1 = digamma(a_curr) - psi_sum - target1
        g2 = digamma(b_curr) - psi_sum - target2
        if max(abs(g1), abs(g2)) < tol:
            return a_curr, b_curr, True
        h11 = polygamma(1, a_curr) - polygamma(1, a_curr + b_curr)
        h22 = polygamma(1, b_curr) - polygamma(1, a_curr + b_curr)
        cross = -polygamma(1, a_curr + b_curr)
        jacobian = np.array([[h11, cross], [cross, h22]], dtype=float)
        grad = np.array([g1, g2], dtype=float)
        try:
            step = np.linalg.solve(jacobian, -grad)
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(jacobian, grad, rcond=None)[0]
        damping = 1.0
        for _ in range(6):
            a_candidate = a_curr + damping * step[0]
            b_candidate = b_curr + damping * step[1]
            if a_candidate > 1e-6 and b_candidate > 1e-6:
                a_curr = a_candidate
                b_curr = b_candidate
                break
            damping *= 0.5
        else:
            return a_curr, b_curr, False
    return a_curr, b_curr, False


def _johnsonsb_bin_probabilities(
    a: float,
    b: float,
    loc: float,
    scale: float,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    if scale <= 1e-12:
        return None, None, None
    lower = (edges[:-1] - loc) / scale
    upper = (edges[1:] - loc) / scale
    epsilon = 1e-12
    lower = np.clip(lower, epsilon, 1.0 - epsilon)
    upper = np.clip(upper, epsilon, 1.0 - epsilon)
    if np.any(upper <= lower):
        return None, None, None
    probabilities = betainc(a, b, upper) - betainc(a, b, lower)
    probabilities = np.clip(probabilities, epsilon, None)
    probabilities = probabilities / float(np.sum(probabilities))
    return probabilities, lower, upper


def _fit_johnsonsb_em(
    inventory: InventorySpec,
    config: FitConfig,
    *,
    max_iter: int = 25,
    tol: float = 1e-5,
) -> FitResult | None:
    x, counts = _prepare_grouped_data(inventory)
    edges = _bin_edges_from_centroids(x)
    total = float(np.sum(counts))
    initial_map = dict(config.initial)
    min_edge = float(np.min(edges[:-1]))
    max_edge = float(np.max(edges[1:]))
    a_guess = float(initial_map.get("a", 1.5))
    b_guess = float(initial_map.get("b", 2.5))
    a = float(np.clip(a_guess, 0.5, 8.0))
    b = float(np.clip(b_guess, 0.5, 8.0))
    loc = float(initial_map.get("loc", min_edge)) - 1e-6
    scale = float(initial_map.get("scale", max_edge - loc))
    if scale <= 0 or loc + scale <= max_edge:
        scale = max_edge - loc + 1e-3
    best_loglik = -np.inf
    converged = False
    last_delta = float("inf")

    last_iteration = 0
    for iteration in range(1, max_iter + 1):
        last_iteration = iteration
        probabilities, lower, upper = _johnsonsb_bin_probabilities(a, b, loc, scale, edges)
        if probabilities is None or lower is None or upper is None:
            break
        loglik = float(np.sum(counts * np.log(probabilities)))
        if loglik < best_loglik - 1e-6:
            break
        best_loglik = loglik

        mass_list: list[float] = []
        logz_list: list[float] = []
        log1mz_list: list[float] = []
        for lower_bound, upper_bound in zip(lower, upper, strict=False):
            mass, elogz, elog1mz = _beta_interval_moments(a, b, lower_bound, upper_bound)
            mass_list.append(mass)
            logz_list.append(elogz)
            log1mz_list.append(elog1mz)
        mass_arr = np.asarray(mass_list, dtype=float)
        if np.any(mass_arr <= 0):
            break
        logz_arr = np.asarray(logz_list, dtype=float)
        log1mz_arr = np.asarray(log1mz_list, dtype=float)
        target_log_z = float(np.sum(counts * logz_arr) / total)
        target_log_one_minus = float(np.sum(counts * log1mz_arr) / total)

        prev_a, prev_b = a, b
        a_new, b_new, ok = _solve_beta_mstep(a, b, target_log_z, target_log_one_minus)
        if not ok:
            break
        a, b = a_new, b_new

        probabilities, _, _ = _johnsonsb_bin_probabilities(a, b, loc, scale, edges)
        if probabilities is None:
            break
        new_loglik = float(np.sum(counts * np.log(probabilities)))
        delta = abs(new_loglik - best_loglik)
        best_loglik = new_loglik
        last_delta = delta
        param_delta = max(abs(a - prev_a), abs(b - prev_b))
        if delta < tol * (1.0 + abs(best_loglik)) or param_delta < 1e-5:
            converged = True
            break

    if not converged:
        return None

    probabilities, _, _ = _johnsonsb_bin_probabilities(a, b, loc, scale, edges)
    if probabilities is None:
        return None
    params = {"a": a, "b": b, "loc": loc, "scale": scale}
    extras: dict[str, float | int | bool | str | np.ndarray | None] = {
        "iterations": last_iteration,
        "converged": converged,
        "delta_loglik": last_delta,
        "method_detail": "em-shape-newton",
    }
    return _assemble_grouped_result(
        "johnsonsb",
        params,
        counts,
        edges,
        probabilities,
        covariance=None,
        method="grouped-em",
        amplitude=total,
        extras=extras,
    )


def _grouped_mle(
    inventory: InventorySpec,
    config: FitConfig | None,
    *,
    distribution: str,
    param_names: tuple[str, ...],
    cdf_callable: Callable[[np.ndarray, dict[str, float]], np.ndarray],
    defaults: dict[str, float],
    positive_parameters: tuple[str, ...],
    scale_parameter: str | None = None,
) -> FitResult:
    if config is None:
        raise ValueError("Grouped estimator requires a FitConfig instance.")

    x, counts = _prepare_grouped_data(inventory)
    edges = _bin_edges_from_centroids(x)
    total = float(np.sum(counts))

    initial_map = dict(config.initial)
    for key, value in defaults.items():
        if key in param_names:
            initial_map.setdefault(key, value)
    transform = {
        name: ("log" if name in positive_parameters else "identity") for name in param_names
    }
    theta0 = _build_initial_vector(param_names, initial_map, transform)

    def objective(theta: np.ndarray) -> float:
        params = _convert_params(theta, param_names, transform)
        cdf_vals = cdf_callable(edges, params)
        if np.any(np.isnan(cdf_vals)):
            return np.inf
        probabilities = np.diff(cdf_vals)
        if cdf_vals[0] > 0:
            probabilities[0] += cdf_vals[0]
        tail_mass = 1.0 - cdf_vals[-1]
        if tail_mass > 0:
            probabilities[-1] += tail_mass
        if np.any(probabilities <= 0):
            return np.inf
        prob_sum = float(np.sum(probabilities))
        if prob_sum <= 0.0:
            return np.inf
        amplitude = total if scale_parameter is None else total / prob_sum
        expected = np.clip(amplitude * probabilities, 1e-12, None)
        return float(np.sum(expected - counts * np.log(expected)))

    result = minimize(objective, theta0, method="L-BFGS-B")
    if not result.success:
        raise ValueError(f"Grouped optimisation failed for {distribution}: {result.message}")

    params = _convert_params(result.x, param_names, transform)
    cdf_vals = cdf_callable(edges, params)
    probabilities = np.diff(cdf_vals)
    if cdf_vals[0] > 0:
        probabilities[0] += cdf_vals[0]
    tail_mass = 1.0 - cdf_vals[-1]
    if tail_mass > 0:
        probabilities[-1] += tail_mass
    probabilities = np.clip(probabilities, 1e-12, None)
    prob_sum = float(np.sum(probabilities))
    amplitude = total if scale_parameter is None else total / max(prob_sum, 1e-12)

    hessian = _numerical_hessian(objective, result.x)
    covariance = None
    if hessian is not None:
        try:
            hessian_inv = np.linalg.inv(hessian)
            pseudo_result = type("OptResult", (), {"hess_inv": hessian_inv})
            covariance = _approximate_covariance(pseudo_result, params, transform, param_names)
        except np.linalg.LinAlgError:  # pragma: no cover - singular Hessian
            covariance = None
    if scale_parameter:
        params[scale_parameter] = amplitude
        if covariance is not None:
            size = covariance.shape[0]
            expanded = np.zeros((size + 1, size + 1), dtype=float)
            expanded[:size, :size] = covariance
            covariance = expanded
    extras: dict[str, float | int | bool | str | np.ndarray | None] = {
        "iterations": int(getattr(result, "nit", 0)),
        "converged": bool(result.success),
        "status": getattr(result, "status", None),
        "message": result.message,
        "bins": x,
    }
    return _assemble_grouped_result(
        distribution,
        params,
        counts,
        edges,
        probabilities,
        covariance=covariance,
        method="grouped-mle",
        amplitude=amplitude,
        extras=extras,
    )


def _fit_weibull_grouped(
    inventory: InventorySpec,
    config: FitConfig | None,
) -> FitResult:
    if config is None:
        raise ValueError("Grouped estimator requires a FitConfig instance.")

    x = np.asarray(inventory.bins, dtype=float)
    y = np.asarray(inventory.tallies, dtype=float)
    weights = config.weights
    edges = _bin_edges_from_centroids(x)
    min_dbh = float(np.min(x))
    location_shift = max(0.0, min_dbh - WEIBULL_CML_OFFSET)
    mode = str(inventory.metadata.get("grouped_weibull_mode", "auto")).lower()
    initial_map = dict(config.initial)
    a0 = float(initial_map.get("a", 2.0))
    beta0 = float(initial_map.get("beta", max(float(np.mean(x)) if x.size else 10.0, 1.0)))
    s0 = float(initial_map.get("s", np.max(y) if y.size else 1.0))

    def model(x_vals: np.ndarray, a: float, beta: float, s: float) -> np.ndarray:
        return weibull_pdf(x_vals, {"a": a, "beta": beta, "s": s})

    params, cov = curve_fit(
        model,
        x,
        y,
        p0=[a0, beta0, s0],
        sigma=weights,
        maxfev=int(2e5),
    )

    fitted = model(x, *params)
    residuals = y - fitted
    rss = float(np.sum(np.square(residuals)))
    n = y.size
    rss_safe = max(rss, 1e-12)
    k_params = 3
    aic = float(n * np.log(rss_safe / max(n, 1)) + 2.0 * k_params) if n else float("nan")
    aicc = float("nan")
    if n - k_params - 1 > 0:
        aicc = aic + (2.0 * k_params * (k_params + 1.0)) / (n - k_params - 1.0)
    bic = float(n * np.log(rss_safe / max(n, 1)) + k_params * np.log(n)) if n else float("nan")
    chisq = float(np.sum(np.square(residuals) / np.clip(fitted, 1e-12, None)))

    fitted_safe = np.clip(fitted, 1e-12, None)
    log_likelihood = float(np.sum(y * np.log(fitted_safe) - fitted_safe))

    total_obs = float(np.sum(y))
    obs_cdf = np.cumsum(y) / total_obs if total_obs > 0 else np.zeros_like(y)
    fitted_sum = float(np.sum(fitted_safe))
    exp_cdf = np.cumsum(fitted_safe) / fitted_sum if fitted_sum > 0 else np.zeros_like(fitted_safe)
    ks = float(np.max(np.abs(obs_cdf - exp_cdf)))
    cvm = float(np.sum((obs_cdf - exp_cdf) ** 2 * y / total_obs)) if total_obs > 0 else 0.0
    denom = np.clip(exp_cdf * (1.0 - exp_cdf), 1e-12, None)
    ad = float(np.sum(y * np.square(obs_cdf - exp_cdf) / denom)) if total_obs > 0 else 0.0

    gof = {
        "rss": rss,
        "log_likelihood": log_likelihood,
        "aic": aic,
        "aicc": float(aicc) if not np.isnan(aicc) else np.nan,
        "bic": bic,
        "chisq": chisq,
        "ks": ks,
        "cvm": cvm,
        "ad": ad,
    }

    diagnostics = {
        "method": "grouped-ls",
        "distribution": "weibull",
        "sample_size": int(total_obs),
        "bins": x,
        "bin_edges": _bin_edges_from_centroids(x),
        "probabilities": np.clip(fitted_safe / fitted_sum, 1e-12, None),
        "observed": y,
        "expected": fitted,
        "residuals": residuals,
        "weights": weights,
        "fitted": fitted,
        "amplitude": fitted_sum,
    }

    param_dict = {"a": float(params[0]), "beta": float(params[1]), "s": float(params[2])}

    ls_probs = _weibull_probabilities(
        param_dict["a"],
        param_dict["beta"],
        edges,
        location_shift=location_shift,
    )
    diagnostics["probabilities"] = ls_probs

    ls_covariance = None
    if cov is not None:
        ls_covariance = np.zeros((3, 3), dtype=float)
        ls_covariance[:3, :3] = cov

    ls_result = FitResult(
        distribution="weibull",
        parameters=param_dict.copy(),
        covariance=ls_covariance,
        gof=gof,
        diagnostics=diagnostics,
    )
    if mode == "ls":
        return ls_result

    def neg_loglik(theta: np.ndarray) -> float:
        shape = float(np.exp(theta[0]))
        scale = float(np.exp(theta[1]))
        probs = _weibull_probabilities(
            shape,
            scale,
            edges,
            location_shift=location_shift,
        )
        return float(-np.sum(y * np.log(probs)))

    theta = np.log([param_dict["a"], param_dict["beta"]])
    nll_current = neg_loglik(theta)
    newton_converged = False

    last_iteration = 0
    for iteration in range(1, 9):
        last_iteration = iteration
        gradient = approx_fprime(theta, neg_loglik, epsilon=1e-6)
        if not np.all(np.isfinite(gradient)):
            break
        if np.linalg.norm(gradient) < 1e-5:
            newton_converged = True
            break
        hessian = _numerical_hessian(neg_loglik, theta)
        if hessian is None:
            break
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(step)):
            break
        theta_candidate = theta - step
        if not np.all(np.isfinite(theta_candidate)):
            break
        nll_candidate = neg_loglik(theta_candidate)
        if np.isnan(nll_candidate) or nll_candidate > nll_current:
            theta_candidate = theta - 0.5 * step
            nll_candidate = neg_loglik(theta_candidate)
            if np.isnan(nll_candidate) or nll_candidate > nll_current:
                break
        theta = theta_candidate
        nll_current = nll_candidate
        if np.linalg.norm(step) < 1e-6:
            newton_converged = True
            break
    else:
        newton_converged = True

    if not newton_converged:
        if mode == "mle":
            opt = minimize(neg_loglik, theta, method="L-BFGS-B")
            if not opt.success:
                raise RuntimeError("Grouped Newton refinement failed in forced 'mle' mode.")
            theta = opt.x
        else:
            notes = ls_result.diagnostics.setdefault("notes", [])
            notes.append("grouped Newton refinement failed – keeping least squares solution")
            return ls_result

    shape = float(np.exp(theta[0]))
    scale = float(np.exp(theta[1]))
    mle_probs = _weibull_probabilities(shape, scale, edges, location_shift=location_shift)

    amplitude = param_dict["s"]
    mle_params = {"a": shape, "beta": scale, "s": amplitude}

    covariance = None
    try:
        hessian = _numerical_hessian(neg_loglik, theta)
        if hessian is not None:
            cov_theta = np.linalg.inv(hessian)
            jac = np.diag([shape, scale])
            cov_params = jac @ cov_theta @ jac
            covariance = np.zeros((3, 3), dtype=float)
            covariance[:2, :2] = cov_params
    except np.linalg.LinAlgError:
        covariance = None

    extras: dict[str, float | int | bool | str | np.ndarray | None] = {
        "iterations": last_iteration,
        "converged": newton_converged,
        "status": None,
        "message": None,
        "weights": weights,
        "mode": mode,
    }

    mle_result = _assemble_grouped_result(
        "weibull",
        mle_params,
        y,
        edges,
        mle_probs,
        covariance=covariance,
        method="grouped-mle",
        amplitude=amplitude,
        extras=extras,
    )

    ls_vector = np.array([param_dict["a"], param_dict["beta"], param_dict["s"]])
    mle_vector = np.array([mle_params["a"], mle_params["beta"], mle_params["s"]])
    if not np.allclose(ls_vector, mle_vector, rtol=1e-3, atol=1e-3) and mode != "mle":
        notes = ls_result.diagnostics.setdefault("notes", [])
        notes.append(
            "grouped likelihood refinement deviated from LS – keeping least squares solution"
        )
        return ls_result

    return mle_result


def _fit_johnsonsb_grouped(
    inventory: InventorySpec,
    config: FitConfig | None,
) -> FitResult:
    if config is None:
        raise ValueError("Grouped estimator requires a FitConfig instance.")

    em_result = _fit_johnsonsb_em(inventory, config)
    if em_result is not None:
        return em_result

    bins = np.asarray(inventory.bins, dtype=float)
    defaults = {
        "a": 1.5,
        "b": 2.5,
        "loc": float(np.min(bins)) * 0.8 if bins.size else 0.0,
        "scale": max(float(np.std(bins, ddof=0)), 1.0),
    }

    def cdf_callable(edges: np.ndarray, params: dict[str, float]) -> np.ndarray:
        return johnsonsb.cdf(
            edges,
            a=params["a"],
            b=params["b"],
            loc=params["loc"],
            scale=params["scale"],
        )

    return _grouped_mle(
        inventory,
        config,
        distribution="johnsonsb",
        param_names=("a", "b", "loc", "scale"),
        cdf_callable=cdf_callable,
        defaults=defaults,
        positive_parameters=("a", "b", "scale"),
    )


def _fit_birnbaum_saunders_grouped(
    inventory: InventorySpec,
    config: FitConfig | None,
) -> FitResult:
    if config is None:
        raise ValueError("Grouped estimator requires a FitConfig instance.")

    em_result = _fit_birnbaum_saunders_em(inventory, config)
    if em_result is not None:
        return em_result

    bins = np.asarray(inventory.bins, dtype=float)
    defaults = {
        "alpha": 1.0,
        "beta": max(float(np.mean(bins)) if bins.size else 10.0, 1.0),
    }

    def cdf_callable(edges: np.ndarray, params: dict[str, float]) -> np.ndarray:
        return fatiguelife.cdf(edges, c=params["alpha"], loc=0.0, scale=params["beta"])

    return _grouped_mle(
        inventory,
        config,
        distribution="birnbaum_saunders",
        param_names=("alpha", "beta"),
        cdf_callable=cdf_callable,
        defaults=defaults,
        positive_parameters=("alpha", "beta"),
    )


def _make_gsm_grouped_estimator(components: int) -> GroupedEstimator:
    def estimator(inventory: InventorySpec, config: FitConfig | None) -> FitResult:
        if config is None:
            raise ValueError("Grouped estimator requires a FitConfig instance.")
        x, counts = _prepare_grouped_data(inventory)
        edges = _bin_edges_from_centroids(x)
        total = float(np.sum(counts))

        initial_map = dict(config.initial)
        default_beta = max(float(np.mean(x)) if x.size else 1.0, 1.0)
        initial_map.setdefault("beta", default_beta)
        omega = []
        for idx in range(1, components):
            omega.append(float(initial_map.get(f"omega{idx}", 1.0 / components)))
        omega.append(max(1.0 - float(np.sum(omega)), 1.0 / components))
        weights = np.asarray(omega, dtype=float)
        weights = np.clip(weights, 1e-6, None)
        weights = weights / float(np.sum(weights))
        beta = max(float(initial_map.get("beta", default_beta)), 1e-6)

        def component_probabilities(beta_value: float) -> np.ndarray:
            scale = 1.0 / max(beta_value, 1e-8)
            probabilities = np.zeros((components, edges.size - 1), dtype=float)
            for comp in range(components):
                upper = gamma_dist.cdf(edges[1:], a=comp + 1, scale=scale)
                lower = gamma_dist.cdf(edges[:-1], a=comp + 1, scale=scale)
                diff = upper - lower
                if lower[0] > 0:
                    diff[0] += lower[0]
                tail = 1.0 - upper[-1]
                if tail > 0:
                    diff[-1] += tail
                probabilities[comp] = np.clip(diff, 1e-12, None)
            return probabilities

        beta_upper = max(50.0, 5.0 * (float(edges[-1]) if edges.size else 10.0))
        tol = 1e-6
        max_iter = 200
        converged = False
        last_iteration = 0

        for idx in range(1, max_iter + 1):
            last_iteration = idx
            comp_prob = component_probabilities(beta)
            mixture = weights @ comp_prob
            mixture = np.clip(mixture, 1e-12, None)
            responsibilities = (weights[:, None] * comp_prob) / mixture
            weights_new = (responsibilities @ counts) / total
            weights_new = np.clip(weights_new, 1e-8, None)
            weights_new = weights_new / float(np.sum(weights_new))

            def neg_loglik(beta_value: float, weights_snapshot=weights_new) -> float:
                comp = component_probabilities(beta_value)
                mix = weights_snapshot @ comp
                if np.any(mix <= 0):
                    return np.inf
                return float(-np.sum(counts * np.log(mix)))

            beta_result = minimize_scalar(
                neg_loglik,
                bounds=(1e-6, beta_upper),
                method="bounded",
                options={"xatol": 1e-6},
            )
            if not beta_result.success:
                raise ValueError(
                    "Grouped optimisation failed for gsm: beta search did not converge."
                )

            beta_new = float(beta_result.x)
            delta = float(np.max(np.abs(weights_new - weights)) + abs(beta_new - beta))

            weights = weights_new
            beta = beta_new
            if delta < tol:
                converged = True
                break

        comp_prob = component_probabilities(beta)
        probabilities = np.clip(weights @ comp_prob, 1e-12, None)
        params = {"beta": float(beta)}
        for idx in range(1, components):
            params[f"omega{idx}"] = float(weights[idx - 1])

        extras: dict[str, float | int | bool | str | np.ndarray | None] = {
            "bins": x,
            "iterations": last_iteration,
            "converged": converged,
            "component_weights": weights,
            "component_probabilities": comp_prob,
            "omega_tail": float(weights[-1]),
            "status": "ok" if converged else "iteration-limit",
        }

        return _assemble_grouped_result(
            f"gsm{components}",
            params,
            counts,
            edges,
            probabilities,
            covariance=None,
            method="grouped-mle",
            amplitude=float(np.sum(counts)),
            extras=extras,
        )

    return estimator


_GROUPED_ESTIMATORS: dict[str, GroupedEstimator] = {
    "weibull": _fit_weibull_grouped,
    "johnsonsb": _fit_johnsonsb_grouped,
    "birnbaum_saunders": _fit_birnbaum_saunders_grouped,
}

_GSM_CACHE: dict[int, GroupedEstimator] = {}


def get_grouped_estimator(name: str) -> GroupedEstimator | None:
    """Return a grouped estimator for the given distribution, if available."""
    key = name.lower()
    estimator = _GROUPED_ESTIMATORS.get(key)
    if estimator is not None:
        return estimator
    if key.startswith("gsm"):
        suffix = key[3:]
        if suffix.isdigit():
            components = int(suffix)
            if components >= 2:
                if components not in _GSM_CACHE:
                    _GSM_CACHE[components] = _make_gsm_grouped_estimator(components)
                estimator = _GSM_CACHE[components]
                _GROUPED_ESTIMATORS[key] = estimator
                return estimator
    return None


def _birnbaum_normal_bounds(
    alpha: float,
    beta: float,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    if alpha <= 0 or beta <= 0:
        return None, None
    epsilon = 1e-12
    clipped = np.clip(edges, epsilon, None)
    sqrt_ratio = np.sqrt(clipped / beta)
    sqrt_inv = np.sqrt(beta / clipped)
    y = (sqrt_ratio - sqrt_inv) / alpha
    if np.any(~np.isfinite(y)):
        return None, None
    return y[:-1], y[1:]


def _solve_birnbaum_moments(mean: float, variance: float) -> tuple[float, float] | None:
    """Solve Birnbaum–Saunders moment identities for (alpha, beta)."""
    if mean <= 0 or variance <= 0:
        return None

    def equation(alpha_sq: float) -> float:
        beta = mean / (1.0 + alpha_sq / 2.0)
        return (beta**2 * alpha_sq / 2.0) * (1.0 + 5.0 * alpha_sq / 4.0) - variance

    try:
        root = brentq(equation, 1e-6, 100.0)
    except ValueError:
        return None
    alpha = float(np.sqrt(root))
    beta = float(mean / (1.0 + root / 2.0))
    if not np.isfinite(alpha) or not np.isfinite(beta):
        return None
    return alpha, beta


def _fit_birnbaum_saunders_em(
    inventory: InventorySpec,
    config: FitConfig,
    *,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> FitResult | None:
    x, counts = _prepare_grouped_data(inventory)
    edges = _bin_edges_from_centroids(x)
    total = float(np.sum(counts))
    if total <= 0:
        return None

    weights = counts.astype(float)
    mean = float(np.average(x, weights=weights))
    variance = float(np.average(np.square(x - mean), weights=weights))

    initial_map = dict(config.initial)
    moment_solution = _solve_birnbaum_moments(mean, variance)
    if moment_solution is not None:
        initial_map.setdefault("alpha", moment_solution[0])
        initial_map.setdefault("beta", moment_solution[1])
    alpha = float(np.clip(initial_map.get("alpha", 1.0), 0.05, 20.0))
    beta = float(
        np.clip(
            initial_map.get("beta", mean if x.size else 10.0),
            1e-3,
            np.max(edges) * 2.0 if edges.size else 10.0,
        )
    )

    best_loglik = -np.inf

    variance_was_clamped = False
    best_candidate: dict[str, float | np.ndarray | bool | int | str] | None = None
    if moment_solution is not None:
        cdf_vals = fatiguelife.cdf(edges, c=alpha, loc=0.0, scale=beta)
        if np.all(np.isfinite(cdf_vals)):
            moment_probs = np.diff(cdf_vals)
            if cdf_vals[0] > 0:
                moment_probs[0] += cdf_vals[0]
            tail_mass = 1.0 - cdf_vals[-1]
            if tail_mass > 0:
                moment_probs[-1] += tail_mass
            moment_probs = np.clip(moment_probs, 1e-12, None)
            moment_probs = moment_probs / float(np.sum(moment_probs))
            best_loglik = float(np.sum(counts * np.log(moment_probs)))
            best_candidate = cast(
                dict[str, float | np.ndarray | bool | int | str],
                {
                    "alpha": alpha,
                    "beta": beta,
                    "probabilities": moment_probs,
                    "iterations": 0,
                    "delta": 0.0,
                    "loglik": best_loglik,
                    "converged": True,
                    "variance_clamped": False,
                    "method_detail": "moment",
                },
            )
    for iteration in range(1, max_iter + 1):
        lower, upper = _birnbaum_normal_bounds(alpha, beta, edges)
        if lower is None or upper is None:
            break

        masses: list[float] = []
        means: list[float] = []
        second_moments: list[float] = []
        for low, up in zip(lower, upper, strict=False):
            mass, mean, mean_sq = _normal_interval_moments(low, up)
            masses.append(mass)
            means.append(mean)
            second_moments.append(mean_sq)
        probabilities = np.asarray(masses, dtype=float)
        if np.any(probabilities <= 0) or not np.all(np.isfinite(probabilities)):
            break
        probabilities = probabilities / float(np.sum(probabilities))
        loglik = float(np.sum(counts * np.log(probabilities)))
        if loglik < best_loglik - 1e-6:
            break
        best_loglik = loglik

        means_arr = np.asarray(means, dtype=float)
        second_arr = np.asarray(second_moments, dtype=float)
        t1 = float(np.sum(counts * means_arr))
        t2 = float(np.sum(counts * second_arr))

        variance_term = t2 / total - 1.0 - (t1 / total) ** 2
        if not np.isfinite(variance_term):
            break
        clamped_iteration = False
        if variance_term <= 1e-8:
            variance_term = 1e-8
            variance_was_clamped = True
            clamped_iteration = True
        raw_alpha = float(np.sqrt(variance_term))
        alpha_new = max(raw_alpha, 0.1 * alpha)

        def neg_loglik_beta(beta_candidate: float, current_alpha: float = alpha_new) -> float:
            if beta_candidate <= 1e-6 or not np.isfinite(beta_candidate):
                return float("inf")
            lower_candidate, upper_candidate = _birnbaum_normal_bounds(
                current_alpha,
                beta_candidate,
                edges,
            )
            if lower_candidate is None or upper_candidate is None:
                return float("inf")
            masses_candidate: list[float] = []
            for low_cand, up_cand in zip(lower_candidate, upper_candidate, strict=False):
                mass_cand, _, _ = _normal_interval_moments(low_cand, up_cand)
                masses_candidate.append(mass_cand)
            probs_candidate = np.asarray(masses_candidate, dtype=float)
            if np.any(probs_candidate <= 0) or not np.all(np.isfinite(probs_candidate)):
                return float("inf")
            probs_candidate = probs_candidate / float(np.sum(probs_candidate))
            return float(-np.sum(counts * np.log(probs_candidate)))

        beta_upper = max(10.0 * np.max(edges), beta * 10.0)
        search = minimize_scalar(
            neg_loglik_beta,
            bounds=(1e-6, beta_upper),
            method="bounded",
            options={"xatol": 1e-6},
        )
        if not search.success:
            break
        beta_new = float(search.x)

        lower_new, upper_new = _birnbaum_normal_bounds(alpha_new, beta_new, edges)
        if lower_new is None or upper_new is None:
            break
        masses_new: list[float] = []
        for low_new, up_new in zip(lower_new, upper_new, strict=False):
            mass_new, _, _ = _normal_interval_moments(low_new, up_new)
            masses_new.append(mass_new)
        probabilities_new = np.asarray(masses_new, dtype=float)
        if np.any(probabilities_new <= 0) or not np.all(np.isfinite(probabilities_new)):
            break
        probabilities_new = probabilities_new / float(np.sum(probabilities_new))
        new_loglik = float(np.sum(counts * np.log(probabilities_new)))

        delta_param = max(
            abs(alpha_new - alpha) / max(alpha, 1e-6),
            abs(beta_new - beta) / max(beta, 1e-6),
        )
        delta_loglik = abs(new_loglik - best_loglik)
        alpha = alpha_new
        beta = beta_new
        best_loglik = new_loglik
        candidate: dict[str, float | np.ndarray | bool | int | str] = {
            "alpha": alpha_new,
            "beta": beta_new,
            "probabilities": probabilities_new,
            "iterations": iteration,
            "delta": delta_param,
            "loglik": new_loglik,
            "converged": delta_param < tol and delta_loglik < tol * (1.0 + abs(best_loglik)),
            "method_detail": "em-normal-moments",
        }
        if variance_was_clamped or clamped_iteration:
            candidate["variance_clamped"] = True
        if best_candidate is None or new_loglik > float(best_candidate["loglik"]):
            best_candidate = candidate

        if candidate["converged"]:
            break

    if best_candidate is None:
        return None

    alpha = float(best_candidate["alpha"])
    beta = float(best_candidate["beta"])
    probabilities_final = np.asarray(best_candidate["probabilities"], dtype=float)
    if np.any(probabilities_final <= 0) or not np.all(np.isfinite(probabilities_final)):
        cdf_vals = fatiguelife.cdf(edges, c=alpha, loc=0.0, scale=beta)
        if np.any(~np.isfinite(cdf_vals)):
            return None
        probabilities_final = np.diff(cdf_vals)
        if cdf_vals[0] > 0:
            probabilities_final[0] += cdf_vals[0]
        tail_mass = 1.0 - cdf_vals[-1]
        if tail_mass > 0:
            probabilities_final[-1] += tail_mass
        probabilities_final = np.clip(probabilities_final, 1e-12, None)
        probabilities_final = probabilities_final / float(np.sum(probabilities_final))

    cdf_vals = fatiguelife.cdf(edges, c=alpha, loc=0.0, scale=beta)
    if np.any(~np.isfinite(cdf_vals)):
        return None
    if probabilities_final.shape[0] != counts.shape[0]:
        probabilities_final = np.diff(cdf_vals)
        if cdf_vals[0] > 0:
            probabilities_final[0] += cdf_vals[0]
        tail_mass = 1.0 - cdf_vals[-1]
        if tail_mass > 0:
            probabilities_final[-1] += tail_mass
        probabilities_final = np.clip(probabilities_final, 1e-12, None)
        probabilities_final = probabilities_final / float(np.sum(probabilities_final))

    params = {"alpha": alpha, "beta": beta}
    extras: dict[str, float | int | bool | str | np.ndarray | None] = {
        "iterations": int(best_candidate["iterations"]),
        "converged": bool(best_candidate["converged"]),
        "delta": float(best_candidate["delta"]),
        "method_detail": str(best_candidate.get("method_detail", "em-normal-moments")),
    }
    if variance_was_clamped or best_candidate.get("variance_clamped"):
        extras["variance_clamped"] = True
    return _assemble_grouped_result(
        "birnbaum_saunders",
        params,
        counts,
        edges,
        probabilities_final,
        covariance=None,
        method="grouped-em",
        amplitude=total,
        extras=extras,
    )
