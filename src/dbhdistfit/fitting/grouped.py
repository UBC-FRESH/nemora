"""Grouped-data estimators for selected distributions."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import curve_fit, minimize, minimize_scalar
from scipy.stats import fatiguelife, johnsonsb
from scipy.stats import gamma as gamma_dist

from ..distributions import weibull_pdf
from ..typing import FitResult, InventorySpec

if TYPE_CHECKING:  # pragma: no cover - typing aid
    from . import FitConfig

GroupedEstimator = Callable[[InventorySpec, "FitConfig | None"], FitResult]


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

    return FitResult(
        distribution="weibull",
        parameters=param_dict,
        covariance=cov,
        gof=gof,
        diagnostics=diagnostics,
    )


def _fit_johnsonsb_grouped(
    inventory: InventorySpec,
    config: FitConfig | None,
) -> FitResult:
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

    try:
        return _grouped_mle(
            inventory,
            config,
            distribution="johnsonsb",
            param_names=("a", "b", "loc", "scale"),
            cdf_callable=cdf_callable,
            defaults=defaults,
            positive_parameters=("a", "b", "scale"),
        )
    except ValueError as exc:
        x, counts = _prepare_grouped_data(inventory)
        sample = np.repeat(x, counts)
        if sample.size < 4:
            raise
        a_fit, b_fit, loc_fit, scale_fit = johnsonsb.fit(sample)
        params = {
            "a": float(a_fit),
            "b": float(b_fit),
            "loc": float(loc_fit),
            "scale": float(scale_fit),
        }
        edges = _bin_edges_from_centroids(x)
        cdf_vals = johnsonsb.cdf(edges, a=a_fit, b=b_fit, loc=loc_fit, scale=scale_fit)
        probabilities = np.diff(cdf_vals)
        if cdf_vals[0] > 0:
            probabilities[0] += cdf_vals[0]
        tail_mass = 1.0 - cdf_vals[-1]
        if tail_mass > 0:
            probabilities[-1] += tail_mass
        probabilities = np.clip(probabilities, 1e-12, None)
        extras: dict[str, float | int | bool | str | np.ndarray | None] = {
            "bins": x,
            "converged": False,
            "status": "fallback-scipy-fit",
            "message": str(exc),
            "iterations": 0,
        }
        return _assemble_grouped_result(
            "johnsonsb",
            params,
            counts,
            edges,
            probabilities,
            covariance=None,
            method="grouped-mle",
            amplitude=float(np.sum(counts)),
            extras=extras,
        )


def _fit_birnbaum_saunders_grouped(
    inventory: InventorySpec,
    config: FitConfig | None,
) -> FitResult:
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
