"""Plot-conditioned stand validation utilities.

The validation compares a synthetic one-hectare-equivalent stand with held-out
FAIB plots. It fits a DBH distribution and a height–DBH relation on training
trees, generates a seeded synthetic tree list, and reports metrics, distribution
distance, species mix, and plot-resampling uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import FitResult, InventorySpec
from .fit import fit_inventory
from .sampling import sample_distribution

__all__ = [
    "ModelComparison",
    "StandMetrics",
    "StandValidationResult",
    "compare_stand_models",
    "generate_empirical_stand",
    "stand_metrics_from_trees",
    "validate_empirical_stand",
    "validate_plot_conditioned_stand",
    "validate_species_weibull_stand",
]

_TREE_COLUMNS = ("plot_id", "species_code", "dbh_cm", "height_m", "tally")


@dataclass(frozen=True, slots=True)
class StandMetrics:
    """Summary metrics for one observed plot or one synthetic stand."""

    trees: int
    stems_per_ha: float
    basal_area_m2_per_ha: float
    qmd_cm: float
    mean_dbh_cm: float
    mean_height_m: float | None
    species_proportions: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable mapping."""
        return {
            "trees": self.trees,
            "stems_per_ha": self.stems_per_ha,
            "basal_area_m2_per_ha": self.basal_area_m2_per_ha,
            "qmd_cm": self.qmd_cm,
            "mean_dbh_cm": self.mean_dbh_cm,
            "mean_height_m": self.mean_height_m,
            "species_proportions": dict(self.species_proportions),
        }


@dataclass(frozen=True, slots=True)
class StandValidationResult:
    """Result of a plot-conditioned stand validation."""

    training: StandMetrics
    holdout: StandMetrics
    synthetic: StandMetrics
    relative_errors: dict[str, float | None]
    dbh_ks_weighted: float
    species_tv_distance: float
    fit_parameters: dict[str, float]
    height_fit: dict[str, float]
    uncertainty: dict[str, dict[str, float]]
    bootstrap_replicates: int
    successful_bootstrap: int
    seed: int
    holdout_plots: dict[str, StandMetrics]
    synthetic_trees: pd.DataFrame
    model: str = "pooled_weibull"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable summary without the synthetic tree rows."""
        return {
            "model": self.model,
            "training": self.training.to_dict(),
            "holdout": self.holdout.to_dict(),
            "synthetic": self.synthetic.to_dict(),
            "relative_errors": dict(self.relative_errors),
            "dbh_ks_weighted": self.dbh_ks_weighted,
            "species_tv_distance": self.species_tv_distance,
            "fit_parameters": dict(self.fit_parameters),
            "height_fit": dict(self.height_fit),
            "uncertainty": {key: dict(value) for key, value in self.uncertainty.items()},
            "bootstrap_replicates": self.bootstrap_replicates,
            "successful_bootstrap": self.successful_bootstrap,
            "seed": self.seed,
            "holdout_plots": {key: value.to_dict() for key, value in self.holdout_plots.items()},
        }

    def comparison_frame(self) -> pd.DataFrame:
        """Return a tidy metric table for reports."""
        rows: list[dict[str, object]] = []
        training = self.training.to_dict()
        holdout = self.holdout.to_dict()
        synthetic = self.synthetic.to_dict()
        for metric in (
            "stems_per_ha",
            "basal_area_m2_per_ha",
            "qmd_cm",
            "mean_dbh_cm",
            "mean_height_m",
        ):
            rows.append(
                {
                    "metric": metric,
                    "training": training[metric],
                    "holdout": holdout[metric],
                    "synthetic": synthetic[metric],
                    "relative_error": self.relative_errors.get(metric),
                }
            )
        return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class ModelComparison:
    """Comparison of candidate stand models on one held-out plot set."""

    results: tuple[StandValidationResult, ...]
    scores: dict[str, float]
    selected: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable comparison summary."""
        return {
            "selected": self.selected,
            "scores": dict(self.scores),
            "models": {result.model: result.to_dict() for result in self.results},
        }

    def comparison_frame(self) -> pd.DataFrame:
        """Return a compact model-comparison table."""
        rows: list[dict[str, object]] = []
        for result in self.results:
            rows.append(
                {
                    "model": result.model,
                    "score": self.scores[result.model],
                    "dbh_ks_weighted": result.dbh_ks_weighted,
                    "species_tv_distance": result.species_tv_distance,
                    "ba_relative_error": result.relative_errors.get("basal_area_m2_per_ha"),
                    "qmd_relative_error": result.relative_errors.get("qmd_cm"),
                    "dbh_relative_error": result.relative_errors.get("mean_dbh_cm"),
                    "height_relative_error": result.relative_errors.get("mean_height_m"),
                }
            )
        return pd.DataFrame(rows)


def stand_metrics_from_trees(
    trees: pd.DataFrame,
    *,
    area_ha: float = 1.0,
) -> StandMetrics:
    """Compute stand metrics from tree rows with expansion tallies.

    Parameters
    ----------
    trees:
        Tree rows with ``dbh_cm``, ``tally``, optional ``height_m``, and
        optional ``species_code``.
    area_ha:
        Area represented by the rows. Observed FAIB plot rows use the default
        one-hectare expansion; a synthetic stand can pass its actual area.

    Raises
    ------
    KeyError
        If required tree columns are missing.
    ValueError
        If ``area_ha`` is not positive or no valid tree rows remain.
    """
    if area_ha <= 0.0 or not np.isfinite(area_ha):
        raise ValueError("area_ha must be finite and positive")
    _require_tree_columns(trees, ("dbh_cm", "tally"))
    dbh = pd.to_numeric(trees["dbh_cm"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(trees["tally"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    mask = np.isfinite(dbh) & (dbh > 0.0) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        raise ValueError("no valid tree rows remain")
    dbh = dbh[mask]
    weights = weights[mask]
    total_weight = float(weights.sum())
    basal_area = float(np.sum(np.pi * np.square(dbh / 200.0) * weights) / area_ha)
    qmd = float(np.sqrt(np.sum(np.square(dbh) * weights) / total_weight))
    mean_dbh = float(np.sum(dbh * weights) / total_weight)

    heights = pd.to_numeric(trees["height_m"], errors="coerce").to_numpy(dtype=float)
    height_mask = mask & np.isfinite(heights) & (heights > 0.0)
    mean_height = (
        float(np.sum(heights[height_mask] * weights[height_mask]) / weights[height_mask].sum())
        if np.any(height_mask)
        else None
    )

    if "species_code" in trees.columns:
        species = trees["species_code"].astype("string").fillna("UNKNOWN").to_numpy()
        species = species[mask]
        species_frame = pd.DataFrame({"species": species, "weight": weights})
        grouped = species_frame.groupby("species")["weight"].sum()
        species_proportions = {
            str(key): float(value / total_weight) for key, value in grouped.items()
        }
    else:
        species_proportions = {}

    return StandMetrics(
        trees=int(np.count_nonzero(mask)),
        stems_per_ha=total_weight / area_ha,
        basal_area_m2_per_ha=basal_area,
        qmd_cm=qmd,
        mean_dbh_cm=mean_dbh,
        mean_height_m=mean_height,
        species_proportions=species_proportions,
    )


def validate_plot_conditioned_stand(
    training_trees: pd.DataFrame,
    holdout_trees: pd.DataFrame,
    *,
    bin_width_cm: float = 2.0,
    area_ha: float = 1.0,
    bootstrap_replicates: int = 100,
    seed: int = 20261030,
) -> StandValidationResult:
    """Fit and validate a synthetic stand against held-out plots.

    Parameters
    ----------
    training_trees:
        Tree rows from training plots, with ``PHF_TREE`` expansion tallies.
    holdout_trees:
        Tree rows from held-out plots.
    bin_width_cm:
        DBH histogram width for the grouped Weibull fit.
    area_ha:
        Synthetic stand area. Metrics are reported per hectare.
    bootstrap_replicates:
        Plot-resampling replicates. Set to zero to skip uncertainty.
    seed:
        Seed for the synthetic stand and plot bootstrap.

    Raises
    ------
    ValueError
        If either input has no valid plots/trees or the fit has too few bins.
    """
    _require_tree_columns(training_trees, _TREE_COLUMNS)
    _require_tree_columns(holdout_trees, _TREE_COLUMNS)
    training_plots = _plot_metrics(training_trees)
    holdout_plots = _plot_metrics(holdout_trees)
    training = _mean_metrics(list(training_plots.values()))
    holdout = _mean_metrics(list(holdout_plots.values()))
    fit_parameters, height_fit, synthetic_trees, synthetic = _fit_and_generate(
        training_trees,
        training,
        bin_width_cm=bin_width_cm,
        area_ha=area_ha,
        seed=seed,
    )
    holdout_dbh = pd.to_numeric(holdout_trees["dbh_cm"], errors="coerce").to_numpy(dtype=float)
    holdout_weights = (
        pd.to_numeric(holdout_trees["tally"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    )
    synthetic_dbh = synthetic_trees["dbh_cm"].to_numpy(dtype=float)
    ks = _weighted_ks(
        synthetic_dbh,
        np.ones_like(synthetic_dbh),
        holdout_dbh,
        holdout_weights,
    )
    tv = _species_tv(synthetic.species_proportions, holdout.species_proportions)
    relative_errors = _relative_errors(synthetic, holdout)
    uncertainty, successful = _bootstrap_uncertainty(
        training_trees,
        bin_width_cm=bin_width_cm,
        area_ha=area_ha,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    return StandValidationResult(
        training=training,
        holdout=holdout,
        synthetic=synthetic,
        relative_errors=relative_errors,
        dbh_ks_weighted=ks,
        species_tv_distance=tv,
        fit_parameters=fit_parameters,
        height_fit=height_fit,
        uncertainty=uncertainty,
        bootstrap_replicates=bootstrap_replicates,
        successful_bootstrap=successful,
        seed=seed,
        holdout_plots=holdout_plots,
        synthetic_trees=synthetic_trees,
    )


def generate_empirical_stand(
    trees: pd.DataFrame,
    *,
    area_ha: float = 1.0,
    seed: int = 20261030,
) -> pd.DataFrame:
    """Generate a one-hectare-equivalent empirical tree list.

    This is a scenario/synthesis helper, not a validation result. It samples
    training tree rows with replacement, weighted by ``tally``.
    """
    _require_tree_columns(trees, _TREE_COLUMNS)
    metrics = _mean_metrics(list(_plot_metrics(trees).values()))
    return _generate_empirical(
        trees,
        count=max(1, int(round(metrics.stems_per_ha * area_ha))),
        seed=seed,
    )


def validate_empirical_stand(
    training_trees: pd.DataFrame,
    holdout_trees: pd.DataFrame,
    *,
    area_ha: float = 1.0,
    bootstrap_replicates: int = 100,
    seed: int = 20261030,
) -> StandValidationResult:
    """Validate a plot-level empirical stand against held-out plots.

    The empirical candidate samples training tree rows with replacement,
    weighted by their expansion tallies, and keeps the observed DBH/height
    pairs. It is a non-parametric alternative to the pooled Weibull model.
    """
    _require_tree_columns(training_trees, _TREE_COLUMNS)
    _require_tree_columns(holdout_trees, _TREE_COLUMNS)
    training_plots = _plot_metrics(training_trees)
    holdout_plots = _plot_metrics(holdout_trees)
    training = _mean_metrics(list(training_plots.values()))
    holdout = _mean_metrics(list(holdout_plots.values()))
    synthetic_trees = _generate_empirical(
        training_trees,
        count=max(1, int(round(training.stems_per_ha * area_ha))),
        seed=seed,
    )
    synthetic = stand_metrics_from_trees(synthetic_trees, area_ha=area_ha)
    ks = _weighted_ks(
        synthetic_trees["dbh_cm"].to_numpy(dtype=float),
        np.ones(len(synthetic_trees), dtype=float),
        pd.to_numeric(holdout_trees["dbh_cm"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(holdout_trees["tally"], errors="coerce").fillna(1.0).to_numpy(dtype=float),
    )
    tv = _species_tv(synthetic.species_proportions, holdout.species_proportions)
    uncertainty, successful = _bootstrap_empirical_uncertainty(
        training_trees,
        area_ha=area_ha,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    return StandValidationResult(
        training=training,
        holdout=holdout,
        synthetic=synthetic,
        relative_errors=_relative_errors(synthetic, holdout),
        dbh_ks_weighted=ks,
        species_tv_distance=tv,
        fit_parameters={},
        height_fit={},
        uncertainty=uncertainty,
        bootstrap_replicates=bootstrap_replicates,
        successful_bootstrap=successful,
        seed=seed,
        holdout_plots=holdout_plots,
        synthetic_trees=synthetic_trees,
        model="empirical",
    )


def validate_species_weibull_stand(
    training_trees: pd.DataFrame,
    holdout_trees: pd.DataFrame,
    *,
    bin_width_cm: float = 2.0,
    area_ha: float = 1.0,
    min_species_trees: int = 25,
    seed: int = 20261030,
) -> StandValidationResult:
    """Validate species-stratified Weibull fits against held-out plots.

    Species with fewer than ``min_species_trees`` training records fall back to
    the pooled Weibull fit. Rare-species overfitting is therefore reported
    rather than hidden.
    """
    _require_tree_columns(training_trees, _TREE_COLUMNS)
    _require_tree_columns(holdout_trees, _TREE_COLUMNS)
    training_plots = _plot_metrics(training_trees)
    holdout_plots = _plot_metrics(holdout_trees)
    training = _mean_metrics(list(training_plots.values()))
    holdout = _mean_metrics(list(holdout_plots.values()))
    pooled = _fit_dbh(training_trees, bin_width_cm)
    height_fit = _fit_height(training_trees)
    max_dbh = float(pd.to_numeric(training_trees["dbh_cm"], errors="coerce").max())
    synthetic_trees, fallbacks = _generate_species_weibull(
        training_trees,
        training,
        pooled.parameters,
        height_fit,
        bin_width_cm=bin_width_cm,
        min_species_trees=min_species_trees,
        max_dbh_cm=max_dbh,
        area_ha=area_ha,
        seed=seed,
    )
    synthetic = stand_metrics_from_trees(synthetic_trees, area_ha=area_ha)
    ks = _weighted_ks(
        synthetic_trees["dbh_cm"].to_numpy(dtype=float),
        np.ones(len(synthetic_trees), dtype=float),
        pd.to_numeric(holdout_trees["dbh_cm"], errors="coerce").to_numpy(dtype=float),
        pd.to_numeric(holdout_trees["tally"], errors="coerce").fillna(1.0).to_numpy(dtype=float),
    )
    tv = _species_tv(synthetic.species_proportions, holdout.species_proportions)
    return StandValidationResult(
        training=training,
        holdout=holdout,
        synthetic=synthetic,
        relative_errors=_relative_errors(synthetic, holdout),
        dbh_ks_weighted=ks,
        species_tv_distance=tv,
        fit_parameters=pooled.parameters,
        height_fit=height_fit,
        uncertainty={"fallback_species": {"count": float(len(fallbacks))}},
        bootstrap_replicates=0,
        successful_bootstrap=0,
        seed=seed,
        holdout_plots=holdout_plots,
        synthetic_trees=synthetic_trees,
        model="species_weibull",
    )


def compare_stand_models(
    training_trees: pd.DataFrame,
    holdout_trees: pd.DataFrame,
    *,
    bin_width_cm: float = 2.0,
    area_ha: float = 1.0,
    seed: int = 20261030,
) -> ModelComparison:
    """Compare pooled, species-stratified, and empirical stand models.

    The score is the mean absolute relative error over the five stand metrics
    plus the weighted DBH KS and species total-variation distance. Lower is
    better. The baseline is retained when scores tie or when no candidate
    improves.
    """
    results = (
        validate_plot_conditioned_stand(
            training_trees,
            holdout_trees,
            bin_width_cm=bin_width_cm,
            area_ha=area_ha,
            bootstrap_replicates=0,
            seed=seed,
        ),
        validate_species_weibull_stand(
            training_trees,
            holdout_trees,
            bin_width_cm=bin_width_cm,
            area_ha=area_ha,
            seed=seed,
        ),
        validate_empirical_stand(
            training_trees,
            holdout_trees,
            area_ha=area_ha,
            bootstrap_replicates=0,
            seed=seed,
        ),
    )
    scores = {result.model: _model_score(result) for result in results}
    selected = min(scores, key=lambda name: scores[name])
    if scores.get("pooled_weibull", float("inf")) <= scores[selected] + 1e-12:
        selected = "pooled_weibull"
    return ModelComparison(results=results, scores=scores, selected=selected)


def _model_score(result: StandValidationResult) -> float:
    errors = [abs(value) for value in result.relative_errors.values() if value is not None]
    error_score = float(np.mean(errors)) if errors else 0.0
    return error_score + result.dbh_ks_weighted + result.species_tv_distance


def _fit_and_generate(
    training_trees: pd.DataFrame,
    training_metrics: StandMetrics,
    *,
    bin_width_cm: float,
    area_ha: float,
    seed: int,
) -> tuple[dict[str, float], dict[str, float], pd.DataFrame, StandMetrics]:
    if bin_width_cm <= 0.0 or not np.isfinite(bin_width_cm):
        raise ValueError("bin_width_cm must be finite and positive")
    fit = _fit_dbh(training_trees, bin_width_cm)
    height_fit = _fit_height(training_trees)
    max_dbh = float(pd.to_numeric(training_trees["dbh_cm"], errors="coerce").max())
    synthetic = _generate_synthetic(
        fit.parameters,
        height_fit,
        training_metrics,
        max_dbh_cm=max_dbh,
        area_ha=area_ha,
        seed=seed,
    )
    metrics = stand_metrics_from_trees(synthetic, area_ha=area_ha)
    return fit.parameters, height_fit, synthetic, metrics


def _fit_dbh(trees: pd.DataFrame, bin_width_cm: float) -> FitResult:
    dbh = pd.to_numeric(trees["dbh_cm"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(trees["tally"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    mask = np.isfinite(dbh) & (dbh > 0.0) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(mask):
        raise ValueError("no valid training trees for the DBH fit")
    dbh = dbh[mask]
    weights = weights[mask]
    maximum = float(np.max(dbh))
    edges = np.arange(0.0, maximum + (2.0 * bin_width_cm), bin_width_cm)
    counts, _ = np.histogram(dbh, bins=edges, weights=weights)
    centers = (edges[:-1] + edges[1:]) / 2.0
    populated = counts > 0.0
    if np.count_nonzero(populated) < 3:
        raise ValueError("DBH fit needs at least three populated bins")
    inventory = InventorySpec(
        name="plot-conditioned-training",
        sampling="faib-psp",
        bins=centers[populated],
        tallies=counts[populated],
        metadata={"grouped": True},
    )
    return fit_inventory(inventory, ["weibull"], {})[0]


def _fit_height(trees: pd.DataFrame) -> dict[str, float]:
    dbh = pd.to_numeric(trees["dbh_cm"], errors="coerce").to_numpy(dtype=float)
    height = pd.to_numeric(trees["height_m"], errors="coerce").to_numpy(dtype=float)
    weights = pd.to_numeric(trees["tally"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    mask = (
        np.isfinite(dbh)
        & (dbh > 0.0)
        & np.isfinite(height)
        & (height > 0.0)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if np.count_nonzero(mask) < 2:
        raise ValueError("height fit needs at least two measured trees")
    log_dbh = np.log(dbh[mask])
    log_height = np.log(height[mask])
    slope, intercept = np.polyfit(log_dbh, log_height, 1)
    residual = log_height - (intercept + slope * log_dbh)
    return {
        "intercept": float(intercept),
        "slope": float(slope),
        "residual_std": float(np.std(residual, ddof=1)),
        "n": float(np.count_nonzero(mask)),
    }


def _generate_synthetic(
    fit_parameters: dict[str, float],
    height_fit: dict[str, float],
    training: StandMetrics,
    *,
    max_dbh_cm: float,
    area_ha: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    count = max(1, int(round(training.stems_per_ha * area_ha)))
    dbh = sample_distribution(
        "weibull",
        fit_parameters,
        size=count,
        random_state=rng,
    )
    dbh = np.clip(dbh, 1.0, max(1.0, max_dbh_cm))
    dbh = dbh[np.isfinite(dbh) & (dbh > 0.0)]
    noise = rng.normal(0.0, height_fit["residual_std"], size=dbh.size)
    height = np.exp(height_fit["intercept"] + height_fit["slope"] * np.log(dbh) + noise)
    height = np.clip(height, 0.5, None)
    species_codes = list(training.species_proportions)
    if species_codes:
        probabilities = np.asarray(
            [training.species_proportions[code] for code in species_codes],
            dtype=float,
        )
        probabilities = probabilities / probabilities.sum()
        species = rng.choice(species_codes, size=dbh.size, p=probabilities)
    else:
        species = np.full(dbh.size, "UNKNOWN", dtype=object)
    return pd.DataFrame(
        {
            "plot_id": "synthetic",
            "tree_id": [f"SYN-{index:06d}" for index in range(dbh.size)],
            "species_code": species,
            "dbh_cm": dbh,
            "height_m": height,
            "tally": 1.0,
            "status": "live",
        }
    )


def _generate_empirical(
    training_trees: pd.DataFrame,
    *,
    count: int,
    seed: int,
) -> pd.DataFrame:
    weights = pd.to_numeric(training_trees["tally"], errors="coerce").fillna(1.0)
    probabilities = weights.to_numpy(dtype=float)
    probabilities = np.clip(probabilities, 0.0, None)
    if probabilities.sum() <= 0.0:
        raise ValueError("training trees have no positive expansion tallies")
    probabilities = probabilities / probabilities.sum()
    rng = np.random.default_rng(seed)
    indices = rng.choice(np.arange(len(training_trees)), size=count, replace=True, p=probabilities)
    selected = training_trees.iloc[indices][["species_code", "dbh_cm", "height_m"]].copy()
    selected["plot_id"] = "synthetic"
    selected["tree_id"] = [f"EMP-{index:06d}" for index in range(len(selected))]
    selected["tally"] = 1.0
    selected["status"] = "live"
    return selected.reset_index(drop=True)


def _generate_species_weibull(
    training_trees: pd.DataFrame,
    training_metrics: StandMetrics,
    pooled_parameters: dict[str, float],
    height_fit: dict[str, float],
    *,
    bin_width_cm: float,
    min_species_trees: int,
    max_dbh_cm: float,
    area_ha: float,
    seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    if min_species_trees < 1:
        raise ValueError("min_species_trees must be positive")
    rng = np.random.default_rng(seed)
    count = max(1, int(round(training_metrics.stems_per_ha * area_ha)))
    rows: list[dict[str, object]] = []
    fallbacks: list[str] = []
    for species, proportion in training_metrics.species_proportions.items():
        species_count = max(1, int(round(count * proportion)))
        subset = training_trees.loc[training_trees["species_code"].astype(str) == species]
        try:
            if len(subset) < min_species_trees:
                raise ValueError("small species sample")
            fit = _fit_dbh(subset, bin_width_cm)
            parameters = fit.parameters
        except ValueError:
            fallbacks.append(str(species))
            parameters = pooled_parameters
        dbh = sample_distribution(
            "weibull",
            parameters,
            size=species_count,
            random_state=rng,
        )
        dbh = np.clip(dbh, 1.0, max(1.0, max_dbh_cm))
        dbh = dbh[np.isfinite(dbh) & (dbh > 0.0)]
        height = np.exp(
            height_fit["intercept"]
            + height_fit["slope"] * np.log(dbh)
            + rng.normal(0.0, height_fit["residual_std"], size=dbh.size)
        )
        height = np.clip(height, 0.5, None)
        for dbh_value, height_value in zip(dbh, height, strict=True):
            rows.append(
                {
                    "plot_id": "synthetic",
                    "species_code": str(species),
                    "dbh_cm": float(dbh_value),
                    "height_m": float(height_value),
                    "tally": 1.0,
                    "status": "live",
                }
            )
    result = pd.DataFrame(rows)
    result["tree_id"] = [f"SPC-{index:06d}" for index in range(len(result))]
    return result, fallbacks


def _plot_metrics(trees: pd.DataFrame) -> dict[str, StandMetrics]:
    _require_tree_columns(trees, ("plot_id", "dbh_cm", "tally"))
    result: dict[str, StandMetrics] = {}
    for plot_id, frame in trees.groupby("plot_id", sort=True):
        result[str(plot_id)] = stand_metrics_from_trees(frame)
    if not result:
        raise ValueError("no plot rows are available")
    return result


def _mean_metrics(metrics: list[StandMetrics]) -> StandMetrics:
    if not metrics:
        raise ValueError("no metrics to average")
    keys = sorted({key for item in metrics for key in item.species_proportions})
    species = {
        key: float(np.mean([item.species_proportions.get(key, 0.0) for item in metrics]))
        for key in keys
    }
    species_total = sum(species.values())
    if species_total > 0.0:
        species = {key: value / species_total for key, value in species.items()}
    heights = [item.mean_height_m for item in metrics if item.mean_height_m is not None]
    return StandMetrics(
        trees=int(round(np.mean([item.trees for item in metrics]))),
        stems_per_ha=float(np.mean([item.stems_per_ha for item in metrics])),
        basal_area_m2_per_ha=float(np.mean([item.basal_area_m2_per_ha for item in metrics])),
        qmd_cm=float(np.mean([item.qmd_cm for item in metrics])),
        mean_dbh_cm=float(np.mean([item.mean_dbh_cm for item in metrics])),
        mean_height_m=float(np.mean(heights)) if heights else None,
        species_proportions=species,
    )


def _relative_errors(
    synthetic: StandMetrics,
    holdout: StandMetrics,
) -> dict[str, float | None]:
    values = {
        "stems_per_ha": (synthetic.stems_per_ha, holdout.stems_per_ha),
        "basal_area_m2_per_ha": (
            synthetic.basal_area_m2_per_ha,
            holdout.basal_area_m2_per_ha,
        ),
        "qmd_cm": (synthetic.qmd_cm, holdout.qmd_cm),
        "mean_dbh_cm": (synthetic.mean_dbh_cm, holdout.mean_dbh_cm),
        "mean_height_m": (
            synthetic.mean_height_m,
            holdout.mean_height_m,
        ),
    }
    errors: dict[str, float | None] = {}
    for key, (observed, expected) in values.items():
        if observed is None or expected is None or expected == 0.0:
            errors[key] = None
        else:
            errors[key] = float((observed - expected) / expected)
    return errors


def _weighted_ks(
    values_a: np.ndarray,
    weights_a: np.ndarray,
    values_b: np.ndarray,
    weights_b: np.ndarray,
) -> float:
    mask_a = np.isfinite(values_a) & np.isfinite(weights_a) & (weights_a > 0.0)
    mask_b = np.isfinite(values_b) & np.isfinite(weights_b) & (weights_b > 0.0)
    values_a = values_a[mask_a]
    weights_a = weights_a[mask_a]
    values_b = values_b[mask_b]
    weights_b = weights_b[mask_b]
    if values_a.size == 0 or values_b.size == 0:
        return float("nan")
    grid = np.unique(np.concatenate([values_a, values_b]))
    cdf_a = _weighted_cdf(values_a, weights_a, grid)
    cdf_b = _weighted_cdf(values_b, weights_b, grid)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _weighted_cdf(values: np.ndarray, weights: np.ndarray, grid: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    positions = np.searchsorted(sorted_values, grid, side="right")
    result = np.zeros(grid.size, dtype=float)
    nonzero = positions > 0
    result[nonzero] = cumulative[positions[nonzero] - 1] / cumulative[-1]
    return result


def _species_tv(
    synthetic: dict[str, float],
    holdout: dict[str, float],
) -> float:
    keys = set(synthetic) | set(holdout)
    return float(0.5 * sum(abs(synthetic.get(key, 0.0) - holdout.get(key, 0.0)) for key in keys))


def _bootstrap_empirical_uncertainty(
    training_trees: pd.DataFrame,
    *,
    area_ha: float,
    replicates: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]], int]:
    if replicates < 1:
        return {}, 0
    plot_ids = sorted(training_trees["plot_id"].astype(str).unique())
    if len(plot_ids) < 2:
        return {}, 0
    rng = np.random.default_rng(seed + 2)
    samples: dict[str, list[float]] = {
        "stems_per_ha": [],
        "basal_area_m2_per_ha": [],
        "qmd_cm": [],
        "mean_dbh_cm": [],
        "mean_height_m": [],
    }
    successful = 0
    for index in range(replicates):
        sampled_ids = rng.choice(plot_ids, size=len(plot_ids), replace=True)
        pieces = []
        for sample_index, plot_id in enumerate(sampled_ids):
            frame = training_trees.loc[training_trees["plot_id"].astype(str) == plot_id].copy()
            frame["plot_id"] = f"{plot_id}__empirical{sample_index}"
            pieces.append(frame)
        sample = pd.concat(pieces, ignore_index=True)
        try:
            training_metrics = _mean_metrics(list(_plot_metrics(sample).values()))
            synthetic_trees = _generate_empirical(
                sample,
                count=max(1, int(round(training_metrics.stems_per_ha * area_ha))),
                seed=seed + index + 1,
            )
            synthetic = stand_metrics_from_trees(synthetic_trees, area_ha=area_ha)
        except ValueError:
            continue
        successful += 1
        samples["stems_per_ha"].append(synthetic.stems_per_ha)
        samples["basal_area_m2_per_ha"].append(synthetic.basal_area_m2_per_ha)
        samples["qmd_cm"].append(synthetic.qmd_cm)
        samples["mean_dbh_cm"].append(synthetic.mean_dbh_cm)
        if synthetic.mean_height_m is not None:
            samples["mean_height_m"].append(synthetic.mean_height_m)
    uncertainty: dict[str, dict[str, float]] = {}
    for key, values in samples.items():
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        uncertainty[key] = {
            "p05": float(np.percentile(array, 5.0)),
            "median": float(np.percentile(array, 50.0)),
            "p95": float(np.percentile(array, 95.0)),
            "replicates": float(array.size),
        }
    return uncertainty, successful


def _bootstrap_uncertainty(
    training_trees: pd.DataFrame,
    *,
    bin_width_cm: float,
    area_ha: float,
    replicates: int,
    seed: int,
) -> tuple[dict[str, dict[str, float]], int]:
    if replicates < 1:
        return {}, 0
    plot_ids = sorted(training_trees["plot_id"].astype(str).unique())
    if len(plot_ids) < 2:
        return {}, 0
    rng = np.random.default_rng(seed + 1)
    samples: dict[str, list[float]] = {
        "stems_per_ha": [],
        "basal_area_m2_per_ha": [],
        "qmd_cm": [],
        "mean_dbh_cm": [],
        "mean_height_m": [],
    }
    successful = 0
    for index in range(replicates):
        sampled_ids = rng.choice(plot_ids, size=len(plot_ids), replace=True)
        pieces = []
        for sample_index, plot_id in enumerate(sampled_ids):
            frame = training_trees.loc[training_trees["plot_id"].astype(str) == plot_id].copy()
            frame["plot_id"] = f"{plot_id}__bootstrap{sample_index}"
            pieces.append(frame)
        sample = pd.concat(pieces, ignore_index=True)
        try:
            training_metrics = _mean_metrics(list(_plot_metrics(sample).values()))
            _, _, _, synthetic = _fit_and_generate(
                sample,
                training_metrics,
                bin_width_cm=bin_width_cm,
                area_ha=area_ha,
                seed=seed + index + 1,
            )
        except (ValueError, FloatingPointError):
            continue
        successful += 1
        samples["stems_per_ha"].append(synthetic.stems_per_ha)
        samples["basal_area_m2_per_ha"].append(synthetic.basal_area_m2_per_ha)
        samples["qmd_cm"].append(synthetic.qmd_cm)
        samples["mean_dbh_cm"].append(synthetic.mean_dbh_cm)
        if synthetic.mean_height_m is not None:
            samples["mean_height_m"].append(synthetic.mean_height_m)
    uncertainty: dict[str, dict[str, float]] = {}
    for key, values in samples.items():
        if not values:
            continue
        array = np.asarray(values, dtype=float)
        uncertainty[key] = {
            "p05": float(np.percentile(array, 5.0)),
            "median": float(np.percentile(array, 50.0)),
            "p95": float(np.percentile(array, 95.0)),
            "replicates": float(array.size),
        }
    return uncertainty, successful


def _require_tree_columns(frame: pd.DataFrame, columns: tuple[str, ...]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError("tree table is missing required columns: " + ", ".join(missing))
