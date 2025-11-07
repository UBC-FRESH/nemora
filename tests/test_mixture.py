import numpy as np
import pytest

from nemora.distfit import (
    MixtureComponentSpec,
    fit_mixture_grouped,
    fit_mixture_samples,
    mixture_cdf,
    mixture_pdf,
    sample_mixture,
)


def _synth_gamma_mixture(
    rng: np.random.Generator,
    size: int = 4000,
) -> np.ndarray:
    part_a = rng.gamma(shape=2.0, scale=8.0, size=size // 2)
    part_b = rng.gamma(shape=6.0, scale=3.0, size=size // 2)
    return np.concatenate([part_a, part_b])


def test_fit_mixture_samples_gamma() -> None:
    rng = np.random.default_rng(1234)
    samples = _synth_gamma_mixture(rng, size=6000)

    specs = [
        MixtureComponentSpec("gamma"),
        MixtureComponentSpec("gamma"),
    ]
    result = fit_mixture_samples(samples, specs, bins=50, random_state=1234, max_iter=150)

    weights = np.array([component.weight for component in result.components])
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0)
    pdf_val = mixture_pdf([5.0], result.components)[0]
    assert pdf_val > 0
    cdf_val = mixture_cdf([5.0], result.components)[0]
    assert 0 <= cdf_val <= 1
    samples = sample_mixture(50, result.components, random_state=42)
    assert samples.shape == (50,)


def test_fit_mixture_grouped_converges_on_gamma_mix() -> None:
    rng = np.random.default_rng(2025)
    samples = _synth_gamma_mixture(rng, size=5000)
    hist, edges = np.histogram(samples, bins=40)
    mids = 0.5 * (edges[:-1] + edges[1:])
    specs = [
        MixtureComponentSpec("gamma"),
        MixtureComponentSpec("gamma"),
    ]
    result = fit_mixture_grouped(
        mids,
        hist,
        specs,
        max_iter=300,
        tol=1e-4,
        random_state=2025,
    )
    assert result.iterations > 0
    weights = np.array([component.weight for component in result.components])
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights > 0)
    responsibilities = result.diagnostics.get("responsibilities")
    assert isinstance(responsibilities, np.ndarray)
    assert responsibilities.shape[0] == len(specs)
    assert responsibilities.shape[1] == mids.size


def test_fit_mixture_grouped_errors() -> None:
    specs = [MixtureComponentSpec("gamma")]
    with pytest.raises(ValueError):
        fit_mixture_grouped([1.0, 2.0], [10, 20], specs)

    with pytest.raises(ValueError):
        fit_mixture_grouped(
            [1.0, 2.0],
            [-1, 20],
            [MixtureComponentSpec("gamma"), MixtureComponentSpec("gamma")],
        )
