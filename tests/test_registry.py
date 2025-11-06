import importlib
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nemora.distributions import (
    GENERALIZED_BETA_DISTRIBUTIONS,
    clear_registry,
    get_distribution,
    list_distributions,
)
from nemora.distributions import base as base_registry


def _dummy_pdf(x: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    return np.full_like(x, params.get("scale", 1.0), dtype=float)


def _reload_registry() -> None:
    """Reload the distributions module to restore built-ins after tests."""
    import nemora.distributions as dist_module

    importlib.reload(dist_module)


class _DummyEntryPoint:
    def __init__(self, name: str, obj: Any) -> None:
        self.name = name
        self._obj = obj

    def load(self) -> Any:
        return self._obj


def _make_entry_points(result: Iterable[_DummyEntryPoint]) -> Any:
    class _EntryPoints(list):
        def __init__(self, values: Iterable[_DummyEntryPoint]) -> None:
            super().__init__(values)

        def select(self, *, group: str) -> list[_DummyEntryPoint]:
            return list(self)

    return _EntryPoints(result)


def test_default_registry_contains_core_distributions() -> None:
    names = list(list_distributions())
    assert "weibull" in names
    assert "gamma" in names
    dist = get_distribution("weibull")
    assert dist.parameters[0] == "a"


def test_generalized_beta_distributions_registered() -> None:
    x = np.array([20.0])
    sample_params = {
        "gb1": {"a": 1.2, "b": 100.0, "p": 2.0, "q": 3.0, "s": 1.0},
        "gb2": {"a": 1.1, "b": 90.0, "p": 2.0, "q": 2.5, "s": 1.0},
        "gg": {"a": 1.3, "beta": 30.0, "p": 2.0, "s": 1.0},
        "ib1": {"b": 40.0, "p": 2.0, "q": 3.0, "s": 1.0},
        "ug": {"b": 120.0, "d": 1.5, "q": 2.5, "s": 1.0},
        "b1": {"b": 120.0, "p": 2.0, "q": 3.0, "s": 1.0},
        "b2": {"b": 60.0, "p": 1.5, "q": 2.5, "s": 1.0},
        "sm": {"a": 1.1, "b": 75.0, "q": 2.0, "s": 1.0},
        "dagum": {"a": 1.2, "b": 80.0, "p": 2.0, "s": 1.0},
        "pareto": {"b": 15.0, "p": 2.5, "s": 1.0},
        "p": {"b": 60.0, "p": 2.0, "s": 1.0},
        "ln": {"mu": 3.0, "sigma2": 0.4, "s": 1.0},
        "ga": {"beta": 10.0, "p": 2.0, "s": 1.0},
        "w": {"a": 2.5, "beta": 25.0, "s": 1.0},
        "f": {"u": 5.0, "v": 10.0, "s": 1.0},
        "l": {"b": 80.0, "q": 2.0, "s": 1.0},
        "il": {"b": 35.0, "p": 2.0, "s": 1.0},
        "fisk": {"a": 1.3, "b": 50.0, "s": 1.0},
        "u": {"b": 80.0, "s": 1.0},
        "halfn": {"sigma2": 15.0, "s": 1.0},
        "chisq": {"p": 4.0, "s": 1.0},
        "exp": {"beta": 12.0, "s": 1.0},
        "r": {"beta": 18.0, "s": 1.0},
        "halft": {"df": 6.0, "s": 1.0},
        "ll": {"b": 45.0, "s": 1.0},
    }

    for dist in GENERALIZED_BETA_DISTRIBUTIONS:
        reg = get_distribution(dist.name)
        assert reg.parameters == dist.parameters
        params = sample_params.get(dist.name)
        assert params is not None, f"Missing sample parameters for {dist.name}"
        values = reg.pdf(x, params)
        assert np.all(np.isfinite(values))


def test_entry_point_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_registry()

    _dummy_distribution = base_registry.Distribution(
        name="entrypoint_demo",
        parameters=("scale",),
        pdf=_dummy_pdf,
        notes="Entry-point supplied distribution.",
    )

    monkeypatch.setattr(
        base_registry.metadata,
        "entry_points",
        lambda: _make_entry_points([_DummyEntryPoint("demo", lambda: _dummy_distribution)]),
    )

    base_registry.load_entry_points()
    assert "entrypoint_demo" in list_distributions()
    dist = get_distribution("entrypoint_demo")
    result = dist.pdf(np.array([1.0, 2.0]), {"scale": 2.5})
    assert np.allclose(result, 2.5)

    _reload_registry()


def test_yaml_registration(tmp_path: Path) -> None:
    clear_registry()

    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        """
metadata:
  title: demo
distributions:
  - name: yaml_demo
    parameters: ["scale"]
    pdf: tests.test_registry:_dummy_pdf
    notes: "YAML supplied distribution."
""",
        encoding="utf-8",
    )

    registered = base_registry.load_yaml_config(config_path)
    assert registered == ["yaml_demo"]
    dist = get_distribution("yaml_demo")
    assert np.allclose(dist.pdf(np.array([0.0]), {"scale": 3.0}), 3.0)

    _reload_registry()
