"""Scaffold tests: the package and every planned module import, with docs."""

import importlib

import pytest

import op_infer

MODULES = [
    "params",
    "priors",
    "observe",
    "forward",
    "likelihood",
    "estimators",
    "posterior",
    "calibrate",
    "compare",
    "identify",
    "design",
    "rng",
    "registry",
    "simulation",
]


def test_version() -> None:
    """The package exposes its version."""
    assert op_infer.__version__ == "0.0.1"


@pytest.mark.parametrize("name", MODULES)
def test_module_scaffold(name: str) -> None:
    """Planned modules retain scope docs; only implemented modules export an API."""
    mod = importlib.import_module(f"op_infer.{name}")
    assert mod.__doc__
    if name not in {"priors", "observe", "simulation"}:
        assert "Scope:" in mod.__doc__
    if name not in {
        "forward",
        "posterior",
        "rng",
        "calibrate",
        "compare",
        "design",
        "priors",
        "observe",
        "simulation",
    }:
        assert mod.__all__ == []
