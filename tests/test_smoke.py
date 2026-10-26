"""Smoke test: the package and its subpackages import."""

import importlib

import pytest

import atm_cash_optimizer

SUBPACKAGES = [
    "data",
    "analysis",
    "scenario",
    "features",
    "forecasting",
    "optimization",
    "simulation",
]


def test_package_has_version() -> None:
    assert atm_cash_optimizer.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"atm_cash_optimizer.{name}")
