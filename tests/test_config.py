"""Tests for atm_cash_optimizer.config."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from atm_cash_optimizer.config import DEFAULT_CONFIG_DIR, load_config


@pytest.mark.parametrize("name", ["costs", "machines", "model", "backtest", "scenario"])
def test_config_loads(name: str) -> None:
    load_config(name)  # type: ignore[call-overload]


@pytest.mark.parametrize(
    ("name", "key", "value"),
    [
        pytest.param("costs", "visit_fee", -1, id="negative-fee"),
        pytest.param("costs", "idle_cash_annual_rate", 1.5, id="rate-above-1"),
        pytest.param("costs", "visit_fees", 100, id="unknown-key-typo"),
        pytest.param("machines", "order_lead_days", -1, id="negative-lead"),
        pytest.param("model", "quantiles", [0.5, 0.05, 0.95], id="unsorted-quantiles"),
        pytest.param("model", "quantiles", [0.05, 0.95], id="quantiles-without-median"),
        pytest.param("backtest", "warmup_days", 280, id="warmup-not-below-test"),
    ],
)
def test_invalid_value_raises(tmp_path: Path, name: str, key: str, value: Any) -> None:
    data = yaml.safe_load((DEFAULT_CONFIG_DIR / f"{name}.yaml").read_text())
    data[key] = value
    (tmp_path / f"{name}.yaml").write_text(yaml.safe_dump(data))

    with pytest.raises(ValidationError):
        load_config(name, config_dir=tmp_path)  # type: ignore[call-overload]
