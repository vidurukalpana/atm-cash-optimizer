"""Typed, validated access to the YAML files in ``configs/``.

Every tunable value of the project lives in ``configs/*.yaml`` and is read through
:func:`load_config`, which returns an immutable pydantic model. Invalid values (negative fees,
rates outside [0, 1], unsorted quantiles, unknown keys, ...) raise
:class:`pydantic.ValidationError` when the file is loaded.

The config directory defaults to ``<repo>/configs`` and can be overridden with the
``ATM_CONFIG_DIR`` environment variable or the ``config_dir`` argument.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Literal, overload

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    ValidationInfo,
    field_validator,
    model_validator,
)

logger = logging.getLogger(__name__)

CONFIG_DIR_ENV = "ATM_CONFIG_DIR"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

UnitInterval = Annotated[float, Field(ge=0, le=1)]
"""A value in [0, 1], e.g. an interest rate or a fraction of capacity."""

OpenUnitInterval = Annotated[float, Field(gt=0, lt=1)]
"""A value in (0, 1), e.g. a quantile level."""


class _Config(BaseModel):
    """Base for all config models: immutable, and unknown keys are rejected (catches typos)."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_sorted_unique(values: list, field_name: str) -> list:
    if values != sorted(set(values)):
        raise ValueError(f"{field_name} must be sorted ascending without duplicates")
    return values


# --------------------------------------------------------------------------------------------
# costs.yaml
# --------------------------------------------------------------------------------------------


class CostsConfig(_Config):
    """Prices used by the optimizer objective and the backtest ``total_cost``."""

    currency: str = Field(min_length=1)
    visit_fee: PositiveFloat
    emergency_visit_fee: PositiveFloat
    idle_cash_annual_rate: UnitInterval
    stockout_penalty_per_day: PositiveFloat


# --------------------------------------------------------------------------------------------
# machines.yaml
# --------------------------------------------------------------------------------------------


class TrackConfig(_Config):
    """Denominations used by one track (``nn5`` or ``lk``)."""

    denominations: list[str] = Field(min_length=1)

    @field_validator("denominations")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("denominations must be unique")
        return v


class AtmConfig(_Config):
    """ATM cassette layout and capacity."""

    notes_per_cassette: PositiveInt
    cassettes: dict[str, PositiveInt] = Field(min_length=1)
    total_capacity_lkr: PositiveFloat

    @field_validator("cassettes")
    @classmethod
    def _numeric_denominations(cls, v: dict[str, int]) -> dict[str, int]:
        for denom in v:
            if not denom.isdigit() or int(denom) <= 0:
                raise ValueError(f"cassette denomination {denom!r} must be a positive integer")
        return v


class CrmConfig(_Config):
    """Cash Recycler Machine vault capacity (CRMs are planned on total cash)."""

    vault_capacity_lkr: PositiveFloat


class MachinesConfig(_Config):
    """Machine capacities, denominations per track and planning constants."""

    nn5_unit_lkr: PositiveFloat
    tracks: dict[str, TrackConfig] = Field(min_length=1)
    atm: AtmConfig
    crm: CrmConfig
    bundle_notes: PositiveInt
    order_lead_days: NonNegativeInt
    planning_horizon_days: PositiveInt
    initial_fill_fraction: UnitInterval

    @model_validator(mode="after")
    def _cassette_for_every_denomination(self) -> MachinesConfig:
        for track, cfg in self.tracks.items():
            if cfg.denominations == ["total"]:
                continue
            missing = set(cfg.denominations) - set(self.atm.cassettes)
            if missing:
                raise ValueError(f"track {track!r}: no ATM cassette for {sorted(missing)}")
        return self


# --------------------------------------------------------------------------------------------
# model.yaml
# --------------------------------------------------------------------------------------------


class LightGBMParams(_Config):
    """Hyperparameters passed to LightGBM."""

    n_estimators: PositiveInt
    learning_rate: Annotated[float, Field(gt=0, le=1)]
    num_leaves: Annotated[int, Field(ge=2)]
    min_child_samples: PositiveInt
    subsample: Annotated[float, Field(gt=0, le=1)]
    colsample_bytree: Annotated[float, Field(gt=0, le=1)]


class ModelConfig(_Config):
    """Forecasting settings: quantile levels, features and retraining schedule."""

    quantiles: list[OpenUnitInterval] = Field(min_length=1)
    horizon_max: PositiveInt
    lags: list[PositiveInt] = Field(min_length=1)
    rolling_windows: list[PositiveInt] = Field(min_length=1)
    weekday_mean_weeks: PositiveInt
    retrain_every_days: PositiveInt
    min_train_days: PositiveInt
    lightgbm: LightGBMParams
    seed: NonNegativeInt

    @field_validator("quantiles")
    @classmethod
    def _quantiles_valid(cls, v: list[float]) -> list[float]:
        _check_sorted_unique(v, "quantiles")
        # q50 is required: the optimizer uses the median for the idle-cash cost.
        if 0.5 not in v:
            raise ValueError("quantiles must include 0.5 (the median)")
        return v

    @field_validator("lags", "rolling_windows")
    @classmethod
    def _sorted(cls, v: list[int], info: ValidationInfo) -> list[int]:
        return _check_sorted_unique(v, info.field_name)


# --------------------------------------------------------------------------------------------
# backtest.yaml
# --------------------------------------------------------------------------------------------


class SweepsConfig(_Config):
    """Parameter values tried for each policy in the backtest sweep."""

    threshold: list[OpenUnitInterval] = Field(min_length=1)
    buffer: list[Annotated[float, Field(ge=0)]] = Field(min_length=1)
    alpha: list[OpenUnitInterval] = Field(min_length=1)

    @field_validator("threshold", "buffer", "alpha")
    @classmethod
    def _sorted(cls, v: list[float], info: ValidationInfo) -> list[float]:
        return _check_sorted_unique(v, info.field_name)


class SensitivityConfig(_Config):
    """Multipliers used to re-cost backtest results."""

    idle_cash_annual_rate_multipliers: list[PositiveFloat] = Field(min_length=1)
    visit_fee_multipliers: list[PositiveFloat] = Field(min_length=1)


class BacktestConfig(_Config):
    """Day-by-day replay settings."""

    test_days: PositiveInt
    warmup_days: NonNegativeInt
    sweeps: SweepsConfig
    fixed_schedule_weekday: Annotated[int, Field(ge=0, le=6)]
    sensitivity: SensitivityConfig
    seed: NonNegativeInt

    @model_validator(mode="after")
    def _warmup_shorter_than_test(self) -> BacktestConfig:
        if self.warmup_days >= self.test_days:
            raise ValueError("warmup_days must be smaller than test_days")
        return self


# --------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------

ConfigName = Literal["costs", "machines", "model", "backtest"]

CONFIG_MODELS: dict[str, type[_Config]] = {
    "costs": CostsConfig,
    "machines": MachinesConfig,
    "model": ModelConfig,
    "backtest": BacktestConfig,
}


def get_config_dir(config_dir: str | Path | None = None) -> Path:
    """Return the config directory: argument, else ``$ATM_CONFIG_DIR``, else ``<repo>/configs``."""
    if config_dir is not None:
        return Path(config_dir)
    env_dir = os.environ.get(CONFIG_DIR_ENV)
    return Path(env_dir) if env_dir else DEFAULT_CONFIG_DIR


@overload
def load_config(name: Literal["costs"], config_dir: str | Path | None = None) -> CostsConfig: ...
@overload
def load_config(
    name: Literal["machines"], config_dir: str | Path | None = None
) -> MachinesConfig: ...
@overload
def load_config(name: Literal["model"], config_dir: str | Path | None = None) -> ModelConfig: ...
@overload
def load_config(
    name: Literal["backtest"], config_dir: str | Path | None = None
) -> BacktestConfig: ...


def load_config(name: ConfigName, config_dir: str | Path | None = None) -> _Config:
    """Load and validate ``configs/<name>.yaml``.

    Args:
        name: One of ``"costs"``, ``"machines"``, ``"model"``, ``"backtest"``.
        config_dir: Directory holding the YAML files (see :func:`get_config_dir`).

    Returns:
        An immutable pydantic model for that file.

    Raises:
        ValueError: If ``name`` is not a known config.
        FileNotFoundError: If the YAML file does not exist.
        pydantic.ValidationError: If a value is missing, has the wrong type or breaks a rule.
    """
    if name not in CONFIG_MODELS:
        raise ValueError(f"unknown config {name!r}; expected one of {sorted(CONFIG_MODELS)}")

    path = get_config_dir(config_dir) / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    logger.debug("Loading config %s from %s", name, path)
    return CONFIG_MODELS[name].model_validate(raw)
