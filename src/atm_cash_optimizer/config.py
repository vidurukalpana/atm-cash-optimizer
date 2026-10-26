"""Validated settings from ``configs/*.yaml``: ``load_config(name)`` returns a frozen model.

Invalid values (negative fees, unsorted quantiles, unknown keys, ...) fail when the file is
loaded. The config directory can be moved with the ``ATM_CONFIG_DIR`` environment variable.
"""

from __future__ import annotations

import logging
import math
import os
import re
from datetime import date
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
OpenUnitInterval = Annotated[float, Field(gt=0, lt=1)]
NonNegativeFloat = Annotated[float, Field(ge=0)]


class _Config(BaseModel):
    """Frozen; unknown keys are rejected so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_sorted_unique(values: list, field_name: str) -> list:
    if values != sorted(set(values)):
        raise ValueError(f"{field_name} must be sorted ascending without duplicates")
    return values


# --- costs.yaml ---


class CostsConfig(_Config):
    """Prices used by the optimizer and in ``total_cost``."""

    currency: str = Field(min_length=1)
    visit_fee: PositiveFloat
    emergency_visit_fee: PositiveFloat
    idle_cash_annual_rate: UnitInterval
    stockout_penalty_per_day: PositiveFloat


# --- machines.yaml ---


class TrackConfig(_Config):
    """Denominations of one track (``nn5`` or ``lk``)."""

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
    """CRMs are planned on total cash, so only the vault size matters."""

    vault_capacity_lkr: PositiveFloat


class MachinesConfig(_Config):
    """Capacities, denominations per track and planning constants."""

    nn5_unit_lkr: PositiveFloat
    tracks: dict[str, TrackConfig] = Field(min_length=1)
    atm: AtmConfig
    crm: CrmConfig
    bundle_notes: PositiveInt
    order_lead_days: NonNegativeInt
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


# --- model.yaml ---


class LightGBMParams(_Config):
    """Hyperparameters passed to LightGBM."""

    n_estimators: PositiveInt
    learning_rate: Annotated[float, Field(gt=0, le=1)]
    num_leaves: Annotated[int, Field(ge=2)]
    min_child_samples: PositiveInt
    subsample: Annotated[float, Field(gt=0, le=1)]
    colsample_bytree: Annotated[float, Field(gt=0, le=1)]


class ModelConfig(_Config):
    """Forecasting: quantiles, features, retraining and compute limits."""

    quantiles: list[OpenUnitInterval] = Field(min_length=1)
    horizon_max: PositiveInt
    lags: list[PositiveInt] = Field(min_length=1)
    rolling_windows: list[PositiveInt] = Field(min_length=1)
    weekday_mean_weeks: PositiveInt
    retrain_every_days: PositiveInt
    lightgbm: LightGBMParams
    n_jobs: PositiveInt
    ets_train_days: PositiveInt
    quantile_retrain_every_days: PositiveInt
    quantile_train_days: PositiveInt
    train_origin_step: PositiveInt
    calibration_tolerance: Annotated[float, Field(ge=0, lt=1)]
    seed: NonNegativeInt

    @field_validator("quantiles")
    @classmethod
    def _quantiles_valid(cls, v: list[float]) -> list[float]:
        _check_sorted_unique(v, "quantiles")
        # The optimizer needs the median for the idle-cash cost.
        if 0.5 not in v:
            raise ValueError("quantiles must include 0.5 (the median)")
        return v

    @field_validator("lags", "rolling_windows")
    @classmethod
    def _sorted(cls, v: list[int], info: ValidationInfo) -> list[int]:
        return _check_sorted_unique(v, info.field_name)

    @field_validator("train_origin_step")
    @classmethod
    def _all_weekdays(cls, v: int) -> int:
        # A multiple of 7 would sample origins on one weekday only.
        if v % 7 == 0:
            raise ValueError("train_origin_step must not be a multiple of 7")
        return v


# --- backtest.yaml ---


class SweepsConfig(_Config):
    """Policy settings tried in the simulation."""

    threshold: list[OpenUnitInterval] = Field(min_length=1)
    alpha: list[OpenUnitInterval] = Field(min_length=1)

    @field_validator("threshold", "alpha")
    @classmethod
    def _sorted(cls, v: list[float], info: ValidationInfo) -> list[float]:
        return _check_sorted_unique(v, info.field_name)


class BacktestConfig(_Config):
    """Test period, simulation sample and policy settings."""

    test_days: PositiveInt
    warmup_days: NonNegativeInt
    report_horizons: list[PositiveInt] = Field(min_length=1)
    simulation_machines: PositiveInt
    sweeps: SweepsConfig
    fixed_schedule_weekday: Annotated[int, Field(ge=0, le=6)]
    seed: NonNegativeInt

    @field_validator("report_horizons")
    @classmethod
    def _sorted_horizons(cls, v: list[int]) -> list[int]:
        return _check_sorted_unique(v, "report_horizons")

    @model_validator(mode="after")
    def _warmup_shorter_than_test(self) -> BacktestConfig:
        if self.warmup_days >= self.test_days:
            raise ValueError("warmup_days must be smaller than test_days")
        return self


# --- scenario.yaml ---

_MONTH_DAY = re.compile(r"^(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$")


def _check_sums_to_one(shares: dict[str, float], what: str) -> None:
    if not math.isclose(sum(shares.values()), 1.0, abs_tol=1e-9):
        raise ValueError(f"{what} must sum to 1, got {sum(shares.values())}")


class LocationModifier(_Config):
    """Demand level, extra noise and optional peak months of a location type."""

    level: PositiveFloat
    noise_sigma: NonNegativeFloat
    peak_months: list[Annotated[int, Field(ge=1, le=12)]] = []
    peak_multiplier: PositiveFloat = 1.0


class CalendarMultipliers(_Config):
    """Multipliers for recurring calendar effects."""

    salary_week: PositiveFloat
    long_weekend_eve: PositiveFloat
    poya_day: PositiveFloat


class EventWindow(_Config):
    """Days ``[from, to]`` around a festival's holiday, with a demand multiplier."""

    start: int = Field(alias="from")
    end: int = Field(alias="to")
    multiplier: PositiveFloat

    @model_validator(mode="after")
    def _ordered(self) -> EventWindow:
        if self.start > self.end:
            raise ValueError("event window 'from' must not be after 'to'")
        return self


class EventConfig(_Config):
    """A festival season, anchored on a holiday name from the ``holidays`` library."""

    anchor: str = Field(min_length=1)
    windows: list[EventWindow] = Field(min_length=1)


class CrmDepositsConfig(_Config):
    """How simulated CRM deposits are drawn."""

    ratio_range: tuple[PositiveFloat, PositiveFloat]
    weekday_profile: list[PositiveFloat] = Field(min_length=7, max_length=7)
    holiday_multiplier: NonNegativeFloat
    noise_sigma: NonNegativeFloat

    @field_validator("ratio_range")
    @classmethod
    def _ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        if v[0] > v[1]:
            raise ValueError("ratio_range must be [low, high]")
        return v


class NoServiceConfig(_Config):
    """Days on which cash-in-transit vans do not run."""

    poya: bool
    public_holidays: bool


class UkNeutralizeConfig(_Config):
    """Original NN5 dates replaced to remove UK holiday effects."""

    days_around_holiday: NonNegativeInt
    extra_windows: list[tuple[str, str]]
    window_weeks: PositiveInt

    @field_validator("extra_windows")
    @classmethod
    def _month_days(cls, v: list[tuple[str, str]]) -> list[tuple[str, str]]:
        for window in v:
            for month_day in window:
                if not _MONTH_DAY.match(month_day):
                    raise ValueError(f"{month_day!r} is not a MM-DD date")
        return v


class BootstrapConfig(_Config):
    """Larger bootstrapped network (``--n-machines`` / ``--years``)."""

    level_sigma: NonNegativeFloat


class ScenarioConfig(_Config):
    """Simulated Sri Lankan network (``lk`` track) built from NN5."""

    start_date: date
    seed: NonNegativeInt
    crm_fraction: UnitInterval
    location_types: dict[str, dict[str, UnitInterval]] = Field(min_length=1)
    location_modifiers: dict[str, LocationModifier] = Field(min_length=1)
    calendar_multipliers: CalendarMultipliers
    salary_week_from_day: Annotated[int, Field(ge=1, le=31)]
    long_weekend_min_days: PositiveInt
    events: dict[str, EventConfig]
    denomination_shares: dict[str, dict[str, UnitInterval]]
    crm_deposits: CrmDepositsConfig
    no_service: NoServiceConfig
    uk_neutralize: UkNeutralizeConfig
    bootstrap: BootstrapConfig

    @model_validator(mode="after")
    def _consistent(self) -> ScenarioConfig:
        for label, probabilities in self.location_types.items():
            _check_sums_to_one(probabilities, f"location_types.{label}")
            unknown = set(probabilities) - set(self.location_modifiers)
            if unknown:
                raise ValueError(f"location_types.{label}: no location_modifiers for {unknown}")
        if "default" not in self.denomination_shares:
            raise ValueError("denomination_shares needs a 'default' entry")
        for location, shares in self.denomination_shares.items():
            _check_sums_to_one(shares, f"denomination_shares.{location}")
            if location != "default" and location not in self.location_modifiers:
                raise ValueError(f"denomination_shares: unknown location type {location!r}")
        return self


# --- Loading ---

ConfigName = Literal["costs", "machines", "model", "backtest", "scenario"]

CONFIG_MODELS: dict[str, type[_Config]] = {
    "costs": CostsConfig,
    "machines": MachinesConfig,
    "model": ModelConfig,
    "backtest": BacktestConfig,
    "scenario": ScenarioConfig,
}


def get_config_dir(config_dir: str | Path | None = None) -> Path:
    """Argument, else ``$ATM_CONFIG_DIR``, else ``<repo>/configs``."""
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
@overload
def load_config(
    name: Literal["scenario"], config_dir: str | Path | None = None
) -> ScenarioConfig: ...


def load_config(name: ConfigName, config_dir: str | Path | None = None) -> _Config:
    """Load and validate ``configs/<name>.yaml`` (raises ``ValidationError`` on bad values)."""
    if name not in CONFIG_MODELS:
        raise ValueError(f"unknown config {name!r}; expected one of {sorted(CONFIG_MODELS)}")

    path = get_config_dir(config_dir) / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")

    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    logger.debug("Loading config %s from %s", name, path)
    return CONFIG_MODELS[name].model_validate(raw)
