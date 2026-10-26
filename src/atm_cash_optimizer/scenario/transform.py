"""Turn NN5 withdrawals into Sri Lankan-looking demand, in the numbered steps below."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import holidays
import numpy as np
import pandas as pd

from atm_cash_optimizer.config import ScenarioConfig
from atm_cash_optimizer.data.impute import fill_weekday_median

logger = logging.getLogger(__name__)

UK_HOLIDAYS = ("GB", "ENG")  # same calendar as the EDA (docs/eda.md)
WEEK = 7
WEEKS_PER_YEAR = 52


def lognormal_noise(
    rng: np.random.Generator, sigma: float | np.ndarray, size: int | None = None
) -> np.ndarray:
    """Multiplicative noise with mean 1."""
    return rng.lognormal(-(np.square(sigma)) / 2, sigma, size)


# --- 1. Remove UK effects ---


def uk_holiday_mask(dates: pd.Series, cfg: ScenarioConfig) -> pd.Series:
    """True on original NN5 dates affected by UK holidays (``uk_neutralize`` in config)."""
    neutralize = cfg.uk_neutralize
    years = range(dates.dt.year.min() - 1, dates.dt.year.max() + 1)
    affected: set[pd.Timestamp] = set()

    for day in holidays.country_holidays(UK_HOLIDAYS[0], subdiv=UK_HOLIDAYS[1], years=years):
        for offset in range(-neutralize.days_around_holiday, neutralize.days_around_holiday + 1):
            affected.add(pd.Timestamp(day + timedelta(days=offset)))

    for year in years:
        for start, end in neutralize.extra_windows:
            first = pd.Timestamp(f"{year}-{start}")
            last = pd.Timestamp(f"{year}-{end}")
            if last < first:  # window wraps into the next year, e.g. 12-15 → 01-03
                last = pd.Timestamp(f"{year + 1}-{end}")
            affected.update(pd.date_range(first, last, freq="D"))

    return dates.isin(affected)


def neutralize_uk_holidays(df: pd.DataFrame, cfg: ScenarioConfig) -> pd.DataFrame:
    """Replace UK-holiday values by the nearby same-weekday median; adds ``is_neutralized``.

    Must run on the original NN5 dates, before :func:`shift_dates`.
    """
    out = df.sort_values(["machine_id", "date"], ignore_index=True)
    out["is_neutralized"] = uk_holiday_mask(out["date"], cfg)
    masked = out["withdrawals_lkr"].mask(out["is_neutralized"])

    filled = [
        fill_weekday_median(group, cfg.uk_neutralize.window_weeks)
        for _, group in masked.set_axis(out["date"]).groupby(out["machine_id"].to_numpy())
    ]
    out["withdrawals_lkr"] = pd.concat(filled).to_numpy()
    logger.info("Neutralized %d UK-holiday values", out["is_neutralized"].sum())
    return out


# --- 2. Optional larger network ---


def bootstrap_network(
    df: pd.DataFrame, n_machines: int, n_days: int, cfg: ScenarioConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """``n_machines`` × ``n_days`` series built from whole weeks of random base machines.

    Weeks are drawn from the same position in the year, so seasonality is kept; each new
    machine gets its own level factor.
    """
    base = df.sort_values(["machine_id", "date"], ignore_index=True)
    n_base = base["machine_id"].nunique()
    base_days = len(base) // n_base
    n_base_weeks = base_days // WEEK
    n_weeks = -(-n_days // WEEK)

    source = rng.integers(n_base, size=n_machines)
    if n_base_weeks >= WEEKS_PER_YEAR:
        position = np.arange(n_weeks) % WEEKS_PER_YEAR
        choices = (n_base_weeks - 1 - position) // WEEKS_PER_YEAR + 1
        weeks = position + WEEKS_PER_YEAR * rng.integers(0, choices, size=(n_machines, n_weeks))
    else:  # too little history to match the position in the year
        weeks = rng.integers(0, n_base_weeks, size=(n_machines, n_weeks))
    day_index = (weeks[:, :, None] * WEEK + np.arange(WEEK)).reshape(n_machines, -1)[:, :n_days]

    width = max(3, len(str(n_machines)))
    out = pd.DataFrame(
        {
            "machine_id": np.repeat([f"NN5_{j + 1:0{width}d}" for j in range(n_machines)], n_days),
            "date": np.tile(
                pd.date_range(base["date"].min(), periods=n_days, freq="D"), n_machines
            ),
        }
    )
    for column in base.columns.drop(["machine_id", "date"]):
        values = base[column].to_numpy().reshape(n_base, base_days)
        out[column] = values[source[:, None], day_index].ravel()

    level = lognormal_noise(rng, cfg.bootstrap.level_sigma, n_machines)
    out["withdrawals_lkr"] *= np.repeat(level, n_days)
    logger.info("Bootstrapped %d machines x %d days", n_machines, n_days)
    return out


# --- 3. Dates ---


def shift_dates(df: pd.DataFrame, start_date: date) -> pd.DataFrame:
    """Shift dates so the first day is ``start_date``; raises unless weekdays are kept."""
    shift = pd.Timestamp(start_date) - df["date"].min()
    if shift.days % WEEK:
        raise ValueError(
            f"start_date {start_date} must fall on a {df['date'].min().day_name()} "
            "so weekdays are preserved"
        )
    return df.assign(date=df["date"] + shift)


# --- 4. Location types ---


def assign_location_types(
    clusters: pd.DataFrame, cfg: ScenarioConfig, rng: np.random.Generator
) -> pd.Series:
    """Draw each machine's ``location_type`` from its cluster label's probabilities."""
    # Sorted, so the result depends only on the seed.
    clusters = clusters.sort_values("machine_id")
    types = []
    for label in clusters["cluster_label"]:
        probabilities = cfg.location_types[label]
        types.append(rng.choice(list(probabilities), p=list(probabilities.values())))
    return pd.Series(types, index=clusters["machine_id"].to_numpy(), name="location_type")


def apply_location_effects(
    df: pd.DataFrame, cfg: ScenarioConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Scale ``withdrawals_lkr`` by location level, peak months and extra daily noise."""
    out = df.sort_values(["machine_id", "date"], ignore_index=True)
    modifiers = cfg.location_modifiers

    level = out["location_type"].map({k: m.level for k, m in modifiers.items()})
    sigma = out["location_type"].map({k: m.noise_sigma for k, m in modifiers.items()})
    peak = pd.Series(1.0, index=out.index)
    for name, m in modifiers.items():
        in_peak = (out["location_type"] == name) & out["date"].dt.month.isin(m.peak_months)
        peak = peak.mask(in_peak, m.peak_multiplier)

    noise = lognormal_noise(rng, sigma.to_numpy())
    out["withdrawals_lkr"] = out["withdrawals_lkr"] * level * peak * noise
    return out


# --- 5. Calendar ---


def calendar_multiplier(calendar: pd.DataFrame, cfg: ScenarioConfig) -> pd.Series:
    """Demand multiplier per day: recurring effects multiply; of overlapping festivals only
    the strongest (furthest from 1) counts."""
    m = cfg.calendar_multipliers
    result = (
        np.where(calendar["is_poya"], m.poya_day, 1.0)
        * np.where(calendar["is_salary_week"], m.salary_week, 1.0)
        * np.where(calendar["is_long_weekend_eve"], m.long_weekend_eve, 1.0)
    )

    festival = pd.Series(1.0, index=calendar.index)
    for event, event_cfg in cfg.events.items():
        day = calendar[f"{event}_day"]
        for window in event_cfg.windows:
            active = day.between(window.start, window.end).fillna(False).astype(bool)
            stronger = abs(np.log(window.multiplier)) > np.abs(np.log(festival))
            festival = festival.mask(active & stronger, window.multiplier)

    return pd.Series(result * festival.to_numpy(), index=calendar.index, name="multiplier")


def apply_calendar(df: pd.DataFrame, calendar: pd.DataFrame, cfg: ScenarioConfig) -> pd.DataFrame:
    """Join the calendar flags and multiply ``withdrawals_lkr`` by each day's factor."""
    cal = calendar.drop(columns="holiday_name").assign(
        multiplier=calendar_multiplier(calendar, cfg)
    )
    out = df.merge(cal, on="date", how="left", validate="many_to_one")
    if out["multiplier"].isna().any():
        raise ValueError("calendar does not cover all dates")
    out["withdrawals_lkr"] *= out.pop("multiplier")
    return out
