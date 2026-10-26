"""Sri Lankan calendar (``lk_calendar``): holidays, Poya, salary week, long weekends, festivals.

Holidays come from the ``holidays`` library; years it lacks need ``configs/lk_holidays.csv``.
``<event>_day`` columns hold the day relative to a festival's holiday (e.g. -2 = two days
before Avurudu), only inside that festival's windows.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import holidays
import pandas as pd

from atm_cash_optimizer.config import ScenarioConfig, get_config_dir

logger = logging.getLogger(__name__)

COUNTRY = "LK"
FALLBACK_CSV = "lk_holidays.csv"
POYA_SUFFIX = "Full Moon Poya Day"


def is_poya_name(name: str) -> bool:
    """True for a Full Moon Poya Day (not the extra "Day Following ..." holiday)."""
    return name.endswith(POYA_SUFFIX) and not name.startswith("Day Following")


def lk_holidays(years: range, fallback_csv: Path | None = None) -> dict[date, list[str]]:
    """Holiday names per date; raises if a year is missing from both library and CSV."""
    result: dict[date, list[str]] = defaultdict(list)
    missing: list[int] = []
    for year in years:
        library = holidays.country_holidays(COUNTRY, years=year)
        if not any(is_poya_name(n) for d in library for n in library.get_list(d)):
            missing.append(year)
            continue
        for day in library:
            result[day].extend(library.get_list(day))

    if missing:
        csv = fallback_csv or get_config_dir() / FALLBACK_CSV
        if not csv.is_file():
            raise ValueError(f"no Sri Lankan holidays for {missing}: add them to {csv}")
        extra = pd.read_csv(csv, parse_dates=["date"])
        extra = extra[extra["date"].dt.year.isin(missing)]
        if set(extra["date"].dt.year) != set(missing):
            raise ValueError(f"{csv} does not cover all of {missing}")
        for row in extra.itertuples():
            result[row.date.date()].append(row.name)
        logger.info("Used %s for years %s", csv, missing)
    return dict(result)


def _long_weekend_eve(non_working: pd.Series, min_days: int) -> pd.Series:
    # Length of the run of non-working days starting on each date.
    run = non_working.astype(int)
    for i in range(len(run) - 2, -1, -1):
        if run.iat[i]:
            run.iat[i] += run.iat[i + 1]
    starts_long_break = run.shift(-1, fill_value=0) >= min_days
    return ~non_working & starts_long_break


def build_lk_calendar(
    start: date, end: date, cfg: ScenarioConfig, fallback_csv: Path | None = None
) -> pd.DataFrame:
    """One row per day from ``start`` to ``end`` (inclusive)."""
    # Look past the end so a long weekend right after it is still detected.
    padded_end = end + timedelta(days=cfg.long_weekend_min_days)
    names = lk_holidays(range(start.year, padded_end.year + 1), fallback_csv)
    dates = pd.date_range(start, padded_end, freq="D", unit="ns")

    cal = pd.DataFrame({"date": dates})
    day_names = [names.get(d.date(), []) for d in dates]
    cal["holiday_name"] = ["; ".join(n) for n in day_names]
    cal["is_public_holiday"] = [bool(n) for n in day_names]
    cal["is_poya"] = [any(is_poya_name(x) for x in n) for n in day_names]
    cal["is_salary_week"] = cal["date"].dt.day >= cfg.salary_week_from_day

    non_working = (cal["date"].dt.day_of_week >= 5) | cal["is_public_holiday"]
    cal["is_long_weekend_eve"] = _long_weekend_eve(non_working, cfg.long_weekend_min_days)
    cal["is_no_service_day"] = (cal["is_poya"] & cfg.no_service.poya) | (
        cal["is_public_holiday"] & cfg.no_service.public_holidays
    )

    for event, event_cfg in cfg.events.items():
        anchors = [pd.Timestamp(d) for d, n in names.items() if event_cfg.anchor in n]
        if not anchors:
            logger.warning("event %r: no %r in the date range", event, event_cfg.anchor)
        offset = pd.Series(pd.NA, index=cal.index, dtype="Int64")
        for anchor in anchors:
            for window in event_cfg.windows:
                days = (cal["date"] - anchor).dt.days
                offset = offset.mask(days.between(window.start, window.end), days)
        cal[f"{event}_day"] = offset

    for event in ("avurudu", "christmas"):
        column = f"{event}_day"
        cal[f"is_{event}_window"] = cal[column].notna() if column in cal else False

    return cal[cal["date"] <= pd.Timestamp(end)].reset_index(drop=True)
