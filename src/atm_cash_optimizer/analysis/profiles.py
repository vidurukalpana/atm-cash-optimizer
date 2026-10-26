"""Demand profiles: how withdrawals vary by weekday, month, month-end and around holidays.

All functions take a long table with ``machine_id``, ``date`` and a value column (default
``withdrawals_lkr``), so they work for both the ``nn5`` and the ``lk`` track.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

VALUE_COL = "withdrawals_lkr"


def weekday_profile(
    df: pd.DataFrame, value_col: str = VALUE_COL, normalize: bool = False
) -> pd.DataFrame:
    """Mean value per machine (rows) and weekday (columns ``0`` = Monday … ``6`` = Sunday).

    With ``normalize=True`` each row is divided by its mean, so ``1.3`` means "30 % above
    this machine's average day" and machines of different size become comparable.
    """
    profile = df.pivot_table(
        index="machine_id", columns=df["date"].dt.day_of_week, values=value_col, aggfunc="mean"
    )
    profile.columns.name = "weekday"
    return profile.div(profile.mean(axis=1), axis=0) if normalize else profile


def monthly_profile(
    df: pd.DataFrame, value_col: str = VALUE_COL, normalize: bool = False
) -> pd.DataFrame:
    """Mean daily value per machine (rows) and calendar month (columns ``1`` … ``12``).

    ``normalize=True`` divides each row by its mean, as in :func:`weekday_profile`.
    """
    profile = df.pivot_table(
        index="machine_id", columns=df["date"].dt.month, values=value_col, aggfunc="mean"
    )
    profile.columns.name = "month"
    return profile.div(profile.mean(axis=1), axis=0) if normalize else profile


def coefficient_of_variation(df: pd.DataFrame, value_col: str = VALUE_COL) -> pd.Series:
    """Standard deviation / mean of the daily values per machine (higher = more volatile)."""
    grouped = df.groupby("machine_id")[value_col]
    return (grouped.std() / grouped.mean()).rename("cv")


def month_end_effect(
    df: pd.DataFrame, value_col: str = VALUE_COL, window_days: int = 3
) -> pd.Series:
    """Ratio of mean demand near the month boundary to mean demand on all other days.

    "Near the month boundary" means the last ``window_days`` days of a month and the first
    ``window_days`` days of the next (salary days). ``1.2`` = 20 % higher around month-end.
    """
    day = df["date"].dt.day
    near_end = (day <= window_days) | (day > df["date"].dt.days_in_month - window_days)
    means = df.groupby(["machine_id", near_end.rename("near_end")])[value_col].mean().unstack()
    return (means[True] / means[False]).rename("month_end_ratio")


def holiday_effect(
    df: pd.DataFrame,
    holiday_dates: Iterable,
    value_col: str = VALUE_COL,
    offsets: Iterable[int] = (-1, 0, 1),
) -> pd.DataFrame:
    """Demand on days around holidays relative to a normal day of the same weekday.

    For each offset (``-1`` = day before the holiday, ``0`` = the holiday, ``1`` = day after),
    every value is divided by the machine's mean for that weekday on non-holiday days, then
    averaged over all machines and holidays.

    Returns:
        DataFrame with columns ``offset``, ``ratio`` and ``n_days`` (holiday dates in the data).
    """
    holidays = pd.DatetimeIndex(pd.to_datetime(list(holiday_dates))).normalize()
    weekday = df["date"].dt.day_of_week
    normal = df[~df["date"].isin(holidays)]
    baseline = normal.groupby(["machine_id", normal["date"].dt.day_of_week])[value_col].mean()
    key = pd.MultiIndex.from_arrays([df["machine_id"], weekday])
    relative = df[value_col] / baseline.reindex(key).to_numpy()

    rows = []
    for offset in offsets:
        days = holidays + pd.Timedelta(days=offset)
        mask = df["date"].isin(days)
        rows.append(
            {
                "offset": offset,
                "ratio": relative[mask].mean(),
                "n_days": df.loc[mask, "date"].nunique(),
            }
        )
    return pd.DataFrame(rows)
