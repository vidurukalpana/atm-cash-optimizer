"""Demand profiles by weekday, month, month-end and around holidays.

Inputs are long tables with ``machine_id``, ``date`` and a value column (both tracks).
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

VALUE_COL = "withdrawals_lkr"


def weekday_profile(
    df: pd.DataFrame, value_col: str = VALUE_COL, normalize: bool = False
) -> pd.DataFrame:
    """Mean per machine and weekday (0 = Monday); ``normalize`` divides by the machine mean."""
    profile = df.pivot_table(
        index="machine_id", columns=df["date"].dt.day_of_week, values=value_col, aggfunc="mean"
    )
    profile.columns.name = "weekday"
    return profile.div(profile.mean(axis=1), axis=0) if normalize else profile


def monthly_profile(
    df: pd.DataFrame, value_col: str = VALUE_COL, normalize: bool = False
) -> pd.DataFrame:
    """Mean per machine and calendar month; ``normalize`` divides by the machine mean."""
    profile = df.pivot_table(
        index="machine_id", columns=df["date"].dt.month, values=value_col, aggfunc="mean"
    )
    profile.columns.name = "month"
    return profile.div(profile.mean(axis=1), axis=0) if normalize else profile


def coefficient_of_variation(df: pd.DataFrame, value_col: str = VALUE_COL) -> pd.Series:
    """Std / mean of the daily values per machine (higher = more volatile)."""
    grouped = df.groupby("machine_id")[value_col]
    return (grouped.std() / grouped.mean()).rename("cv")


def month_end_effect(
    df: pd.DataFrame, value_col: str = VALUE_COL, window_days: int = 3
) -> pd.Series:
    """Mean demand in the first and last ``window_days`` days of a month / other days."""
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
    """Demand ``offset`` days from a holiday relative to a normal day of the same weekday.

    Returns ``offset``, ``ratio`` and ``n_days`` (holiday dates found in the data).
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
