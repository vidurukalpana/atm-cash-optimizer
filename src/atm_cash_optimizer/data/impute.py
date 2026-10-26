"""Fill missing daily withdrawals using values from the same weekday.

ATM withdrawals follow a strong weekly pattern, so a missing Tuesday is best estimated from
nearby Tuesdays:

1. median of the non-missing values on the same weekday within ±``window_weeks`` weeks;
2. if all of those are missing, the median of that weekday over the whole series.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

WINDOW_WEEKS = 4
"""Weeks searched on each side of a missing day (method fixed by the development plan)."""


def fill_weekday_median(s: pd.Series, window_weeks: int = WINDOW_WEEKS) -> pd.Series:
    """Fill NaNs in one machine's daily series (consecutive days, ``DatetimeIndex``).

    Uses the same-weekday median within ±``window_weeks``, then the series' weekday median.
    """
    # Rows are consecutive days, so shifting by 7 * k rows moves k weeks.
    neighbours = pd.concat(
        [s.shift(sign * 7 * k) for k in range(1, window_weeks + 1) for sign in (1, -1)],
        axis=1,
    )
    local = neighbours.median(axis=1, skipna=True)
    weekday = s.groupby(s.index.day_of_week).transform("median")
    return s.fillna(local).fillna(weekday)


def impute_weekday_median(df: pd.DataFrame, window_weeks: int = WINDOW_WEEKS) -> pd.DataFrame:
    """Fill NaNs in ``withdrawals_raw`` and add the boolean column ``is_imputed``.

    Args:
        df: Long table with ``machine_id``, ``date``, ``withdrawals_raw``; dates must be
            continuous within each machine (as produced by :func:`~.tsf.read_tsf`).
        window_weeks: Weeks searched on each side of a missing day.

    Returns:
        A copy of ``df`` sorted by machine and date, with no NaNs left in ``withdrawals_raw``
        unless a weekday has no observed value at all in that series.
    """
    out = df.sort_values(["machine_id", "date"], ignore_index=True)
    out["is_imputed"] = out["withdrawals_raw"].isna()

    filled = [
        fill_weekday_median(group.set_index("date")["withdrawals_raw"], window_weeks)
        for _, group in out.groupby("machine_id", sort=False)
    ]
    out["withdrawals_raw"] = pd.concat(filled).to_numpy()

    logger.info("Imputed %d of %d values", out["is_imputed"].sum(), len(out))
    return out
