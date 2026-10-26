"""Rolling-origin backtest: which days are forecast from, and which data a forecast may use.

The test period is the last ``test_days`` days of a track. Every day before a test day is an
**origin** ``o``: the forecast is made at the end of ``o`` and covers ``o+1 .. o+H``. Models are
(re)trained on **retrain dates**, every ``retrain_every_days`` origins, using only data with
``date <= retrain date`` (enforced by :func:`training_data` and :func:`history_slice`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atm_cash_optimizer.io import read_table, write_table

TRACKS = ("nn5", "lk")


@dataclass(frozen=True)
class SeriesPanel:
    """All target series of a track as one matrix (series × days, same dates for all)."""

    keys: pd.DataFrame
    """One row per series: ``machine_id``, ``target``, ``location_type``, ``machine_type``."""
    dates: pd.DatetimeIndex
    values: np.ndarray
    """Shape ``(len(keys), len(dates))``."""


def load_series(track: str) -> SeriesPanel:
    """Load the daily target series of a track.

    ``withdrawals`` for every machine; on ``lk`` also ``net`` = withdrawals − deposits for CRMs.
    """
    if track == "nn5":
        nn5 = read_table("nn5_daily")
        df = nn5.assign(
            target="withdrawals", y=nn5["withdrawals_lkr"], location_type=None, machine_type="ATM"
        )
    elif track == "lk":
        lk = read_table("lk_daily")
        withdrawals = lk.assign(target="withdrawals", y=lk["withdrawals_lkr"])
        crm = lk[lk["machine_type"] == "CRM"]
        net = crm.assign(target="net", y=crm["withdrawals_lkr"] - crm["deposits_lkr"])
        df = pd.concat([withdrawals, net], ignore_index=True)
    else:
        raise ValueError(f"unknown track {track!r}; expected one of {TRACKS}")

    wide = df.pivot_table(index=["machine_id", "target"], columns="date", values="y")
    if wide.isna().any().any():
        raise ValueError(f"{track}: series have gaps or different date ranges")
    static = df.groupby(["machine_id", "target"])[["location_type", "machine_type"]].first()
    keys = static.loc[wide.index].reset_index()
    return SeriesPanel(keys=keys, dates=pd.DatetimeIndex(wide.columns), values=wide.to_numpy())


def origin_indices(n_days: int, test_days: int) -> np.ndarray:
    """Positions of the origins: the day before each of the last ``test_days`` days."""
    if test_days >= n_days:
        raise ValueError(f"test_days ({test_days}) must be smaller than the series ({n_days})")
    return np.arange(n_days - test_days - 1, n_days - 1)


def origins(dates: pd.DatetimeIndex, test_days: int) -> pd.DatetimeIndex:
    """Origin dates of the backtest (see :func:`origin_indices`)."""
    return dates[origin_indices(len(dates), test_days)]


def retrain_dates(origin_dates: pd.DatetimeIndex, every_days: int) -> pd.DatetimeIndex:
    """Every ``every_days``-th origin, starting with the first one."""
    return origin_dates[::every_days]


def training_data(df: pd.DataFrame, cutoff: pd.Timestamp, days: int | None = None) -> pd.DataFrame:
    """Rows with ``date <= cutoff`` (and, if ``days`` is given, only the last ``days`` days).

    The only way models get training data, so nothing after ``cutoff`` can leak in.
    """
    keep = df["date"] <= cutoff
    if days is not None:
        keep &= df["date"] > cutoff - pd.Timedelta(days=days)
    out = df[keep]
    assert out["date"].max() <= cutoff, "training data after the cutoff"
    return out


def history_slice(origin_index: int, days: int | None = None) -> slice:
    """Array slice of the history up to and including ``origin_index`` (last ``days`` days)."""
    start = 0 if days is None else max(0, origin_index - days + 1)
    return slice(start, origin_index + 1)


def save_forecasts(track: str, new: pd.DataFrame) -> None:
    """Write ``forecasts_<track>``, replacing only the models in ``new`` (others are kept)."""
    name = f"forecasts_{track}"
    try:
        existing = read_table(name)
        new = pd.concat(
            [existing[~existing["model"].isin(new["model"].unique())], new], ignore_index=True
        )
    except FileNotFoundError:
        pass
    write_table(new, name)


def cumulative_actuals(panel: SeriesPanel, origin_idx: np.ndarray, horizon: int) -> np.ndarray:
    """True ``y_cum`` per series, origin and ``h = 1..horizon``; NaN past the end of the data.

    Shape ``(n_series, n_origins, horizon)``.
    """
    cs = np.cumsum(panel.values, axis=1)
    end = origin_idx[:, None] + np.arange(1, horizon + 1)[None, :]
    valid = end < len(panel.dates)
    out = cs[:, np.minimum(end, len(panel.dates) - 1)] - cs[:, origin_idx][:, :, None]
    return np.where(valid[None, :, :], out, np.nan)
