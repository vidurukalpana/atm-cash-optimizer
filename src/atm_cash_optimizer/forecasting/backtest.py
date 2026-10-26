"""Rolling-origin backtest: origins, leak-proof history and true cumulative demand.

The test period is the last ``test_days`` days; each day before a test day is an origin ``o``,
forecasting ``o+1 .. o+H`` from data up to ``o`` only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from atm_cash_optimizer.io import read_table, write_table

TRACKS = ("nn5", "lk")


@dataclass(frozen=True)
class SeriesPanel:
    """All target series of a track as one matrix (series × days)."""

    keys: pd.DataFrame  # machine_id, target, location_type, machine_type
    dates: pd.DatetimeIndex
    values: np.ndarray  # (len(keys), len(dates))


def load_series(track: str) -> SeriesPanel:
    """Daily ``withdrawals`` series; on ``lk`` also ``net`` = withdrawals − deposits (CRMs)."""
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
    """Positions of the day before each of the last ``test_days`` days."""
    if test_days >= n_days:
        raise ValueError(f"test_days ({test_days}) must be smaller than the series ({n_days})")
    return np.arange(n_days - test_days - 1, n_days - 1)


def history_slice(origin_index: int, days: int | None = None) -> slice:
    """Slice of the last ``days`` days up to and including ``origin_index``."""
    start = 0 if days is None else max(0, origin_index - days + 1)
    return slice(start, origin_index + 1)


def save_forecasts(track: str, new: pd.DataFrame) -> None:
    """Write ``forecasts_<track>``, replacing only the models in ``new``."""
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
    """True ``y_cum`` as ``(series, origins, horizon)``; NaN past the data end."""
    cs = np.cumsum(panel.values, axis=1)
    end = origin_idx[:, None] + np.arange(1, horizon + 1)[None, :]
    valid = end < len(panel.dates)
    out = cs[:, np.minimum(end, len(panel.dates) - 1)] - cs[:, origin_idx][:, :, None]
    return np.where(valid[None, :, :], out, np.nan)
