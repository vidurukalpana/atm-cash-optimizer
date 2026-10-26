"""Forecast accuracy: MAE, sMAPE, pinball loss and interval coverage, by horizon and segment.

All metrics compare a forecast of ``y_cum(o, h)`` with the true cumulative demand.

- **MAE** — mean absolute error, in LKR.
- **sMAPE** — mean of ``2·|f − a| / (|f| + |a|)``, in %. Unstable when demand is near zero
  (CRM ``net``), so read MAE for that target.
- **pinball** — quantile loss averaged over the quantiles a model provides (for a median-only
  forecast this is MAE / 2).
- **coverage_90** — share of actuals inside ``[q05, q95]``; empty for median-only forecasts.

Segments: ``all``; ``calendar`` (``holiday_window`` if any day in ``o+1 .. o+h`` is a public
holiday, long-weekend eve or festival day — UK holidays for ``nn5`` — else ``normal``);
``location_type``; ``machine_type``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atm_cash_optimizer.config import load_config
from atm_cash_optimizer.forecasting.backtest import SeriesPanel, cumulative_actuals
from atm_cash_optimizer.io import read_table
from atm_cash_optimizer.scenario.transform import uk_holiday_mask

KEYS = ["machine_id", "target", "origin_date", "h"]
SEGMENT_TYPES = ("all", "calendar", "location_type", "machine_type")
INTERVAL = (0.05, 0.95)


def quantile_column(q: float) -> str:
    """Column name of a quantile, e.g. ``0.05`` → ``q05``."""
    return f"q{round(q * 100):02d}"


# --------------------------------------------------------------------------------------------
# Metrics on arrays
# --------------------------------------------------------------------------------------------


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(forecast - actual)))


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Symmetric mean absolute percentage error in % (0 where both are 0)."""
    denominator = np.abs(forecast) + np.abs(actual)
    terms = np.divide(
        2 * np.abs(forecast - actual),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(100 * np.mean(terms))


def pinball(actual: np.ndarray, forecast: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss of forecasts of quantile ``q``."""
    diff = actual - forecast
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of actuals within ``[lower, upper]``."""
    return float(np.mean((actual >= lower) & (actual <= upper)))


# --------------------------------------------------------------------------------------------
# Backtest frames (same row order as baselines.to_frame: series, origin, h)
# --------------------------------------------------------------------------------------------


def _index_frame(panel: SeriesPanel, origin_idx: np.ndarray, horizon: int) -> pd.DataFrame:
    n_series, n_origins = len(panel.keys), len(origin_idx)
    return pd.DataFrame(
        {
            "machine_id": np.repeat(panel.keys["machine_id"].to_numpy(), n_origins * horizon),
            "target": np.repeat(panel.keys["target"].to_numpy(), n_origins * horizon),
            "origin_date": np.tile(np.repeat(panel.dates[origin_idx], horizon), n_series),
            "h": np.tile(np.arange(1, horizon + 1), n_series * n_origins),
        }
    )


def cumulative_actuals_frame(
    panel: SeriesPanel, origin_idx: np.ndarray, horizon: int
) -> pd.DataFrame:
    """``KEYS`` + ``actual`` (true ``y_cum``; NaN past the end of the data)."""
    out = _index_frame(panel, origin_idx, horizon)
    out["actual"] = cumulative_actuals(panel, origin_idx, horizon).ravel()
    return out


def special_days(track: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """True on holiday-like days: public holidays, long-weekend eves and festival windows."""
    if track == "nn5":
        return uk_holiday_mask(pd.Series(dates), load_config("scenario")).to_numpy()
    cal = read_table("lk_calendar").set_index("date").reindex(dates)
    event_days = [c for c in cal.columns if c.endswith("_day") and not c.startswith("is_")]
    festival = cal[event_days].notna().any(axis=1)
    return (cal["is_public_holiday"] | cal["is_long_weekend_eve"] | festival).to_numpy()


def segment_frame(
    track: str, panel: SeriesPanel, origin_idx: np.ndarray, horizon: int
) -> pd.DataFrame:
    """``KEYS`` + one column per segment type (see the module docstring)."""
    out = _index_frame(panel, origin_idx, horizon)
    cs = np.cumsum(special_days(track, panel.dates))
    end = np.minimum(origin_idx[:, None] + np.arange(1, horizon + 1), len(panel.dates) - 1)
    n_special = cs[end] - cs[origin_idx][:, None]
    out["all"] = "all"
    out["calendar"] = np.where(
        np.tile(n_special.ravel(), len(panel.keys)) > 0, "holiday_window", "normal"
    )
    for column in ("location_type", "machine_type"):
        out[column] = np.repeat(panel.keys[column].to_numpy(), len(origin_idx) * horizon)
    return out


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def evaluate(
    forecasts: pd.DataFrame,
    actual: pd.DataFrame,
    segments: pd.DataFrame,
    report_horizons: list[int],
) -> pd.DataFrame:
    """Metrics per model, target, segment and horizon.

    Returns:
        Long table: ``model, target, segment_type, segment, h, n, mae, smape, pinball,
        coverage_90``. Rows without a known actual are left out.
    """
    df = forecasts.merge(actual, on=KEYS).merge(segments, on=KEYS)
    df = df[df["h"].isin(report_horizons) & df["actual"].notna()]
    quantile_cols = [c for c in df.columns if c.startswith("q") and c[1:].isdigit()]
    lo, hi = (quantile_column(q) for q in INTERVAL)

    rows = []
    group_keys = ["model", "target", "h"]
    for segment_type in SEGMENT_TYPES:
        if df[segment_type].isna().all():  # e.g. no location types on nn5
            continue
        for (model, target, h, segment), g in df.groupby([*group_keys, segment_type]):
            a = g["actual"].to_numpy()
            available = [c for c in quantile_cols if g[c].notna().all()]
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "segment_type": segment_type,
                    "segment": segment,
                    "h": h,
                    "n": len(g),
                    "mae": mae(a, g["q50"].to_numpy()),
                    "smape": smape(a, g["q50"].to_numpy()),
                    "pinball": np.mean(
                        [pinball(a, g[c].to_numpy(), int(c[1:]) / 100) for c in available]
                    ),
                    "coverage_90": coverage(a, g[lo].to_numpy(), g[hi].to_numpy())
                    if {lo, hi} <= set(available)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def headline(metrics: pd.DataFrame) -> dict[str, float]:
    """MAE and sMAPE over all series per target and report horizon, e.g. ``withdrawals_mae_h7``."""
    overall = metrics[metrics["segment_type"] == "all"]
    out = {}
    for row in overall.itertuples():
        out[f"{row.target}_mae_h{row.h}"] = row.mae
        out[f"{row.target}_smape_h{row.h}"] = row.smape
    return out
