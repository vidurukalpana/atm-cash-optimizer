"""Forecast accuracy by horizon and segment, plus the MLflow setup.

Metrics: MAE (LKR), sMAPE (%, unreliable near zero, e.g. CRM ``net``), pinball loss and
coverage. Segments: all, calendar (``holiday_window`` vs ``normal``), location and machine type.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atm_cash_optimizer.config import load_config
from atm_cash_optimizer.forecasting.backtest import SeriesPanel, cumulative_actuals
from atm_cash_optimizer.io import DEFAULT_DATA_DIR, read_table
from atm_cash_optimizer.scenario.transform import uk_holiday_mask

KEYS = ["machine_id", "target", "origin_date", "h"]
SEGMENT_TYPES = ("all", "calendar", "location_type", "machine_type")
INTERVAL = (0.05, 0.95)
MLFLOW_DIR = DEFAULT_DATA_DIR.parent / "mlruns"


def quantile_column(q: float) -> str:
    """``0.05`` → ``q05``."""
    return f"q{round(q * 100):02d}"


# --- Metrics on arrays ---


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(forecast - actual)))


def smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Symmetric MAPE in % (0 where both values are 0)."""
    denominator = np.abs(forecast) + np.abs(actual)
    terms = np.divide(
        2 * np.abs(forecast - actual),
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(100 * np.mean(terms))


def pinball(actual: np.ndarray, forecast: np.ndarray, q: float) -> float:
    """Pinball (quantile) loss for quantile ``q``."""
    diff = actual - forecast
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def coverage(actual: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Share of actuals within ``[lower, upper]``."""
    return float(np.mean((actual >= lower) & (actual <= upper)))


def quantile_coverage(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Share of actuals at or below the forecast (should match its level)."""
    return float(np.mean(actual <= forecast))


def pinball_by_quantile(
    df: pd.DataFrame, quantiles: list[float], horizons: list[int]
) -> pd.DataFrame:
    """``target, h, quantile, pinball`` for each quantile column."""
    df = df[df["h"].isin(horizons) & df["actual"].notna()]
    rows = []
    for (target, h), group in df.groupby(["target", "h"]):
        actual = group["actual"].to_numpy()
        for q in quantiles:
            loss = pinball(actual, group[quantile_column(q)].to_numpy(), q)
            rows.append({"target": target, "h": h, "quantile": q, "pinball": loss})
    return pd.DataFrame(rows)


def use_mlflow_experiment(name: str) -> None:
    """Use ``mlruns/`` (SQLite + artifacts) and select experiment ``name``."""
    import mlflow

    MLFLOW_DIR.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DIR / 'mlflow.db'}")
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=str(MLFLOW_DIR / "artifacts"))
    mlflow.set_experiment(name)


# --- Backtest frames (row order: series, origin, h — as in baselines.to_frame) ---


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
    """Backtest keys + ``actual`` (true ``y_cum``)."""
    out = _index_frame(panel, origin_idx, horizon)
    out["actual"] = cumulative_actuals(panel, origin_idx, horizon).ravel()
    return out


def special_days(track: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """Holiday-like days: public holidays, long-weekend eves, festival windows."""
    if track == "nn5":
        return uk_holiday_mask(pd.Series(dates), load_config("scenario")).to_numpy()
    cal = read_table("lk_calendar").set_index("date").reindex(dates)
    event_days = [c for c in cal.columns if c.endswith("_day") and not c.startswith("is_")]
    festival = cal[event_days].notna().any(axis=1)
    return (cal["is_public_holiday"] | cal["is_long_weekend_eve"] | festival).to_numpy()


def segment_frame(
    track: str, panel: SeriesPanel, origin_idx: np.ndarray, horizon: int
) -> pd.DataFrame:
    """Backtest keys + one column per segment type."""
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


# --- Report ---


def evaluate(
    forecasts: pd.DataFrame,
    actual: pd.DataFrame,
    segments: pd.DataFrame,
    report_horizons: list[int],
) -> pd.DataFrame:
    """Metrics per model, target, segment and horizon (rows without an actual are skipped)."""
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
    """Overall MAE and sMAPE per target and horizon, e.g. ``withdrawals_mae_h7``."""
    overall = metrics[metrics["segment_type"] == "all"]
    out = {}
    for row in overall.itertuples():
        out[f"{row.target}_mae_h{row.h}"] = row.mae
        out[f"{row.target}_smape_h{row.h}"] = row.smape
    return out
