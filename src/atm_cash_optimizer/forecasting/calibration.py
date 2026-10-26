"""Check and correct the coverage of quantile forecasts (conformal shifts).

Before each retrain, recent out-of-sample forecasts with known actuals are checked per target,
horizon and quantile. If coverage misses by more than the tolerance, the ``q``-quantile of
``actual − forecast`` is added to the next forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atm_cash_optimizer.forecasting.evaluation import quantile_column, quantile_coverage

GROUP = ["target", "h"]


def conformal_shifts(
    calibration: pd.DataFrame, quantiles: list[float], tolerance: float
) -> pd.DataFrame:
    """``target, h, quantile, n, coverage, shift, applied`` (shift 0 when not applied)."""
    rows = []
    for (target, h), group in calibration.groupby(GROUP):
        actual = group["actual"].to_numpy()
        for q in quantiles:
            forecast = group[quantile_column(q)].to_numpy()
            cov = quantile_coverage(actual, forecast)
            applied = abs(cov - q) > tolerance
            rows.append(
                {
                    "target": target,
                    "h": h,
                    "quantile": q,
                    "n": len(group),
                    "coverage": cov,
                    "shift": float(np.quantile(actual - forecast, q)) if applied else 0.0,
                    "applied": applied,
                }
            )
    return pd.DataFrame(rows)


def apply_shifts(predictions: pd.DataFrame, shifts: pd.DataFrame) -> pd.DataFrame:
    """Add the shifts to the matching ``target``/``h`` rows (returns a copy)."""
    out = predictions.copy()
    if shifts.empty:
        return out
    wide = shifts.pivot_table(index=GROUP, columns="quantile", values="shift")
    wide.columns = [quantile_column(q) for q in wide.columns]
    keys = pd.MultiIndex.from_frame(out[GROUP])
    for column in wide.columns:
        out[column] = out[column] + wide[column].reindex(keys).fillna(0.0).to_numpy()
    return out


def coverage_by_segment(
    df: pd.DataFrame, quantiles: list[float], segment_types: list[str]
) -> pd.DataFrame:
    """Coverage of each quantile per target and segment."""
    rows = []
    for segment_type in segment_types:
        if df[segment_type].isna().all():
            continue
        for (target, segment), group in df.groupby(["target", segment_type]):
            actual = group["actual"].to_numpy()
            row = {"target": target, "segment_type": segment_type, "segment": segment}
            row["n"] = len(group)
            for q in quantiles:
                column = quantile_column(q)
                row[f"cov_{column}"] = quantile_coverage(actual, group[column].to_numpy())
            rows.append(row)
    return pd.DataFrame(rows)
