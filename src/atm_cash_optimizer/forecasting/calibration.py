"""Check and correct the coverage of quantile forecasts.

A ``q95`` forecast is honest when about 95 % of actual values fall at or below it. Before each
retrain, the out-of-sample forecasts of the most recent window (whose actuals are known by then)
are checked per target, horizon and quantile. When coverage misses the level by more than
``calibration_tolerance``, an additive **conformal shift** is estimated: the ``q``-quantile of
``actual − forecast``, which would have given exactly the right coverage on that window. The
shift is then added to the new model's forecasts until the next retrain.

Shifts are per horizon because the error of a 14-day total is much larger than that of a 1-day
total.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atm_cash_optimizer.forecasting.evaluation import quantile_column, quantile_coverage

GROUP = ["target", "h"]


def conformal_shifts(
    calibration: pd.DataFrame, quantiles: list[float], tolerance: float
) -> pd.DataFrame:
    """Coverage and correction per target, horizon and quantile.

    Args:
        calibration: Out-of-sample rows with ``target``, ``h``, ``actual`` and quantile columns.
        quantiles: Quantile levels to check.
        tolerance: Allowed miss in coverage (e.g. 0.02 = 2 points) before a shift is applied.

    Returns:
        ``target, h, quantile, n, coverage, shift, applied`` (``shift`` is 0 when not applied).
    """
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
    """Add the shifts to the matching ``target``/``h`` rows of ``predictions`` (copy)."""
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
    """Coverage of each quantile per target and segment (rows need ``actual``).

    Returns:
        ``target, segment_type, segment, n`` plus one ``cov_qXX`` column per quantile.
    """
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
