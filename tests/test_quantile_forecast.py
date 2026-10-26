"""Tests for the LightGBM quantile forecast: monotone quantiles, coverage, training, calibration."""

import numpy as np
import pandas as pd

from atm_cash_optimizer.config import load_config
from atm_cash_optimizer.forecasting.calibration import apply_shifts, conformal_shifts
from atm_cash_optimizer.forecasting.evaluation import quantile_column, quantile_coverage
from atm_cash_optimizer.forecasting.quantile_model import (
    fit_quantile_models,
    predict_quantiles,
    sort_quantiles,
)
from atm_cash_optimizer.forecasting.train import training_rows

CFG = load_config("model")
TINY = {"n_estimators": 20, "min_child_samples": 5, "n_jobs": 1}  # fast LightGBM for tests


def _features(n_origins: int = 60) -> pd.DataFrame:
    """Two machines, one target, h = 1..3; y_cum grows with h and a noisy feature."""
    rng = np.random.default_rng(0)
    rows = []
    for machine, (cluster, location) in {"A": (0, "branch"), "B": (1, "bus_stand")}.items():
        for o in pd.date_range("2024-01-01", periods=n_origins, freq="D"):
            for h in (1, 2, 3):
                level = rng.uniform(80, 120)
                rows.append(
                    {
                        "machine_id": machine,
                        "origin_date": o,
                        "h": h,
                        "target": "withdrawals",
                        "cluster_id": cluster,
                        "location_type": location,
                        "machine_type": "ATM",
                        "y_roll_mean_7": level,
                        "y_cum": h * level * rng.lognormal(0, 0.2),
                    }
                )
    return pd.DataFrame(rows)


def test_sort_quantiles_makes_them_monotone() -> None:
    crossed = pd.DataFrame({"q05": [10.0, 5.0], "q50": [8.0, 6.0], "q95": [9.0, 7.0]})

    fixed = sort_quantiles(crossed)

    assert fixed.loc[0].tolist() == [8.0, 9.0, 10.0]
    assert (np.diff(fixed.to_numpy(), axis=1) >= 0).all()


def test_quantile_coverage_on_known_data() -> None:
    actual = np.arange(1.0, 11.0)  # 1 .. 10

    assert quantile_coverage(actual, np.full(10, 8.0)) == 0.8
    assert quantile_coverage(actual, actual) == 1.0  # "at or below" counts ties


def test_train_and_predict_on_tiny_data() -> None:
    features = _features()

    models = fit_quantile_models(features, CFG, TINY)
    predictions = sort_quantiles(predict_quantiles(models, features.head(30)))

    assert list(predictions.columns) == [quantile_column(q) for q in CFG.quantiles]
    assert len(predictions) == 30
    assert (np.diff(predictions.to_numpy(), axis=1) >= 0).all()
    # The median should track h: 3-day totals are larger than 1-day totals.
    by_h = predictions["q50"].groupby(features.head(30)["h"].to_numpy()).mean()
    assert by_h[3] > by_h[1]


def test_training_never_uses_labels_after_retrain_date() -> None:
    features = _features()
    retrain_date = pd.Timestamp("2024-02-15")

    rows = training_rows(features, retrain_date, CFG)
    label_end = rows["origin_date"] + pd.to_timedelta(rows["h"], unit="D")

    assert len(rows) > 0
    assert label_end.max() <= retrain_date
    assert rows["origin_date"].max() <= retrain_date - pd.Timedelta(days=CFG.horizon_max)
    # Every train_origin_step-th origin only.
    days = (rows["origin_date"] - pd.Timestamp(0)).dt.days
    assert (days % CFG.train_origin_step == 0).all()


def test_conformal_shift_restores_coverage() -> None:
    rng = np.random.default_rng(1)
    actual = rng.normal(100, 10, 2000)
    # q90 forecast that is too low: covers only about 50 % of the actuals.
    calibration = pd.DataFrame(
        {"target": "withdrawals", "h": 1, "actual": actual, "q90": np.full(2000, 100.0)}
    )

    shifts = conformal_shifts(calibration, [0.9], tolerance=0.02)
    corrected = apply_shifts(calibration, shifts)

    assert bool(shifts["applied"].iloc[0])
    assert abs(quantile_coverage(actual, corrected["q90"].to_numpy()) - 0.9) < 0.01
    # Within tolerance: no shift.
    good = calibration.assign(q90=np.quantile(actual, 0.9))
    assert conformal_shifts(good, [0.9], tolerance=0.02)["shift"].iloc[0] == 0.0
