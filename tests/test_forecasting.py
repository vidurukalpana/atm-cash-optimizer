"""Tests for the baseline forecasters, the backtest helpers and the evaluation metrics."""

import numpy as np
import pytest

from atm_cash_optimizer.forecasting import evaluation
from atm_cash_optimizer.forecasting.backtest import history_slice, origin_indices
from atm_cash_optimizer.forecasting.baselines import seasonal_naive, weekday_mean

WEEK = np.array([10.0, 11, 12, 20, 18, 8, 9])


def test_seasonal_naive_on_weekly_pattern() -> None:
    values = np.tile(WEEK, 6)[None, :]  # one series, 6 identical weeks
    origin = np.array([27])  # a Sunday; next day is a Monday

    daily = seasonal_naive(values, origin, horizon=14)

    assert daily[0, 0].tolist() == np.tile(WEEK, 2).tolist()  # exact for a perfect pattern
    assert np.cumsum(daily[0, 0])[-1] == 2 * WEEK.sum()  # h = 14 repeats the week


def test_weekday_mean_hand_computed() -> None:
    values = np.arange(1.0, 36)[None, :]  # day i has value i + 1
    origin = np.array([34])

    daily = weekday_mean(values, origin, horizon=8, weeks=4)

    # Day o+1 has the weekday of days 28, 21, 14, 7 → values 29, 22, 15, 8.
    assert daily[0, 0, 0] == (29 + 22 + 15 + 8) / 4
    assert daily[0, 0, 7] == daily[0, 0, 0]  # o+8 is the same weekday as o+1


def test_history_never_after_origin() -> None:
    window = history_slice(40, days=28)
    assert (window.start, window.stop) == (13, 41)  # days 13 .. 40, never 41+

    # Baselines ignore everything after the origin.
    values = np.arange(60.0)[None, :]
    changed = np.where(np.arange(60) > 40, 1e9, values)
    for forecast in (seasonal_naive, lambda v, o, h: weekday_mean(v, o, h, weeks=4)):
        assert np.array_equal(
            forecast(values, np.array([40]), 14), forecast(changed, np.array([40]), 14)
        )


def test_origins_are_the_days_before_the_test_period() -> None:
    idx = origin_indices(n_days=100, test_days=10)

    assert idx.tolist() == list(range(89, 99))  # forecast days 90 .. 99 at h = 1


def test_metrics_hand_computed() -> None:
    actual = np.array([100.0, 200.0])
    forecast = np.array([110.0, 150.0])

    assert evaluation.mae(actual, forecast) == 30
    assert evaluation.smape(actual, forecast) == pytest.approx(100 * (20 / 210 + 100 / 350) / 2)
    assert evaluation.pinball(actual, forecast, 0.5) == 15  # = MAE / 2 for the median
    assert evaluation.coverage(actual, forecast - 20, forecast + 20) == 0.5
