"""Tests for the policy simulation: hand-computed replays, policies, metrics, no future data."""

import numpy as np
import pandas as pd
import pytest

from atm_cash_optimizer.config import load_config
from atm_cash_optimizer.optimization.planner import plan_settings
from atm_cash_optimizer.optimization.types import TOTAL, ForecastWindow
from atm_cash_optimizer.simulation.engine import simulate_machine
from atm_cash_optimizer.simulation.metrics import summarize
from atm_cash_optimizer.simulation.policies import (
    FixedSchedulePolicy,
    OptimizerPolicy,
    ThresholdPolicy,
)
from atm_cash_optimizer.simulation.state import MachineSpec

MILLION = 1_000_000.0
ORIGIN = pd.Timestamp("2025-01-05")  # a Sunday; day 1 is Monday 6 January
DAYS = 14
SETTINGS = plan_settings(0.95)


def _spec(no_service: set[int] = frozenset(), forecasts: bool = False) -> MachineSpec:
    """One ATM, capacity 10M, constant demand of 1M a day for 14 days."""
    dates = pd.date_range(ORIGIN, periods=DAYS + 1, freq="D")
    demand = np.full(DAYS + 1, MILLION)
    windows = {}
    if forecasts:  # perfect forecasts: every quantile = the true cumulative demand
        cumulative = np.cumsum(np.full(DAYS, MILLION))
        for o in dates[:-1]:
            windows[o] = ForecastWindow(o, {q: cumulative for q in (0.05, 0.5, 0.95)})
    return MachineSpec(
        machine_id="A",
        machine_type="ATM",
        location_type="branch",
        capacity={TOTAL: 10 * MILLION},
        shares={TOTAL: 1.0},
        dates=dates,
        withdrawals={TOTAL: demand},
        deposits=np.zeros(DAYS + 1),
        no_service_dates=frozenset(ORIGIN + pd.Timedelta(days=t) for t in no_service),
        forecasts=windows,
    )


def test_fixed_schedule_on_constant_demand() -> None:
    # Full machine (10M), 1M a day, Monday visits that fill to the top:
    # week 1 ends at 3M, the Monday visit loads 7M, week 2 ends at 3M again.
    daily, orders = simulate_machine(_spec(), FixedSchedulePolicy(weekday=0), SETTINGS, 1.0)

    assert daily["end_cash"].tolist() == [m * MILLION for m in (9, 8, 7, 6, 5, 4, 3) * 2]
    assert daily.loc[daily["visit"], "date"].dt.day_name().tolist() == ["Monday", "Monday"]
    assert daily["loaded"].sum() == 7 * MILLION
    assert not daily["stockout"].any()


def test_threshold_waits_for_the_next_service_day() -> None:
    # Cash falls below 30 % (3M) at the end of day 8; day 9 has no service ⇒ visit on day 10.
    daily, _ = simulate_machine(
        _spec(no_service={9}), ThresholdPolicy(threshold=0.3), SETTINGS, 1.0
    )

    visits = daily.loc[daily["visit"], "date"].tolist()
    assert visits == [ORIGIN + pd.Timedelta(days=10)]
    assert not daily["stockout"].any()


def test_optimizer_with_perfect_forecasts_never_stocks_out() -> None:
    daily, _ = simulate_machine(
        _spec(no_service={5}, forecasts=True), OptimizerPolicy(alpha=0.95), SETTINGS, 0.3
    )

    assert not daily["stockout"].any()
    assert not daily["emergency_visit"].any()
    assert ORIGIN + pd.Timedelta(days=5) not in set(daily.loc[daily["visit"], "date"])


def test_engine_never_passes_future_demand_to_a_policy() -> None:
    seen = []

    class Spy:
        name, param = "spy", 0.0

        def decide(self, day, spec, state, info):
            seen.append((day, info.history["date"].max(), info.forecast))
            return []

    simulate_machine(_spec(forecasts=True), Spy(), SETTINGS, 1.0)

    assert len(seen) == DAYS + 1  # the origin plus every simulated day
    for day, latest_demand, forecast in seen:
        assert latest_demand <= day
        assert forecast is None or forecast.origin_date == day


def test_metrics_match_the_worked_example() -> None:
    # docs/metrics.md: three days, one visit, one stockout ⇒ total cost LKR 43,506.85.
    daily = pd.DataFrame(
        {
            "policy": "p",
            "param": 0.0,
            "machine_id": "A",
            "date": pd.date_range("2025-01-01", periods=3),
            "end_cash": [10e6, 6e6, 0.0],
            "withdrawals_demand": [4e6, 4e6, 6.5e6],
            "withdrawals_served": [4e6, 4e6, 6e6],
            "visit": [True, False, False],
            "emergency_visit": False,
            "stockout": [False, False, True],
            "vault_full": False,
        }
    )

    row = summarize(daily, load_config("costs"), warmup_days=0).iloc[0]

    assert row["avg_idle_cash"] == pytest.approx(5_333_333.33, abs=0.01)
    assert row["idle_cash_cost"] == pytest.approx(3_506.85, abs=0.01)
    assert row["stockout_rate"] == pytest.approx(1 / 3)
    assert row["unmet_withdrawals_lkr"] == 500_000
    assert row["total_cost"] == pytest.approx(43_506.85, abs=0.01)
