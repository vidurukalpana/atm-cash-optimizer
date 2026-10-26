"""Hand-checkable tests for the replenishment MILPs (one small machine each)."""

import numpy as np
import pandas as pd

from atm_cash_optimizer.config import load_config
from atm_cash_optimizer.optimization.model import solve_atm, solve_crm
from atm_cash_optimizer.optimization.types import (
    TOTAL,
    ForecastWindow,
    MachineState,
    PlanSettings,
)

ORIGIN = pd.Timestamp("2025-01-01")
MILLION = 1_000_000.0
BUNDLE = 100 * 5000  # one bundle of the "total" denomination = LKR 500,000


def _settings(visit_fee: float | None = None) -> PlanSettings:
    costs = load_config("costs")
    if visit_fee is not None:
        costs = costs.model_copy(update={"visit_fee": visit_fee})
    return PlanSettings(alpha=0.95, costs=costs, lead_days=1, bundle_notes=100, largest_note=5000)


def _atm(cash: float, capacity: float) -> MachineState:
    return MachineState("A", "ATM", {TOTAL: cash}, {TOTAL: capacity}, {TOTAL: 1.0})


def _window(median: list[float], upper: list[float] | None = None) -> ForecastWindow:
    """Cumulative forecasts; q05 = 0.8 × median, q95 = ``upper`` (default 1.2 × median)."""
    med = np.array(median)
    up = np.array(upper) if upper is not None else 1.2 * med
    return ForecastWindow(ORIGIN, {0.05: 0.8 * med, 0.5: med, 0.95: up})


def _steady(per_day: float, days: int) -> list[float]:
    return [per_day * t for t in range(1, days + 1)]


def test_zero_demand_means_no_visit() -> None:
    plan = solve_atm(_atm(MILLION, 10 * MILLION), _window([0.0] * 7), _settings(), set())

    assert plan.visits == []
    assert plan.status == "Optimal" and not plan.used_slack


def test_visit_before_no_service_days() -> None:
    # 1M in the machine; demand passes 1M on day 3; days 2 and 3 have no service ⇒ visit day 1.
    window = _window([0.3e6, 0.6e6, 1.5e6], upper=[0.3e6, 0.6e6, 1.5e6])

    plan = solve_atm(_atm(MILLION, 10 * MILLION), window, _settings(), no_service_days={2, 3})

    assert [v.day for v in plan.visits] == [1]
    assert plan.visits[0].load[TOTAL] == BUNDLE  # 1M + 0.5M covers 1.5M
    assert not plan.used_slack


def test_small_capacity_needs_several_visits() -> None:
    # Capacity 1M, demand 3.5M over the week ⇒ one load cannot cover it. (q95 = median here: with
    # a wider q95 a 1M machine cannot cover a week's q95 at all, and slack is used instead.)
    demand = _steady(0.5e6, 7)
    plan = solve_atm(_atm(MILLION, MILLION), _window(demand, upper=demand), _settings(), set())

    assert len(plan.visits) >= 2
    assert all(v.load[TOTAL] <= MILLION for v in plan.visits)
    assert not plan.used_slack


def test_higher_visit_fee_means_fewer_or_equal_visits() -> None:
    state, window = _atm(MILLION, 10 * MILLION), _window(_steady(0.5e6, 14))

    cheap = solve_atm(state, window, _settings(visit_fee=10.0), set())
    expensive = solve_atm(state, window, _settings(visit_fee=500_000.0), set())

    assert len(expensive.visits) <= len(cheap.visits)
    assert len(expensive.visits) < len(cheap.visits)  # the difference is large enough to show


def test_crm_with_large_deposits_schedules_a_removal() -> None:
    # Deposits exceed withdrawals by about 2M a day: a 15M vault of 20M fills in ~2 days.
    state = MachineState("C", "CRM", {TOTAL: 15 * MILLION}, {TOTAL: 20 * MILLION}, {TOTAL: 1.0})
    net = -np.array(_steady(2 * MILLION, 7))
    window = ForecastWindow(ORIGIN, {0.05: 1.5 * net, 0.5: net, 0.95: 0.5 * net})

    plan = solve_crm(state, window, _settings(), set(), vault_capacity=20 * MILLION)

    assert any(v.remove > 0 for v in plan.visits)
    assert any(v.action in ("REMOVE", "LOAD_AND_REMOVE") for v in plan.visits)
    assert not plan.used_slack


def test_impossible_safety_uses_slack() -> None:
    # Empty machine with 1M capacity, but 5M demand tomorrow.
    plan = solve_atm(_atm(0.0, MILLION), _window([5 * MILLION]), _settings(), set())

    assert plan.used_slack
    assert plan.status == "Optimal"


def test_committed_decision_is_the_first_orderable_day() -> None:
    window = _window([0.3e6, 0.6e6, 1.5e6], upper=[0.3e6, 0.6e6, 1.5e6])
    plan = solve_atm(_atm(MILLION, 10 * MILLION), window, _settings(), no_service_days={2, 3})

    assert plan.committed(lead_days=1) is plan.visits[0]
    assert plan.committed(lead_days=1).date == ORIGIN + pd.Timedelta(days=1)
