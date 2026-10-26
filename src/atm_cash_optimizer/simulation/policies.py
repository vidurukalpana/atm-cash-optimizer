"""Policies compared in the simulation, all under the same limits.

At the end of each day a policy may order visits; it sees only demand up to that day and that
day's forecast. Limits: lead time, no visits on no-service days, whole bundles, capacity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Protocol

import pandas as pd

from atm_cash_optimizer.optimization import types as opt
from atm_cash_optimizer.optimization.planner import solve_machine
from atm_cash_optimizer.simulation.state import (
    CRM_TARGET_FRACTION,
    MachineSpec,
    MachineState,
    Order,
)


@dataclass(frozen=True)
class PolicyInfo:
    """What a policy may know at the end of ``day``."""

    history: pd.DataFrame
    forecast: opt.ForecastWindow | None
    settings: opt.PlanSettings


class Policy(Protocol):
    name: str
    param: float

    def decide(
        self, day: pd.Timestamp, spec: MachineSpec, state: MachineState, info: PolicyInfo
    ) -> list[Order]: ...


# --- Shared helpers ---


def next_service_day(
    day: pd.Timestamp, lead_days: int, no_service_dates: frozenset[pd.Timestamp]
) -> pd.Timestamp:
    """First date at least ``lead_days`` ahead on which vans run."""
    candidate = day + pd.Timedelta(days=max(lead_days, 1))
    while candidate in no_service_dates:
        candidate += pd.Timedelta(days=1)
    return candidate


def round_down_to_bundles(amount: float, bundle: float) -> float:
    """Whole bundles not above ``amount`` (never negative)."""
    return max(0.0, math.floor(amount / bundle) * bundle)


def fill_order(
    day: pd.Timestamp,
    visit_date: pd.Timestamp,
    spec: MachineSpec,
    state: MachineState,
    settings: opt.PlanSettings,
    emergency: bool = False,
) -> Order:
    """ATM: fill every denomination. CRM: load or remove to half the vault."""
    if spec.is_crm:
        target = CRM_TARGET_FRACTION * spec.capacity[opt.TOTAL]
        gap = target - state.cash[opt.TOTAL]
        load = round_down_to_bundles(gap, settings.bundle_value(opt.TOTAL)) if gap > 0 else 0.0
        return Order(spec.machine_id, day, visit_date, {opt.TOTAL: load}, max(0.0, -gap), emergency)
    load = {
        d: round_down_to_bundles(spec.capacity[d] - state.cash[d], settings.bundle_value(d))
        for d in spec.capacity
    }
    return Order(spec.machine_id, day, visit_date, load, 0.0, emergency)


# --- Policies ---


@dataclass(frozen=True)
class FixedSchedulePolicy:
    """Visit weekly on ``weekday`` (0 = Monday), or the next service day after it."""

    weekday: int
    name: str = "fixed_schedule"

    @property
    def param(self) -> float:
        return float(self.weekday)

    def decide(
        self, day: pd.Timestamp, spec: MachineSpec, state: MachineState, info: PolicyInfo
    ) -> list[Order]:
        lead = info.settings.lead_days
        scheduled = day + pd.Timedelta(days=max(lead, 1))
        if scheduled.weekday() != self.weekday:
            return []
        visit = next_service_day(day, lead, spec.no_service_dates)
        if state.has_order_by(visit):
            return []
        return [fill_order(day, visit, spec, state, info.settings)]


@dataclass(frozen=True)
class ThresholdPolicy:
    """Refill when cash falls below ``threshold`` × capacity (any denomination).

    CRMs are also emptied to half when cash rises above ``(1 − threshold)`` × capacity.
    """

    threshold: float
    name: str = "threshold"

    @property
    def param(self) -> float:
        return self.threshold

    def decide(
        self, day: pd.Timestamp, spec: MachineSpec, state: MachineState, info: PolicyInfo
    ) -> list[Order]:
        low = any(state.cash[d] < self.threshold * spec.capacity[d] for d in spec.capacity)
        high = spec.is_crm and state.total > (1 - self.threshold) * spec.capacity[opt.TOTAL]
        if not (low or high):
            return []
        visit = next_service_day(day, info.settings.lead_days, spec.no_service_dates)
        if state.has_order_by(visit):
            return []
        return [fill_order(day, visit, spec, state, info.settings)]


@dataclass(frozen=True)
class OptimizerPolicy:
    """Re-plan daily with the MILP and send only the first orderable visit."""

    alpha: float
    name: str = "optimizer"

    @property
    def param(self) -> float:
        return self.alpha

    def decide(
        self, day: pd.Timestamp, spec: MachineSpec, state: MachineState, info: PolicyInfo
    ) -> list[Order]:
        if info.forecast is None:
            return []
        machine = opt.MachineState(
            spec.machine_id, spec.machine_type, dict(state.cash), spec.capacity, spec.shares
        )
        no_service = {
            t
            for t in range(1, info.forecast.horizon + 1)
            if day + pd.Timedelta(days=t) in spec.no_service_dates
        }
        settings = replace(info.settings, alpha=self.alpha)
        plan = solve_machine(machine, info.forecast, settings, no_service)
        visit = plan.committed(info.settings.lead_days)
        if visit is None or state.has_order_by(visit.date):
            return []
        return [Order(spec.machine_id, day, visit.date, dict(visit.load), visit.remove)]
