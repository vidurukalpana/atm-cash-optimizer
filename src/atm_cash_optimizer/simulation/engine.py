"""Day-by-day replay of one machine under one policy.

Each simulated day:

1. execute the orders for today (loads clipped at capacity, removals at the cash present);
2. customers withdraw: ``served = min(demand, cash)`` per denomination; unmet ⇒ ``stockout``;
3. CRM deposits: ``accepted = min(deposits, vault space)``; rejected ⇒ ``vault_full``;
4. record the end-of-day state;
5. the policy decides, seeing only demand up to today and today's forecast;
6. emergency rule (same for every policy): after a stockout with no visit ordered by the next
   service day, an emergency visit tomorrow fills the machine (CRM: to half the vault).

``spec.dates[0]`` is the origin before the first day: the policy decides there first.
"""

from __future__ import annotations

import logging

import pandas as pd

from atm_cash_optimizer.optimization.types import PlanSettings
from atm_cash_optimizer.simulation.policies import (
    Policy,
    PolicyInfo,
    fill_order,
    next_service_day,
)
from atm_cash_optimizer.simulation.state import MachineSpec, MachineState, Order

logger = logging.getLogger(__name__)


def _execute(orders: list[Order], spec: MachineSpec, state: MachineState) -> tuple[float, float]:
    """Apply today's orders; returns ``(loaded, removed)`` in LKR."""
    loaded = removed = 0.0
    for order in orders:
        for d, amount in order.load.items():
            room = max(0.0, spec.capacity[d] - state.cash[d])
            if amount > room + 1e-6:
                logger.debug("%s %s: load of %s clipped at capacity", spec.machine_id, d, amount)
            added = min(amount, room)
            state.cash[d] += added
            loaded += added
        if order.remove > 0:
            taken = min(order.remove, state.total)
            # CRMs (the only machines with removals) hold one "total" denomination.
            (denom,) = state.cash
            state.cash[denom] -= taken
            removed += taken
    return loaded, removed


def simulate_machine(
    spec: MachineSpec, policy: Policy, settings: PlanSettings, fill_fraction: float
) -> tuple[pd.DataFrame, list[Order]]:
    """Replay ``spec`` under ``policy``.

    Returns:
        ``(daily, orders)``: one row per simulated day (``sim_daily`` columns) and every order
        the policy or the emergency rule created.
    """
    state = MachineState(cash=spec.initial_cash(fill_fraction))
    all_orders: list[Order] = []
    rows = []

    def decide(day: pd.Timestamp) -> None:
        info = PolicyInfo(
            history=spec.history(day), forecast=spec.forecasts.get(day), settings=settings
        )
        new = policy.decide(day, spec, state, info)
        state.pending.extend(new)
        all_orders.extend(new)

    decide(spec.dates[0])
    for i, day in enumerate(spec.dates[1:], start=1):
        start_cash = state.total

        # 1. today's visits
        today = [o for o in state.pending if o.visit_date == day]
        state.pending = [o for o in state.pending if o.visit_date != day]
        loaded, removed = _execute(today, spec, state)

        # 2. withdrawals
        demand = served = 0.0
        for d, series in spec.withdrawals.items():
            wanted = float(series[i])
            got = min(wanted, state.cash[d])
            state.cash[d] -= got
            demand += wanted
            served += got
        stockout = served < demand - 1e-6

        # 3. deposits (CRM)
        deposits = float(spec.deposits[i])
        accepted = 0.0
        if spec.is_crm:
            (denom,) = state.cash
            accepted = min(deposits, max(0.0, spec.capacity[denom] - state.cash[denom]))
            state.cash[denom] += accepted

        # 4. record
        rows.append(
            {
                "policy": policy.name,
                "param": policy.param,
                "machine_id": spec.machine_id,
                "date": day,
                "start_cash": start_cash,
                "loaded": loaded,
                "removed": removed,
                "withdrawals_demand": demand,
                "withdrawals_served": served,
                "deposits_demand": deposits,
                "deposits_accepted": accepted,
                "end_cash": state.total,
                **{f"end_cash_{d}_lkr": v for d, v in state.cash.items()},
                "visit": any(not o.emergency for o in today),
                "emergency_visit": any(o.emergency for o in today),
                "stockout": stockout,
                "vault_full": accepted < deposits - 1e-6,
            }
        )

        # 5. policy decision (forecasts exist up to the day before the last)
        decide(day)

        # 6. emergency rule
        if stockout:
            next_service = next_service_day(day, settings.lead_days, spec.no_service_dates)
            if not state.has_order_by(next_service):
                tomorrow = day + pd.Timedelta(days=1)
                emergency = fill_order(day, tomorrow, spec, state, settings, emergency=True)
                state.pending.append(emergency)
                all_orders.append(emergency)

    return pd.DataFrame(rows), all_orders
