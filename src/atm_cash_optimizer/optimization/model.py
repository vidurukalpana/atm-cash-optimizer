"""Mixed-integer linear programs (MILP) that plan visits for one machine.

Notation (see ``docs/DEVELOPMENT_PLAN.md``, branch 9): days ``t = 1..H``, denominations ``d``.
``U_t`` is the ``alpha``-quantile and ``M_t`` the median of cumulative demand ``C(o, t)``
(``M_0 = 0``). ``y_t`` = visit on day ``t``; ``b_{d,t}`` = bundles loaded;
``X_{d,t} = B_d·b_{d,t}``.

ATM, per denomination (demand split by shares ``s_d``):

1. ``X_{d,t} <= cap_d · y_t``
2. ``y_t = 0`` on no-service days and before the lead time
3. safety: ``I0_d + Σ_{τ<=t} X_{d,τ} + slack_{d,t} >= s_d · U_t``
4. capacity: ``I0_d + Σ_{τ<=t} X_{d,τ} − s_d · M_{t−1} <= cap_d``

minimize ``visit_fee·Σy + rate/365·Σ(I0 + ΣX − s·M_t) + stockout_penalty·Σ slack / B_d``.

CRM, on total cash with removals ``r_t`` and net-outflow quantiles ``N``:

- empty-safety: ``I0 + ΣX − Σr + slack_t >= Q_alpha(N_t)``
- full-safety: ``I0 + ΣX − Σr − Q_{1−alpha}(N_t) − full_t <= vault_capacity``
- removals only on visit days, and never more than the cash expected in the machine
- idle cost on ``I0 + ΣX − Σr − Q_0.5(N_t)``

Slack variables keep every problem feasible: when a rule cannot be met (demand above capacity,
no possible visit), the plan breaks it at a high penalty and is flagged ``used_slack``.
"""

from __future__ import annotations

import pandas as pd
import pulp

from atm_cash_optimizer.optimization.types import (
    TOTAL,
    ForecastWindow,
    MachineState,
    Plan,
    PlanSettings,
    Visit,
)

SOLVER_TIME_LIMIT = 10  # seconds per machine (fixed by the development plan)
SLACK_TOLERANCE = 1e-6
DAYS_PER_YEAR = 365


def _solver() -> pulp.LpSolver:
    # One thread per problem: machines are solved in parallel by the planner instead.
    return pulp.PULP_CBC_CMD(msg=False, timeLimit=SOLVER_TIME_LIMIT, threads=1)


def _blocked(t: int, no_service_days: set[int], lead_days: int) -> bool:
    return t in no_service_days or t < lead_days


def _daily_rate(settings: PlanSettings) -> float:
    return settings.costs.idle_cash_annual_rate / DAYS_PER_YEAR


def solve_atm(
    state: MachineState,
    window: ForecastWindow,
    settings: PlanSettings,
    no_service_days: set[int],
) -> Plan:
    """Cheapest visit plan for an ATM over the forecast window (see module docstring)."""
    days = range(1, window.horizon + 1)
    upper = window.get(settings.alpha)
    median = window.get(0.5)
    denoms = list(state.capacity)
    bundle = {d: settings.bundle_value(d) for d in denoms}

    prob = pulp.LpProblem("atm", pulp.LpMinimize)
    y = {t: pulp.LpVariable(f"y_{t}", cat="Binary") for t in days}
    b = {
        (d, t): pulp.LpVariable(f"b_{d}_{t}", lowBound=0, cat="Integer")
        for d in denoms
        for t in days
    }
    slack = {(d, t): pulp.LpVariable(f"slack_{d}_{t}", lowBound=0) for d in denoms for t in days}

    def loaded(d: str, t: int) -> pulp.LpAffineExpression:
        return pulp.lpSum(bundle[d] * b[d, tau] for tau in range(1, t + 1))

    idle = []
    for t in days:
        if _blocked(t, no_service_days, settings.lead_days):
            prob += y[t] == 0, f"no_visit_{t}"
        previous_median = median[t - 2] if t > 1 else 0.0
        for d in denoms:
            share, cash, cap = state.shares[d], state.cash[d], state.capacity[d]
            prob += bundle[d] * b[d, t] <= cap * y[t], f"load_only_on_visit_{d}_{t}"
            prob += cash + loaded(d, t) + slack[d, t] >= share * upper[t - 1], f"safety_{d}_{t}"
            prob += cash + loaded(d, t) - share * previous_median <= cap, f"capacity_{d}_{t}"
            idle.append(cash + loaded(d, t) - share * median[t - 1])

    costs = settings.costs
    prob += (
        costs.visit_fee * pulp.lpSum(y.values())
        + _daily_rate(settings) * pulp.lpSum(idle)
        + costs.stockout_penalty_per_day
        * pulp.lpSum(slack[d, t] / bundle[d] for d in denoms for t in days)
    )
    prob.solve(_solver())

    visits = [
        Visit(
            day=t,
            date=window.origin_date + pd.Timedelta(days=t),
            action="LOAD",
            load={d: bundle[d] * round(b[d, t].value() or 0) for d in denoms},
        )
        for t in days
        if (y[t].value() or 0) > 0.5
    ]
    return Plan(
        machine_id=state.machine_id,
        origin_date=window.origin_date,
        alpha=settings.alpha,
        visits=visits,
        objective=float(pulp.value(prob.objective) or 0.0),
        status=pulp.LpStatus[prob.status],
        used_slack=any((s.value() or 0) > SLACK_TOLERANCE for s in slack.values()),
    )


def solve_crm(
    state: MachineState,
    window: ForecastWindow,
    settings: PlanSettings,
    no_service_days: set[int],
    vault_capacity: float,
) -> Plan:
    """Cheapest load/remove plan for a CRM on total cash (see module docstring)."""
    days = range(1, window.horizon + 1)
    upper = window.get(settings.alpha)
    lower = window.lower(1 - settings.alpha)
    median = window.get(0.5)
    bundle = settings.bundle_value(TOTAL)
    cash = state.cash[TOTAL]
    big_m = vault_capacity + max(cash, 0.0)  # a removal can never exceed a full vault

    prob = pulp.LpProblem("crm", pulp.LpMinimize)
    y = {t: pulp.LpVariable(f"y_{t}", cat="Binary") for t in days}
    b = {t: pulp.LpVariable(f"b_{t}", lowBound=0, cat="Integer") for t in days}
    r = {t: pulp.LpVariable(f"remove_{t}", lowBound=0) for t in days}
    empty = {t: pulp.LpVariable(f"slack_empty_{t}", lowBound=0) for t in days}
    full = {t: pulp.LpVariable(f"slack_full_{t}", lowBound=0) for t in days}

    def net_change(t: int) -> pulp.LpAffineExpression:
        return pulp.lpSum(bundle * b[tau] - r[tau] for tau in range(1, t + 1))

    idle = []
    for t in days:
        if _blocked(t, no_service_days, settings.lead_days):
            prob += y[t] == 0, f"no_visit_{t}"
        previous_median = median[t - 2] if t > 1 else 0.0
        prob += bundle * b[t] <= vault_capacity * y[t], f"load_only_on_visit_{t}"
        prob += r[t] <= big_m * y[t], f"remove_only_on_visit_{t}"
        prob += cash + net_change(t) - previous_median >= 0, f"remove_at_most_cash_{t}"
        prob += cash + net_change(t) + empty[t] >= upper[t - 1], f"empty_safety_{t}"
        prob += cash + net_change(t) - lower[t - 1] - full[t] <= vault_capacity, f"full_safety_{t}"
        idle.append(cash + net_change(t) - median[t - 1])

    costs = settings.costs
    prob += (
        costs.visit_fee * pulp.lpSum(y.values())
        + _daily_rate(settings) * pulp.lpSum(idle)
        + costs.stockout_penalty_per_day * pulp.lpSum((empty[t] + full[t]) / bundle for t in days)
    )
    prob.solve(_solver())

    visits = []
    for t in days:
        if (y[t].value() or 0) <= 0.5:
            continue
        load = bundle * round(b[t].value() or 0)
        remove = float(r[t].value() or 0)
        action = "LOAD_AND_REMOVE" if load and remove else ("LOAD" if load else "REMOVE")
        visits.append(
            Visit(
                day=t,
                date=window.origin_date + pd.Timedelta(days=t),
                action=action,
                load={TOTAL: load},
                remove=remove,
            )
        )
    used_slack = any((v.value() or 0) > SLACK_TOLERANCE for v in [*empty.values(), *full.values()])
    return Plan(
        machine_id=state.machine_id,
        origin_date=window.origin_date,
        alpha=settings.alpha,
        visits=visits,
        objective=float(pulp.value(prob.objective) or 0.0),
        status=pulp.LpStatus[prob.status],
        used_slack=used_slack,
    )
