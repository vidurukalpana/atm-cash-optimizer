"""Plan every machine of a track for one origin date (rolling horizon).

Only each plan's first orderable visit is meant to be sent; the rest is re-planned tomorrow.

Run: ``python -m atm_cash_optimizer.optimization.planner --track lk --date 2025-08-10``
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from atm_cash_optimizer import setup_logging
from atm_cash_optimizer.config import load_config
from atm_cash_optimizer.forecasting.evaluation import quantile_column
from atm_cash_optimizer.io import read_table
from atm_cash_optimizer.optimization.model import solve_atm, solve_crm
from atm_cash_optimizer.optimization.types import (
    TOTAL,
    ForecastWindow,
    MachineState,
    Plan,
    PlanSettings,
)

logger = logging.getLogger(__name__)

FORECAST_MODEL = "quantile_lgbm"


@dataclass(frozen=True)
class PlanningData:
    """Inputs of :func:`plan_all`, loaded once per track."""

    track: str
    machines: pd.DataFrame  # machine_id, machine_type, location_type
    forecasts: dict[pd.Timestamp, pd.DataFrame]  # quantile forecasts per origin date
    quantiles: list[float]
    no_service_dates: frozenset[pd.Timestamp]


def load_planning_data(track: str) -> PlanningData:
    """Quantile forecasts, machines and no-service days of a track."""
    model_cfg = load_config("model")
    forecasts = read_table(f"forecasts_{track}")
    forecasts = forecasts[forecasts["model"] == FORECAST_MODEL]
    if forecasts.empty:
        raise ValueError(f"no {FORECAST_MODEL} forecasts for {track}; run forecasting.train")

    if track == "lk":
        daily = read_table("lk_daily")
        machines = daily.groupby("machine_id", as_index=False)[
            ["machine_type", "location_type"]
        ].first()
        calendar = read_table("lk_calendar")
        no_service = frozenset(calendar.loc[calendar["is_no_service_day"], "date"])
    else:
        ids = read_table("nn5_daily")["machine_id"].unique()
        machines = pd.DataFrame({"machine_id": ids, "machine_type": "ATM", "location_type": None})
        no_service = frozenset()

    return PlanningData(
        track=track,
        machines=machines.sort_values("machine_id", ignore_index=True),
        forecasts={o: g for o, g in forecasts.groupby("origin_date")},
        quantiles=list(model_cfg.quantiles),
        no_service_dates=no_service,
    )


def capacity_and_shares(
    track: str, machine_type: str, location_type: str | None
) -> tuple[dict[str, float], dict[str, float]]:
    """Capacity per denomination and each denomination's share of demand."""
    machines = load_config("machines")
    if track == "nn5":
        return {TOTAL: machines.atm.total_capacity_lkr}, {TOTAL: 1.0}
    if machine_type == "CRM":
        return {TOTAL: machines.crm.vault_capacity_lkr}, {TOTAL: 1.0}
    scenario = load_config("scenario")
    all_shares = scenario.denomination_shares
    shares = all_shares.get(location_type, all_shares["default"])
    capacity = {
        d: float(machines.atm.cassettes[d] * machines.atm.notes_per_cassette * int(d))
        for d in machines.tracks["lk"].denominations
    }
    return capacity, dict(shares)


def initial_states(data: PlanningData) -> dict[str, MachineState]:
    """Machines filled to ``initial_fill_fraction`` (for standalone runs)."""
    fill = load_config("machines").initial_fill_fraction
    states = {}
    for row in data.machines.itertuples():
        capacity, shares = capacity_and_shares(data.track, row.machine_type, row.location_type)
        states[row.machine_id] = MachineState(
            machine_id=row.machine_id,
            machine_type=row.machine_type,
            cash={d: fill * c for d, c in capacity.items()},
            capacity=capacity,
            shares=shares,
        )
    return states


def forecast_window(
    data: PlanningData, origin_date: pd.Timestamp, machine_id: str, machine_type: str
) -> ForecastWindow:
    """Withdrawal quantiles for ATMs, net-outflow quantiles for CRMs."""
    target = "net" if machine_type == "CRM" else "withdrawals"
    rows = data.forecasts[origin_date]
    rows = rows[(rows["machine_id"] == machine_id) & (rows["target"] == target)].sort_values("h")
    if rows.empty:
        raise ValueError(f"no {target} forecast for {machine_id} at {origin_date.date()}")
    return ForecastWindow(
        origin_date=origin_date,
        quantiles={q: rows[quantile_column(q)].to_numpy(dtype=float) for q in data.quantiles},
    )


def plan_settings(alpha: float) -> PlanSettings:
    """Solver settings from the configs."""
    machines_cfg = load_config("machines")
    return PlanSettings(
        alpha=alpha,
        costs=load_config("costs"),
        lead_days=machines_cfg.order_lead_days,
        bundle_notes=machines_cfg.bundle_notes,
        largest_note=max(int(d) for d in machines_cfg.atm.cassettes),
    )


def solve_machine(
    state: MachineState,
    window: ForecastWindow,
    settings: PlanSettings,
    no_service_days: set[int],
) -> Plan:
    """Solve with the ATM or the CRM model."""
    if state.machine_type == "CRM":
        return solve_crm(state, window, settings, no_service_days, state.capacity[TOTAL])
    return solve_atm(state, window, settings, no_service_days)


def plan_all(
    origin_date: pd.Timestamp | str,
    track: str,
    alpha: float,
    states: dict[str, MachineState] | None = None,
    data: PlanningData | None = None,
) -> list[Plan]:
    """Solve every machine for ``origin_date`` at safety level ``alpha`` (a forecast quantile).

    ``states`` defaults to :func:`initial_states`; pass pre-loaded ``data`` when planning many days.
    """
    origin_date = pd.Timestamp(origin_date)
    data = data or load_planning_data(track)
    states = states or initial_states(data)
    model_cfg = load_config("model")
    settings = plan_settings(alpha)
    if origin_date not in data.forecasts:
        raise ValueError(f"no forecasts for origin {origin_date.date()} on {track}")

    jobs = []
    for row in data.machines.itertuples():
        window = forecast_window(data, origin_date, row.machine_id, row.machine_type)
        no_service = {
            t
            for t in range(1, window.horizon + 1)
            if origin_date + pd.Timedelta(days=t) in data.no_service_dates
        }
        jobs.append(delayed(solve_machine)(states[row.machine_id], window, settings, no_service))
    return Parallel(n_jobs=model_cfg.n_jobs)(jobs)


def plans_frame(plans: list[Plan], lead_days: int) -> pd.DataFrame:
    """One row per planned visit; ``committed`` marks the decision sent today."""
    rows = []
    for plan in plans:
        committed = plan.committed(lead_days)
        for visit in plan.visits:
            rows.append(
                {
                    "machine_id": plan.machine_id,
                    "origin_date": plan.origin_date,
                    "visit_date": visit.date,
                    "day": visit.day,
                    "action": visit.action,
                    **{f"load_{d}_lkr": amount for d, amount in visit.load.items()},
                    "remove_lkr": visit.remove,
                    "committed": visit is committed,
                    "alpha": plan.alpha,
                    "objective": plan.objective,
                    "status": plan.status,
                    "used_slack": plan.used_slack,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Plan one day and log a summary (writes nothing)."""
    parser = argparse.ArgumentParser(description="Plan all machines for one origin date.")
    parser.add_argument("--track", choices=["nn5", "lk"], default="lk")
    parser.add_argument("--date", help="origin date (default: first date with forecasts)")
    parser.add_argument("--alpha", type=float, default=0.95)
    args = parser.parse_args()
    setup_logging()

    data = load_planning_data(args.track)
    origin = pd.Timestamp(args.date) if args.date else min(data.forecasts)
    started = time.perf_counter()
    plans = plan_all(origin, args.track, args.alpha, data=data)
    seconds = time.perf_counter() - started

    lead = load_config("machines").order_lead_days
    visits = plans_frame(plans, lead)
    statuses = pd.Series([p.status for p in plans]).value_counts().to_dict()
    logger.info(
        "%s %s alpha=%.2f: %d machines in %.1fs, statuses %s, %d with slack",
        args.track,
        origin.date(),
        args.alpha,
        len(plans),
        seconds,
        statuses,
        sum(p.used_slack for p in plans),
    )
    if not visits.empty:
        logger.info(
            "%d visits planned over %d days, %d to send today; by action: %s",
            len(visits),
            int(visits["day"].max()),
            int(visits["committed"].sum()),
            visits["action"].value_counts().to_dict(),
        )
        logger.info("first decisions to send:\n%s", visits[visits["committed"]].head(15))
    mean_objective = np.mean([p.objective for p in plans])
    logger.info("mean objective per machine: LKR %s", f"{mean_objective:,.0f}")


if __name__ == "__main__":
    main()
