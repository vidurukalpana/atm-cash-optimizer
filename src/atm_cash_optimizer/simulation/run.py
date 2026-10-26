"""Simulate all policy settings on a seeded sample of machines and write the results.

Outputs: ``sim_daily_<track>``, ``sim_summary_<track>``, ``orders_<track>`` (+ CSV in
``data/orders/``), ``reports/results_<track>.md`` and ``reports/frontier_<track>.html``.

Run: ``python -m atm_cash_optimizer.simulation.run --track lk [--machines N] [--timing DAYS]``
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from atm_cash_optimizer import setup_logging
from atm_cash_optimizer.config import load_config
from atm_cash_optimizer.forecasting.backtest import origin_indices
from atm_cash_optimizer.io import get_data_dir, get_reports_dir, read_table, write_table
from atm_cash_optimizer.optimization.planner import (
    capacity_and_shares,
    forecast_window,
    load_planning_data,
    plan_settings,
)
from atm_cash_optimizer.optimization.types import TOTAL
from atm_cash_optimizer.simulation import compare
from atm_cash_optimizer.simulation.engine import simulate_machine
from atm_cash_optimizer.simulation.metrics import network_totals, summarize
from atm_cash_optimizer.simulation.policies import (
    FixedSchedulePolicy,
    OptimizerPolicy,
    Policy,
    ThresholdPolicy,
)
from atm_cash_optimizer.simulation.state import MachineSpec, Order

logger = logging.getLogger(__name__)


def sample_machines(machines: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Seeded sample of ``n`` machines, keeping the ATM/CRM mix."""
    rng = np.random.default_rng(seed)
    n = min(n, len(machines))
    parts = []
    for _, group in machines.groupby("machine_type"):
        k = round(n * len(group) / len(machines))
        parts.append(group.iloc[np.sort(rng.choice(len(group), size=k, replace=False))])
    return pd.concat(parts).sort_values("machine_id", ignore_index=True)


def load_specs(track: str, n_machines: int, seed: int, test_days: int) -> list[MachineSpec]:
    """One :class:`MachineSpec` per sampled machine."""
    data = load_planning_data(track)
    daily = read_table(f"{track}_daily").sort_values(["machine_id", "date"])
    dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
    first = origin_indices(len(dates), test_days)[0]
    sim_dates = dates[first:]
    daily = daily[daily["date"] >= sim_dates[0]]

    specs = []
    for row in sample_machines(data.machines, n_machines, seed).itertuples():
        rows = daily[daily["machine_id"] == row.machine_id]
        capacity, shares = capacity_and_shares(track, row.machine_type, row.location_type)
        if TOTAL in capacity:
            withdrawals = {TOTAL: rows["withdrawals_lkr"].to_numpy(dtype=float)}
        else:
            withdrawals = {d: rows[f"withdrawals_{d}_lkr"].to_numpy(dtype=float) for d in capacity}
        deposits = (
            rows["deposits_lkr"].to_numpy(dtype=float)
            if "deposits_lkr" in rows
            else np.zeros(len(rows))
        )
        forecasts = {
            o: forecast_window(data, o, row.machine_id, row.machine_type)
            for o in sim_dates
            if o in data.forecasts
        }
        specs.append(
            MachineSpec(
                machine_id=row.machine_id,
                machine_type=row.machine_type,
                location_type=row.location_type,
                capacity=capacity,
                shares=shares,
                dates=sim_dates,
                withdrawals=withdrawals,
                deposits=deposits,
                no_service_dates=data.no_service_dates,
                forecasts=forecasts,
            )
        )
    return specs


def policies() -> list[Policy]:
    """Every setting in ``backtest.yaml``; optimizer first so the slow tasks start early."""
    cfg = load_config("backtest")
    return [
        *[OptimizerPolicy(a) for a in cfg.sweeps.alpha],
        FixedSchedulePolicy(cfg.fixed_schedule_weekday),
        *[ThresholdPolicy(t) for t in cfg.sweeps.threshold],
    ]


def _truncate(spec: MachineSpec, days: int) -> MachineSpec:
    return replace(
        spec,
        dates=spec.dates[: days + 1],
        withdrawals={d: w[: days + 1] for d, w in spec.withdrawals.items()},
        deposits=spec.deposits[: days + 1],
    )


def orders_frame(orders: list[tuple[str, float, Order]], last_date: pd.Timestamp) -> pd.DataFrame:
    """One row per order executed within the simulated days."""
    rows = [
        {
            "policy": policy,
            "param": param,
            "machine_id": o.machine_id,
            "created": o.created,
            "visit_date": o.visit_date,
            "action": o.action,
            **{f"amount_{d}_lkr": v for d, v in o.load.items()},
            "remove_amount_lkr": o.remove,
            "emergency": o.emergency,
        }
        for policy, param, o in orders
        if o.visit_date <= last_date
    ]
    return pd.DataFrame(rows)


def run(track: str, n_machines: int | None = None, timing: int | None = None) -> None:
    """Simulate ``track``; ``timing = N`` times N optimizer days and writes nothing."""
    backtest = load_config("backtest")
    model = load_config("model")
    costs = load_config("costs")
    fill = load_config("machines").initial_fill_fraction
    n_machines = n_machines or backtest.simulation_machines

    specs = load_specs(track, n_machines, backtest.seed, backtest.test_days)
    settings = plan_settings(backtest.sweeps.alpha[0])
    run_policies = policies()
    if timing:
        specs = [_truncate(s, timing) for s in specs]
        run_policies = [OptimizerPolicy(max(backtest.sweeps.alpha))]
    n_atm = sum(not s.is_crm for s in specs)
    logger.info(
        "%s: %d machines (%d ATM, %d CRM), %d days, %d policy settings, %d cores",
        track,
        len(specs),
        n_atm,
        len(specs) - n_atm,
        len(specs[0].dates) - 1,
        len(run_policies),
        model.n_jobs,
    )

    started = time.perf_counter()
    results = Parallel(n_jobs=model.n_jobs, batch_size=1)(
        delayed(simulate_machine)(spec, policy, settings, fill)
        for policy in run_policies
        for spec in specs
    )
    seconds = time.perf_counter() - started
    if timing:
        full_days = backtest.test_days
        estimate = seconds * full_days / timing * len(backtest.sweeps.alpha) / 60
        logger.info(
            "TIMING %s: %d days of the optimizer in %.0fs ⇒ full run ≈ %.0f min "
            "(the simple policies add only seconds)",
            track,
            timing,
            seconds,
            estimate,
        )
        return

    daily = pd.concat([r[0] for r in results], ignore_index=True)
    task_policies = [p for p in run_policies for _ in specs]  # same order as the tasks
    orders = [
        (policy.name, policy.param, order)
        for policy, (_, created) in zip(task_policies, results, strict=True)
        for order in created
    ]
    summary = summarize(daily, costs, backtest.warmup_days)
    machines = pd.DataFrame(
        {
            "machine_id": [s.machine_id for s in specs],
            "machine_type": [s.machine_type for s in specs],
            "location_type": [s.location_type for s in specs],
        }
    )

    write_table(daily, f"sim_daily_{track}")
    write_table(summary.merge(machines, on="machine_id"), f"sim_summary_{track}")
    order_rows = orders_frame(orders, daily["date"].max())
    write_table(order_rows, f"orders_{track}")
    orders_dir = get_data_dir() / "orders"
    orders_dir.mkdir(parents=True, exist_ok=True)
    order_rows.to_csv(orders_dir / f"orders_{track}.csv", index=False)

    reports = get_reports_dir()
    reports.mkdir(parents=True, exist_ok=True)
    setup = {
        "machines": f"{len(specs)} ({n_atm} ATM, {len(specs) - n_atm} CRM), seeded sample",
        "simulated days": f"{len(specs[0].dates) - 1} ({specs[0].dates[1].date()} to "
        f"{specs[0].dates[-1].date()}), first {backtest.warmup_days} excluded",
        "costs": f"visit LKR {costs.visit_fee:,.0f}, emergency LKR "
        f"{costs.emergency_visit_fee:,.0f}, idle cash {costs.idle_cash_annual_rate:.0%} a year, "
        f"stockout LKR {costs.stockout_penalty_per_day:,.0f} per day",
        "run time": f"{seconds:.0f} s on {model.n_jobs} cores",
    }
    report = compare.results_report(track, summary, machines, setup)
    (reports / f"results_{track}.md").write_text(report, encoding="utf-8")
    compare.frontier_chart(network_totals(summary), track, reports / f"frontier_{track}.html")
    logger.info("%s done in %.0fs: %s", track, seconds, reports / f"results_{track}.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate the replenishment policies.")
    parser.add_argument("--track", choices=["nn5", "lk"], required=True)
    parser.add_argument("--machines", type=int, help="number of machines (default: config)")
    parser.add_argument("--timing", type=int, metavar="DAYS", help="time N days, write nothing")
    args = parser.parse_args()
    setup_logging()
    run(args.track, args.machines, args.timing)


if __name__ == "__main__":
    main()
