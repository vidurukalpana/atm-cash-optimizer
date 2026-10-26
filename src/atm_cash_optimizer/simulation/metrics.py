"""Policy metrics exactly as defined in ``docs/metrics.md``."""

from __future__ import annotations

import pandas as pd

from atm_cash_optimizer.config import CostsConfig

GROUP = ["policy", "param", "machine_id"]
DAYS_PER_YEAR = 365


def after_warmup(daily: pd.DataFrame, warmup_days: int) -> pd.DataFrame:
    """Drop the first ``warmup_days`` simulated days."""
    first_kept = daily["date"].min() + pd.Timedelta(days=warmup_days)
    return daily[daily["date"] >= first_kept]


def summarize(daily: pd.DataFrame, costs: CostsConfig, warmup_days: int) -> pd.DataFrame:
    """One row per ``(policy, param, machine_id)``."""
    df = after_warmup(daily, warmup_days)
    out = df.groupby(GROUP).agg(
        days=("date", "size"),
        avg_idle_cash=("end_cash", "mean"),
        sum_end_cash=("end_cash", "sum"),
        visits=("visit", "sum"),
        emergency_visits=("emergency_visit", "sum"),
        stockout_days=("stockout", "sum"),
        unmet_withdrawals_lkr=("withdrawals_demand", "sum"),
        served=("withdrawals_served", "sum"),
        vault_full_days=("vault_full", "sum"),
    )
    out["unmet_withdrawals_lkr"] -= out.pop("served")
    out["idle_cash_cost"] = out.pop("sum_end_cash") * costs.idle_cash_annual_rate / DAYS_PER_YEAR
    out["stockout_rate"] = out["stockout_days"] / out["days"]
    out["total_cost"] = (
        costs.visit_fee * out["visits"]
        + costs.emergency_visit_fee * out["emergency_visits"]
        + out["idle_cash_cost"]
        + costs.stockout_penalty_per_day * out["stockout_days"]
    )
    return out.reset_index()


def network_totals(summary: pd.DataFrame) -> pd.DataFrame:
    """Per ``(policy, param)``: sums over machines, rates over all machine-days."""
    sums = [
        "visits",
        "emergency_visits",
        "stockout_days",
        "unmet_withdrawals_lkr",
        "vault_full_days",
        "idle_cash_cost",
        "total_cost",
        "days",
    ]
    out = summary.groupby(["policy", "param"])[sums].sum()
    weighted = summary.assign(idle=summary["avg_idle_cash"] * summary["days"])
    out["avg_idle_cash"] = weighted.groupby(["policy", "param"])["idle"].sum() / out["days"]
    out["stockout_rate"] = out["stockout_days"] / out["days"]
    return out.reset_index()
