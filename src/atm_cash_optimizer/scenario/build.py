"""Build the simulated Sri Lankan network: ``lk_calendar`` and ``lk_daily`` (all simulated).

Run: ``python -m atm_cash_optimizer.scenario.build [--n-machines N --years Y]``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from atm_cash_optimizer import setup_logging
from atm_cash_optimizer.config import ScenarioConfig, load_config
from atm_cash_optimizer.io import read_table, write_table
from atm_cash_optimizer.scenario.calendar import build_lk_calendar
from atm_cash_optimizer.scenario.crm import choose_machine_types, simulate_deposits
from atm_cash_optimizer.scenario.denominations import split_denominations
from atm_cash_optimizer.scenario.transform import (
    apply_calendar,
    apply_location_effects,
    assign_location_types,
    bootstrap_network,
    neutralize_uk_holidays,
    shift_dates,
)

logger = logging.getLogger(__name__)

DAYS_PER_YEAR = 364  # whole weeks, so the network ends on the same weekday it starts


def build_scenario(
    nn5: pd.DataFrame,
    clusters: pd.DataFrame,
    cfg: ScenarioConfig,
    denominations: list[str],
    n_machines: int | None = None,
    n_days: int | None = None,
    holiday_csv: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(lk_calendar, lk_daily)``; ``n_machines``/``n_days`` bootstrap a larger network."""
    rng = np.random.default_rng(cfg.seed)

    df = nn5[["machine_id", "date", "withdrawals_lkr", "is_imputed"]]
    df = neutralize_uk_holidays(df, cfg)
    df = df.merge(clusters[["machine_id", "cluster_id", "cluster_label"]], on="machine_id")
    if n_machines is not None or n_days is not None:
        base_days = df.groupby("machine_id").size().iloc[0]
        n_machines = n_machines or df["machine_id"].nunique()
        df = bootstrap_network(df, n_machines, n_days or base_days, cfg, rng)

    df = shift_dates(df, cfg.start_date)
    df["machine_id"] = df["machine_id"].str.replace("NN5_", "LK_", regex=False)

    per_machine = df.groupby("machine_id")["cluster_label"].first().reset_index()
    df["location_type"] = df["machine_id"].map(assign_location_types(per_machine, cfg, rng))
    df = apply_location_effects(df, cfg, rng)

    end = df["date"].max().date()
    calendar = build_lk_calendar(cfg.start_date, end, cfg, holiday_csv)
    df = apply_calendar(df, calendar, cfg)

    machine_ids = sorted(df["machine_id"].unique())
    df["machine_type"] = df["machine_id"].map(
        choose_machine_types(machine_ids, cfg.crm_fraction, rng)
    )
    df["deposits_lkr"] = simulate_deposits(df, cfg, rng)
    df = split_denominations(df, cfg, denominations)
    df["is_simulated"] = True

    event_days = [f"{event}_day" for event in cfg.events]
    flags = [
        "is_no_service_day",
        "is_public_holiday",
        "is_poya",
        "is_avurudu_window",
        "is_christmas_window",
        "is_salary_week",
        "is_long_weekend_eve",
    ]
    columns = [
        "machine_id",
        "date",
        "machine_type",
        "location_type",
        "cluster_id",
        "withdrawals_lkr",
        *[f"withdrawals_{d}_lkr" for d in denominations],
        "deposits_lkr",
        *flags,
        *event_days,
        "is_neutralized",
        "is_imputed",
        "is_simulated",
    ]
    return calendar, df[columns]


def run(n_machines: int | None = None, years: float | None = None) -> None:
    """Build and write ``lk_calendar`` and ``lk_daily``."""
    cfg = load_config("scenario")
    denominations = load_config("machines").tracks["lk"].denominations
    n_days = round(years * DAYS_PER_YEAR) if years else None

    calendar, daily = build_scenario(
        read_table("nn5_daily"),
        read_table("machine_clusters"),
        cfg,
        denominations,
        n_machines,
        n_days,
    )
    write_table(calendar, "lk_calendar")
    write_table(daily, "lk_daily")
    logger.info(
        "lk_daily: %d machines (%d CRM), %s to %s",
        daily["machine_id"].nunique(),
        daily.loc[daily["machine_type"] == "CRM", "machine_id"].nunique(),
        daily["date"].min().date(),
        daily["date"].max().date(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-machines", type=int, help="larger network: number of machines")
    parser.add_argument("--years", type=float, help="larger network: length in years")
    args = parser.parse_args()
    setup_logging()
    run(args.n_machines, args.years)


if __name__ == "__main__":
    main()
