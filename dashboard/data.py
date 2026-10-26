"""Read-only data access for the dashboard; results are cached per session."""

from __future__ import annotations

import re

import pandas as pd
import streamlit as st

from atm_cash_optimizer.config import load_config
from atm_cash_optimizer.io import query
from atm_cash_optimizer.optimization.planner import capacity_and_shares

TRACKS = {"lk": "Sri Lanka (Simulated)", "nn5": "UK ATMs (Real Data)"}


def _check(track: str) -> None:
    # Track names go into SQL, so only known names are accepted.
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}")


def _check_machine(machine_id: str) -> None:
    if not re.fullmatch(r"[A-Z0-9_]+", machine_id):
        raise ValueError(f"invalid machine id {machine_id!r}")


@st.cache_data(show_spinner=False)
def tables() -> set[str]:
    return set(query("select table_name from information_schema.tables")["table_name"])


def available_tracks() -> list[str]:
    """Tracks whose simulation results exist."""
    return [t for t in TRACKS if f"sim_summary_{t}" in tables()]


@st.cache_data(show_spinner=False)
def track_overview(track: str) -> dict:
    """Machines, history length and simulation sample of a track."""
    _check(track)
    machine_type = "machine_type" if track == "lk" else "'ATM'"
    history = query(
        f"""
        select count(distinct machine_id) as machines,
               count(distinct case when {machine_type} = 'CRM' then machine_id end) as crms,
               count(distinct date) as days,
               min(date) as first_date,
               max(date) as last_date
        from {track}_daily
        """
    ).iloc[0]
    simulated = query(
        f"select count(distinct machine_id) as n, max(days) as days from sim_summary_{track}"
    ).iloc[0]
    return {
        "machines": int(history["machines"]),
        "atms": int(history["machines"] - history["crms"]),
        "crms": int(history["crms"]),
        "days": int(history["days"]),
        "first_date": pd.Timestamp(history["first_date"]),
        "last_date": pd.Timestamp(history["last_date"]),
        "simulated_machines": int(simulated["n"]),
        "simulated_days": int(simulated["days"]),
    }


@st.cache_data(show_spinner=False)
def sim_summary(track: str) -> pd.DataFrame:
    """Per policy setting and machine: the simulation metrics plus machine type and location."""
    _check(track)
    return query(f"select * from sim_summary_{track}")


@st.cache_data(show_spinner=False)
def sim_daily(track: str) -> pd.DataFrame:
    """Day-by-day simulation state of every machine under every policy setting."""
    _check(track)
    return query(f"select * from sim_daily_{track}")


@st.cache_data(show_spinner=False)
def orders(track: str) -> pd.DataFrame:
    """Van visits ordered in the simulation."""
    _check(track)
    return query(f"select * from orders_{track}")


@st.cache_data(show_spinner=False)
def capacities(track: str) -> dict[str, dict[str, float]]:
    """Capacity per denomination of every simulated machine."""
    machines = sim_summary(track)[["machine_id", "machine_type", "location_type"]]
    return {
        row.machine_id: capacity_and_shares(track, row.machine_type, row.location_type)[0]
        for row in machines.drop_duplicates("machine_id").itertuples()
    }


@st.cache_data(show_spinner=False)
def machine_forecasts(track: str, machine_id: str) -> pd.DataFrame:
    """Quantile forecasts of one machine's withdrawals for every origin date."""
    _check(track)
    _check_machine(machine_id)
    return query(
        f"""
        select * from forecasts_{track}
        where model = 'quantile_lgbm' and target = 'withdrawals' and machine_id = '{machine_id}'
        order by origin_date, h
        """
    )


@st.cache_data(show_spinner=False)
def machine_withdrawals(track: str, machine_id: str) -> pd.Series:
    """Real daily withdrawals of one machine, indexed by date."""
    _check(track)
    _check_machine(machine_id)
    rows = query(
        f"select date, withdrawals_lkr from {track}_daily where machine_id = '{machine_id}'"
    )
    return rows.set_index("date")["withdrawals_lkr"].sort_index()


@st.cache_data(show_spinner=False)
def simulation_info(track: str) -> dict:
    """Sample size, period and warmup of the simulation."""
    _check(track)
    machines = sim_summary(track).drop_duplicates("machine_id")["machine_type"]
    period = query(f"select min(date) as first, max(date) as last from sim_daily_{track}").iloc[0]
    first, last = pd.Timestamp(period["first"]), pd.Timestamp(period["last"])
    return {
        "machines": len(machines),
        "atms": int((machines == "ATM").sum()),
        "crms": int((machines == "CRM").sum()),
        "first_date": first,
        "last_date": last,
        "days": (last - first).days + 1,
        "warmup_days": load_config("backtest").warmup_days,
        "network_machines": track_overview(track)["machines"],
    }
