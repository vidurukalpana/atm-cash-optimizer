"""Read-only data access for the dashboard; results are cached per session."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from atm_cash_optimizer.io import query
from atm_cash_optimizer.optimization.planner import capacity_and_shares

TRACKS = {"lk": "Sri Lanka (simulated)", "nn5": "UK ATMs (real data)"}


def _check(track: str) -> None:
    # Track names go into SQL, so only known names are accepted.
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}")


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
