"""Machine Insights: demand forecast and cash history of one machine."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import ui
from atm_cash_optimizer.simulation.compare import best_settings
from atm_cash_optimizer.simulation.metrics import network_totals
from data import (
    capacities,
    machine_forecasts,
    machine_withdrawals,
    sim_daily,
    sim_summary,
    simulation_info,
)

BAND = "rgba(31,111,235,0.15)"

track = ui.current_track()
summary = sim_summary(track)
machines = summary.drop_duplicates("machine_id").set_index("machine_id").sort_index()
best = best_settings(network_totals(summary)).set_index("policy")


def describe(machine_id: str) -> str:
    kind = machines.at[machine_id, "machine_type"]
    location = machines.at[machine_id, "location_type"]
    place = f", {location.replace('_', ' ').title()}" if isinstance(location, str) else ""
    return f"{machine_id} ({kind}{place})"


ui.header("Machine Insights", "Demand forecast and cash history of one machine")
ui.simulated_badge(track)
ui.simulation_chips(simulation_info(track))

left, right = st.columns([2, 3])
machine_id = left.selectbox("Machine", list(machines.index), format_func=describe)
policy = right.segmented_control(
    "Strategy",
    list(best.index),
    default="optimizer",
    format_func=ui.POLICY_NAMES.get,
)
policy = policy or "optimizer"
param = best.at[policy, "param"]

# --- Cash in the machine ---
st.subheader("Cash in the Machine")
runs = summary[(summary["policy"] == policy) & (summary["param"] == param)]
run = runs.set_index("machine_id").loc[machine_id]
cols = st.columns(4)
cols[0].metric("Van Visits", int(run["visits"]))
cols[1].metric("Days Ran Empty", int(run["stockout_days"]))
cols[2].metric("Average Cash (LKR)", ui.lkr_millions(run["avg_idle_cash"]))
cols[3].metric("Total Cost (LKR)", ui.lkr_millions(run["total_cost"]))

daily = sim_daily(track)
daily = daily[
    (daily["policy"] == policy) & (daily["param"] == param) & (daily["machine_id"] == machine_id)
]
capacity = sum(capacities(track)[machine_id].values()) / 1e6
visits = daily[daily["visit"] | daily["emergency_visit"]]
empty = daily[daily["stockout"]]

cash = go.Figure()
cash.add_scatter(
    x=daily["date"], y=daily["end_cash"] / 1e6, name="Cash", line={"color": ui.BRIGHT, "width": 2}
)
cash.add_scatter(
    x=visits["date"],
    y=visits["end_cash"] / 1e6,
    name="Van Visit",
    mode="markers",
    marker={"color": ui.TEAL, "size": 8, "symbol": "triangle-up"},
)
cash.add_scatter(
    x=empty["date"],
    y=empty["end_cash"] / 1e6,
    name="Ran Empty",
    mode="markers",
    marker={"color": "#DC2626", "size": 10, "symbol": "x"},
)
cash.add_hline(
    y=capacity, line={"color": ui.SLATE, "dash": "dash", "width": 1}, annotation_text="Capacity"
)
cash = ui.style_figure(cash, height=360)
cash.update_yaxes(title="LKR M", rangemode="tozero")
st.plotly_chart(cash)

# --- Demand forecast ---
st.subheader("Demand Forecast")
forecasts = machine_forecasts(track, machine_id)
withdrawals = machine_withdrawals(track, machine_id)
origins = pd.to_datetime(forecasts["origin_date"].unique())
latest_full = origins[origins <= withdrawals.index.max() - pd.Timedelta(days=14)].max()
day = st.date_input(
    "Forecast Made On",
    value=latest_full,
    min_value=origins.min(),
    max_value=origins.max(),
)
window = forecasts[forecasts["origin_date"] == pd.Timestamp(day)]
dates = pd.Timestamp(day) + pd.to_timedelta(window["h"], unit="D")
actual = withdrawals.reindex(dates).cumsum()

fan = go.Figure()
fan.add_scatter(x=dates, y=window["q95"] / 1e6, line={"width": 0}, showlegend=False)
fan.add_scatter(
    x=dates,
    y=window["q05"] / 1e6,
    fill="tonexty",
    fillcolor=BAND,
    line={"width": 0},
    name="Likely Range (90%)",
)
fan.add_scatter(
    x=dates, y=window["q50"] / 1e6, name="Forecast", line={"color": ui.BRIGHT, "width": 2}
)
fan.add_scatter(
    x=actual.index,
    y=actual / 1e6,
    name="Actual",
    mode="lines+markers",
    line={"color": ui.NAVY, "width": 2},
)
fan = ui.style_figure(fan, height=360)
fan.update_yaxes(title="Cash Withdrawn So Far (LKR M)", rangemode="tozero")
st.plotly_chart(fan)
st.caption("Running total of cash withdrawn over the next 14 days.")
