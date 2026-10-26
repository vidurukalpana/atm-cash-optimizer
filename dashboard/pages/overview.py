"""Overview: all machines at a glance and how the planning works."""

import streamlit as st

import ui
from data import TRACKS, track_overview

track = ui.current_track()
info = track_overview(track)

ui.header("ATM Cash Optimizer", "Forecast-driven cash planning for ATMs and CRMs")
ui.simulated_badge(track)

cols = st.columns(4)
cols[0].metric("ATMs", info["atms"])
cols[1].metric("CRMs", info["crms"])
cols[2].metric("Days of History", f"{info['days']:,}")
cols[3].metric("Simulated Machines", info["simulated_machines"])

period = f"{info['first_date']:%d %b %Y} to {info['last_date']:%d %b %Y}"
st.caption(f"{TRACKS[track]}: {info['machines']} machines, {period}")

st.subheader("How It Works")
steps = st.columns(4)
with steps[0]:
    ui.step_card(1, "Forecast", "Demand range for the next 14 days")
with steps[1]:
    ui.step_card(2, "Plan", "Cheapest visits and amounts")
with steps[2]:
    ui.step_card(3, "Simulate", "Replay past days, day by day")
with steps[3]:
    ui.step_card(4, "Compare", "Cost and empty machines vs simple rules")
