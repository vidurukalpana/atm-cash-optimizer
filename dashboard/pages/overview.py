"""Overview: the network at a glance and how the planning works."""

import streamlit as st

import ui
from data import TRACKS, track_overview

track = ui.current_track()
info = track_overview(track)

ui.header("ATM Cash Optimizer", "Forecast-driven cash planning for ATMs and cash recyclers")
ui.simulated_badge(track)

cols = st.columns(4)
cols[0].metric("Machines", info["machines"])
cols[1].metric("ATMs · CRMs", f"{info['atms']} · {info['crms']}")
cols[2].metric("Days of history", f"{info['days']:,}")
cols[3].metric("Simulated machines", info["simulated_machines"])

st.caption(f"{TRACKS[track]} · {info['first_date']:%d %b %Y} – {info['last_date']:%d %b %Y}")

st.subheader("How it works")
steps = st.columns(4)
with steps[0]:
    ui.step_card(1, "Forecast", "Demand range for the next 14 days")
with steps[1]:
    ui.step_card(2, "Plan", "Cheapest visits and amounts per note")
with steps[2]:
    ui.step_card(3, "Simulate", "Replay past days, day by day")
with steps[3]:
    ui.step_card(4, "Compare", "Cost and stockouts vs simple rules")
