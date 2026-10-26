"""Machines: status of every simulated machine on the last simulated day."""

import pandas as pd
import streamlit as st

import ui
from atm_cash_optimizer.simulation.compare import best_settings
from atm_cash_optimizer.simulation.metrics import network_totals
from data import capacities, orders, sim_daily, sim_summary

LOW_CASH_SHARE = 0.2  # a note type below 20% of its space counts as low
RECENT_DAYS = 7
STATUS_COLORS = {"OK": ui.TEAL, "Low cash": "#B45309", "Ran empty": "#DC2626"}
MACHINE_TYPES = {"ATM": "ATM", "CRM": "Cash recycler"}
ACTIONS = {"LOAD": "Load", "REMOVE": "Remove", "LOAD_AND_REMOVE": "Load and remove"}

track = ui.current_track()
summary = sim_summary(track)
network = network_totals(summary)
best_alpha = best_settings(network).query("policy == 'optimizer'")["param"].iloc[0]
options = list(network.index)
default = network.index[(network["policy"] == "optimizer") & (network["param"] == best_alpha)][0]

ui.header("Machines", "Status of every machine on the latest day")
ui.simulated_badge(track)

choice = st.selectbox(
    "Strategy",
    options,
    index=options.index(default),
    format_func=lambda i: ui.setting_name(network.at[i, "policy"], network.at[i, "param"]),
)
policy, param = network.at[choice, "policy"], network.at[choice, "param"]

daily = sim_daily(track)
daily = daily[(daily["policy"] == policy) & (daily["param"] == param)]
last_day = daily["date"].max()
latest = daily[daily["date"] == last_day].set_index("machine_id")
recent = daily[daily["date"] > last_day - pd.Timedelta(days=RECENT_DAYS)]
ran_empty = recent.groupby("machine_id")["stockout"].any()
machines = summary.drop_duplicates("machine_id").set_index("machine_id")

rows = []
for machine_id, capacity in capacities(track).items():
    cash = latest.loc[machine_id]
    lowest = min(cash[f"end_cash_{d}_lkr"] / space for d, space in capacity.items())
    if ran_empty[machine_id]:
        status = "Ran empty"
    elif lowest < LOW_CASH_SHARE:
        status = "Low cash"
    else:
        status = "OK"
    location = machines.at[machine_id, "location_type"]
    rows.append(
        {
            "Machine": machine_id,
            "Type": MACHINE_TYPES[machines.at[machine_id, "machine_type"]],
            "Location": location.replace("_", " ").title() if isinstance(location, str) else "—",
            "Cash (LKR M)": cash["end_cash"] / 1e6,
            "Fill": 100 * cash["end_cash"] / sum(capacity.values()),
            "Status": status,
        }
    )
status = pd.DataFrame(rows)

counts = status["Status"].value_counts()
cols = st.columns(4)
cols[0].metric("Machines OK", int(counts.get("OK", 0)))
cols[1].metric("Low cash", int(counts.get("Low cash", 0)))
cols[2].metric(f"Ran empty (last {RECENT_DAYS} days)", int(counts.get("Ran empty", 0)))
cols[3].metric("Cash in machines", ui.lkr_millions(latest["end_cash"].sum()))

st.subheader(f"Machines on {last_day:%d %b %Y}")
show = st.segmented_control("Show", ["All", "Needs attention"], default="All")
if show == "Needs attention":
    status = status[status["Status"] != "OK"]
st.dataframe(
    status.style.map(lambda s: f"color: {STATUS_COLORS[s]}; font-weight: 600", subset=["Status"]),
    hide_index=True,
    column_config={
        "Cash (LKR M)": st.column_config.NumberColumn(format="%.2f"),
        "Fill": st.column_config.ProgressColumn(format="%.0f%%", min_value=0, max_value=100),
    },
)

st.subheader("Van visits on this day")
visits = orders(track)
visits = visits[
    (visits["policy"] == policy) & (visits["param"] == param) & (visits["visit_date"] == last_day)
]
if visits.empty:
    st.caption("No van visits on this day.")
else:
    amounts = visits.filter(like="amount_").filter(like="_lkr").fillna(0)
    st.dataframe(
        pd.DataFrame(
            {
                "Machine": visits["machine_id"],
                "Action": visits["action"].map(ACTIONS),
                "Load (LKR M)": amounts.drop(columns="remove_amount_lkr").sum(axis=1) / 1e6,
                "Remove (LKR M)": visits["remove_amount_lkr"] / 1e6,
                "Emergency": visits["emergency"].map({True: "Yes", False: "No"}),
            }
        ),
        hide_index=True,
        column_config={
            "Load (LKR M)": st.column_config.NumberColumn(format="%.2f"),
            "Remove (LKR M)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
