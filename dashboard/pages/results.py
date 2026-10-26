"""Results: what the smart plan saves compared with simple refill rules."""

import plotly.express as px
import streamlit as st

import ui
from atm_cash_optimizer.simulation.compare import best_settings, breakdown, savings
from atm_cash_optimizer.simulation.metrics import network_totals
from data import sim_summary, simulation_info

track = ui.current_track()
summary = sim_summary(track)
network = network_totals(summary)
network["setting"] = [
    ui.setting_name(p, v) for p, v in zip(network["policy"], network["param"], strict=True)
]
best = best_settings(network)
result = savings(best)

ui.header("Results", "Smart cash planning compared with simple refill rules")
ui.simulated_badge(track)
ui.simulation_chips(simulation_info(track), full=True)

with st.expander("What Do These Mean?"):
    st.markdown(
        "- **Smart plan**: forecasts each machine's demand and plans the cheapest van visits.\n"
        "- **Refill rule**: send a van when cash drops below a set level, and fill it up.\n"
        "- **Weekly visit**: a van fills every machine on the same day each week."
    )

empty_gap = 100 * (result["optimizer_stockout_rate"] - result["baseline_stockout_rate"])
cols = st.columns(4)
cols[0].metric("Cost Saving", f"{result['saving_pct']:.1f}%")
cols[1].metric("Smart Plan Cost (LKR)", ui.lkr_millions(result["optimizer_cost"]))
cols[2].metric("Best Rule Cost (LKR)", ui.lkr_millions(result["baseline_cost"]))
cols[3].metric(
    "Days Ran Empty",
    f"{result['optimizer_stockout_rate']:.2%}",
    delta=f"{empty_gap:+.2f} pts vs best rule",
    delta_color="inverse",
)


def ranked_bars(column: str, scale: float, label: str, fmt: str):
    """Horizontal bars, best (smallest) at the top, value written on each bar."""
    df = network.assign(value=network[column] * scale).sort_values("value")
    fig = px.bar(
        df,
        x="value",
        y="setting",
        color="policy",
        orientation="h",
        text="value",
        category_orders={"setting": list(df["setting"])},
    )
    fig = ui.style_figure(fig, height=330)
    fig.update_traces(texttemplate=fmt, textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False)
    # Room on the right for the value written after the longest bar.
    fig.update_xaxes(
        title=label, showgrid=False, showticklabels=False, range=[0, 1.18 * df["value"].max()]
    )
    fig.update_yaxes(title=None)
    return fig


st.subheader("Strategy Comparison")
st.caption("Shorter bars are better.")
left, right = st.columns(2)
left.plotly_chart(ranked_bars("total_cost", 1e-6, "Total Cost (LKR M)", "%{x:.1f}"))
right.plotly_chart(ranked_bars("stockout_rate", 100, "Days Machines Ran Empty (%)", "%{x:.1f}%"))

st.subheader("All Strategies")
table = network.assign(
    total_cost=network["total_cost"] / 1e6,
    avg_idle_cash=network["avg_idle_cash"] / 1e6,
    stockout_rate=100 * network["stockout_rate"],
).sort_values("total_cost")
columns = ["setting", "total_cost", "visits", "emergency_visits", "stockout_rate", "avg_idle_cash"]
st.dataframe(
    table[columns],
    hide_index=True,
    column_config={
        "setting": "Strategy",
        "total_cost": st.column_config.NumberColumn("Total Cost (LKR M)", format="%.2f"),
        "visits": st.column_config.NumberColumn("Van Visits", format="%d"),
        "emergency_visits": st.column_config.NumberColumn("Emergency Visits", format="%d"),
        "stockout_rate": st.column_config.NumberColumn("Ran Empty (% of Days)", format="%.2f"),
        "avg_idle_cash": st.column_config.NumberColumn(
            "Cash Left in Machine (LKR M)", format="%.2f"
        ),
    },
)

segments = {"machine_type": "By Machine Type", "location_type": "By Location"}
parts = {by: breakdown(summary, best, by) for by in segments}
parts = {by: part for by, part in parts.items() if not part.empty}
if parts:
    st.subheader("Best Setting of Each Strategy")
    for tab, (by, part) in zip(st.tabs([segments[b] for b in parts]), parts.items(), strict=True):
        with tab:
            bars = px.bar(
                part.assign(
                    total_cost=part["total_cost"] / 1e6,
                    **{by: part[by].str.replace("_", " ").str.title()},
                ),
                x=by,
                y="total_cost",
                color="policy",
                barmode="group",
            )
            bars = ui.style_figure(bars, height=340)
            bars.update_xaxes(title=None)
            bars.update_yaxes(title="Total Cost (LKR M)")
            st.plotly_chart(bars)
