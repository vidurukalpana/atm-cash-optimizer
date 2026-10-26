"""Shared look of the dashboard: page setup, header, track selector and small components."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from data import TRACKS, available_tracks

NAVY = "#0B2545"
BLUE = "#1F4E9E"
BRIGHT = "#1F6FEB"
SKY = "#F3F7FD"
LINE = "#DCE6F5"
TEAL = "#0EA5A4"
MUTED = "#64748B"
SLATE = "#94A3B8"

POLICY_COLORS = {"optimizer": BRIGHT, "threshold": TEAL, "fixed_schedule": SLATE}
POLICY_NAMES = {
    "optimizer": "Smart plan",
    "threshold": "Refill rule",
    "fixed_schedule": "Weekly visit",
}

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp label, .stApp input, .stApp button,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}}
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; max-width: 1200px; }}
[data-testid="stSidebar"] {{ background: {SKY}; border-right: 1px solid {LINE}; }}
.stApp h2, .stApp h3 {{ color: {NAVY}; font-weight: 700; letter-spacing: -0.01em; }}

.app-header {{
    background: linear-gradient(120deg, {NAVY} 0%, {BLUE} 100%);
    border-radius: 14px; padding: 1.5rem 1.9rem; margin-bottom: 1.3rem;
    box-shadow: 0 6px 18px rgba(11,37,69,0.18);
}}
.app-header h1 {{
    color: #FFFFFF; font-size: 1.85rem; margin: 0; font-weight: 700; letter-spacing: -0.02em;
}}
.app-header p {{ color: #D6E4FF; margin: 0.3rem 0 0; font-size: 1rem; }}

.badge {{
    display: inline-block; background: #FFF4DB; color: #92400E; border: 1px solid #F6D68A;
    border-radius: 999px; padding: 0.15rem 0.75rem; font-size: 0.75rem; font-weight: 600;
    letter-spacing: 0.04em; margin-bottom: 1rem;
}}

[data-testid="stMetric"] {{
    background: #FFFFFF; border: 1px solid {LINE}; border-left: 5px solid {BRIGHT};
    border-radius: 12px; padding: 0.9rem 1.1rem; box-shadow: 0 2px 10px rgba(31,78,158,0.07);
    height: 8.2rem; box-sizing: border-box;
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.85rem; font-weight: 500; }}
[data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 600; }}

.step {{
    background: #FFFFFF; border: 1px solid {LINE}; border-top: 4px solid {TEAL};
    border-radius: 12px; padding: 1rem 1.1rem; height: 9.5rem; box-sizing: border-box;
}}
.step .num {{
    display: inline-flex; align-items: center; justify-content: center; width: 1.9rem;
    height: 1.9rem; border-radius: 50%; background: {BLUE}; color: #FFFFFF; font-weight: 700;
    margin-bottom: 0.5rem;
}}
.step h4 {{ margin: 0 0 0.2rem; color: {NAVY}; font-size: 1.02rem; font-weight: 600; }}
.step p {{ margin: 0; color: {MUTED}; font-size: 0.88rem; }}
</style>
"""


def setup() -> None:
    """Page config and styles; called once by ``app.py`` on every run."""
    st.set_page_config(page_title="ATM Cash Optimizer", page_icon=":material/atm:", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)


def track_selector() -> str:
    """Sidebar choice of track, kept across pages."""
    tracks = available_tracks()
    if not tracks:
        st.error("No results found. Run the pipeline first (see README).")
        st.stop()
    st.session_state.setdefault("track", tracks[0])
    st.sidebar.selectbox("Network", tracks, format_func=TRACKS.get, key="track")
    return st.session_state["track"]


def current_track() -> str:
    return st.session_state["track"]


def header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="app-header"><h1>{title}</h1>{sub}</div>', unsafe_allow_html=True)


def simulated_badge(track: str) -> None:
    """The lk track is simulated; say so wherever it is shown."""
    if track == "lk":
        st.markdown('<span class="badge">SIMULATED DATA</span>', unsafe_allow_html=True)


def step_card(number: int, title: str, text: str) -> None:
    st.markdown(
        f'<div class="step"><span class="num">{number}</span><h4>{title}</h4><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def setting_name(policy: str, param: float) -> str:
    """Plain display name of a policy setting, e.g. ``Refill below 20%``."""
    if policy == "optimizer":
        return f"Smart plan ({param:.0%} safety)"
    if policy == "threshold":
        return f"Refill below {param:.0%}"
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return f"Weekly visit ({days[int(param)]})"


def lkr_millions(value: float) -> str:
    return f"LKR {value / 1e6:,.2f}M"


def style_figure(fig: go.Figure, height: int = 380) -> go.Figure:
    """Dashboard look for Plotly charts: Inter font, policy colours, light grid, no title."""
    fig.update_layout(
        title=None,
        height=height,
        font={"family": "Inter, -apple-system, sans-serif", "color": NAVY, "size": 13},
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        legend={"orientation": "h", "y": 1.08, "x": 0, "title": None},
    )
    fig.update_xaxes(gridcolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, zeroline=False)
    for trace in fig.data:
        if trace.name in POLICY_COLORS:
            trace.marker.color = POLICY_COLORS[trace.name]
            trace.name = POLICY_NAMES[trace.name]
    return fig
