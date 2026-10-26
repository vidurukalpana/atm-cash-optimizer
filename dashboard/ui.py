"""Shared look of the dashboard: page setup, header, track selector and small components."""

from __future__ import annotations

import streamlit as st

from data import TRACKS, available_tracks

ORANGE = "#F37021"
YELLOW = "#FDB913"
CREAM = "#FFF6E6"
TEXT = "#2E2E2E"
MUTED = "#7A7A7A"

_CSS = f"""
<style>
#MainMenu, footer, [data-testid="stDecoration"] {{ visibility: hidden; }}
.block-container {{ padding-top: 1.6rem; max-width: 1200px; }}
[data-testid="stSidebar"] {{ background: {CREAM}; }}

.app-header {{
    background: linear-gradient(90deg, {YELLOW} 0%, {ORANGE} 100%);
    border-radius: 14px; padding: 1.4rem 1.8rem; margin-bottom: 1.4rem; color: white;
}}
.app-header h1 {{ color: white; font-size: 1.9rem; margin: 0; font-weight: 700; }}
.app-header p {{ color: white; opacity: 0.92; margin: 0.25rem 0 0; font-size: 1rem; }}

.badge {{
    display: inline-block; background: {CREAM}; color: {ORANGE}; border: 1px solid {YELLOW};
    border-radius: 999px; padding: 0.15rem 0.75rem; font-size: 0.8rem; font-weight: 600;
    margin-bottom: 1rem;
}}

[data-testid="stMetric"] {{
    background: white; border: 1px solid #F1E3CC; border-left: 5px solid {ORANGE};
    border-radius: 12px; padding: 0.9rem 1.1rem; box-shadow: 0 2px 8px rgba(243,112,33,0.08);
}}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.85rem; }}

.step {{
    background: white; border: 1px solid #F1E3CC; border-top: 4px solid {YELLOW};
    border-radius: 12px; padding: 1rem 1.1rem; height: 100%;
}}
.step .num {{
    display: inline-flex; align-items: center; justify-content: center; width: 1.9rem;
    height: 1.9rem; border-radius: 50%; background: {ORANGE}; color: white; font-weight: 700;
    margin-bottom: 0.5rem;
}}
.step h4 {{ margin: 0 0 0.2rem; color: {TEXT}; font-size: 1.02rem; }}
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
