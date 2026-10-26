"""Dashboard entry point: ``streamlit run dashboard/app.py``."""

import streamlit as st

import ui

ui.setup()
page = st.navigation([st.Page("pages/overview.py", title="Overview", icon=":material/dashboard:")])
ui.track_selector()
page.run()
