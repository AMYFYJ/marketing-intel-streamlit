from __future__ import annotations

import streamlit as st

from features import budget_optimizer, competitor_intelligence, demand_pulse, performance_dashboard
from utils.layout import render_app_header, render_data_sources


st.set_page_config(
    page_title="Marketing Intel Streamlit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    render_app_header()
    tabs = st.tabs(
        [
            "Performance",
            "Competitor Intelligence",
            "Demand Pulse",
            "Budget Optimizer",
            "Data Sources",
        ]
    )

    with tabs[0]:
        performance_dashboard.render()
    with tabs[1]:
        competitor_intelligence.render()
    with tabs[2]:
        demand_pulse.render()
    with tabs[3]:
        budget_optimizer.render()
    with tabs[4]:
        render_data_sources()


if __name__ == "__main__":
    main()
