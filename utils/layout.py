from __future__ import annotations

import streamlit as st


def render_app_header() -> None:
    st.title("Marketing Intelligence Dashboard")
    st.caption("Paid media performance, competitor intelligence, demand signals, and budget planning in one Streamlit app.")


def render_data_sources() -> None:
    st.subheader("Data Sources")
    st.markdown(
        """
        - Public campaign dataset target: Kaggle Digital Advertising Campaign Performance Dataset.
        - Live trend sources: GDELT, Reddit feeds, YouTube Data API when configured, Meta Ad Library where available, and TikTok Creative Center where accessible.
        - Budget optimizer: 250k-row synthetic media-mix dataset with future connector interfaces.
        """
    )
