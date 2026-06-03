from __future__ import annotations

import streamlit as st


def render_app_header() -> None:
    st.title("Marketing Intelligence Dashboard")
    st.caption("Paid media performance, competitor intelligence, demand signals, and budget planning in one Streamlit app.")


def render_data_sources() -> None:
    st.subheader("Data Sources")
    st.markdown(
        """
        **Campaign performance**
        - Preferred dataset: Kaggle Digital Advertising Campaign Performance Dataset.
        - Local CSV names supported: `digital_advertising_campaign_performance.csv`, `digital_ad_campaigns.csv`, `paid_media_campaigns.csv`, or `campaign_performance.csv`.
        - If no CSV is present, the app uses a deterministic fallback sample for deployment demos.

        **Competitor and demand intelligence**
        - GDELT and Reddit provide no-key public trend signals.
        - YouTube Data API is used when `YOUTUBE_API_KEY` is configured.
        - Meta Ad Library API is used when `META_ACCESS_TOKEN` is configured; otherwise the app provides public search links.
        - TikTok Creative Center is covered with live deep links because a stable public API is not available here.

        **Budget optimizer**
        - Uses a 250k-row synthetic media-mix dataset generated in memory.
        - CSV upload and API connector stubs are included for future real campaign data.
        """
    )
