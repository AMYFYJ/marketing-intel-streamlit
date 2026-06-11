from __future__ import annotations

import streamlit as st


def render_app_header() -> None:
    st.title("Marketing Intelligence Command Center")
    st.caption("Paid media diagnostics, competitor signals, demand momentum, and goal-based budget planning in one Streamlit app.")


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
        - TikTok Creative Center, LinkedIn Ad Library, and the EU-only X Ads Repository are covered with live deep links because stable public APIs are not available here.

        **Budget optimizer**
        - Goal-driven planner: enter a budget and a goal (conversions, revenue, traffic, awareness, leads, app installs, or follower/engagement growth) to get platform mix, per-platform campaign type and audience, allocation, and expected cost-per-goal and goal volume.
        - Follower growth is not reported by ad APIs, so that goal uses ad engagement (likes/comments/shares/saves) as a documented proxy.
        - Uses a 250k-row synthetic media-mix dataset generated in memory; an Advanced ROAS/CPA optimizer is also available.
        - CSV upload and API connector stubs are included for future real campaign data.
        """
    )
