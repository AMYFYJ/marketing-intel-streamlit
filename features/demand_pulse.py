from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data_sources.trend_sources import (
    TrendQuery,
    compute_trend_summary,
    fetch_demand_pulse,
    parse_keywords,
    recommend_campaign_angles,
)


@st.cache_data(ttl=900, show_spinner=True)
def _cached_demand_pulse(
    keywords: tuple[str, ...],
    lookback_days: int,
    max_items_per_source: int,
    sources: tuple[str, ...],
    youtube_api_key: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_demand_pulse(
        TrendQuery(keywords=keywords, lookback_days=lookback_days, max_items_per_source=max_items_per_source),
        sources=sources,
        youtube_api_key=youtube_api_key,
    )


def render() -> None:
    st.subheader("Demand Pulse")
    st.caption("Track live category and competitor demand signals from public news, social, video, and trend-export sources.")

    with st.form("demand_pulse_controls"):
        c1, c2, c3 = st.columns([2, 1, 1])
        raw_keywords = c1.text_area("Keywords", value="AI marketing, retail media, TikTok ads", height=88)
        lookback_days = c2.slider("Lookback days", min_value=1, max_value=30, value=7)
        max_items = c3.slider("Items per source", min_value=5, max_value=50, value=20, step=5)
        sources = st.multiselect(
            "Sources",
            ["GDELT", "Reddit", "YouTube", "Google Trends export", "Pinterest export"],
            default=["GDELT", "Reddit"],
        )
        submitted = st.form_submit_button("Refresh live demand signals")

    keywords = parse_keywords(raw_keywords)
    if not submitted:
        st.info("Choose keywords and refresh to fetch cached live demand signals. GDELT and Reddit need no API key; YouTube needs `YOUTUBE_API_KEY` in Streamlit secrets.")
        return
    if not keywords:
        st.warning("Enter at least one keyword.")
        return

    youtube_key = _get_secret("YOUTUBE_API_KEY")
    items, statuses = _cached_demand_pulse(keywords, lookback_days, max_items, tuple(sources), youtube_key)
    _render_status(statuses)
    if items.empty:
        st.warning("No live demand items were returned. Try broader keywords or fewer source filters.")
        return

    summary = compute_trend_summary(items)
    angles = recommend_campaign_angles(summary, items)
    _render_summary(summary, angles)
    _render_items(items)


def _render_status(statuses: pd.DataFrame) -> None:
    with st.expander("Source status", expanded=False):
        st.dataframe(statuses, use_container_width=True, hide_index=True)


def _render_summary(summary: pd.DataFrame, angles: pd.DataFrame) -> None:
    st.markdown("#### Trend Velocity")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(summary, x="keyword", y="velocity", color="source", title="Velocity by keyword and source"), use_container_width=True)
    c2.plotly_chart(px.scatter(summary, x="mentions", y="avg_sentiment", size="velocity", color="source", hover_name="keyword", title="Mentions vs sentiment"), use_container_width=True)

    st.markdown("#### Recommended Campaign Angles")
    st.dataframe(angles, use_container_width=True, hide_index=True)


def _render_items(items: pd.DataFrame) -> None:
    st.markdown("#### Latest Signals")
    display = items.sort_values("published_at", ascending=False)[["source", "keyword", "title", "author", "published_at", "sentiment", "url"]].head(250)
    st.dataframe(display, use_container_width=True, hide_index=True)


def _get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None
