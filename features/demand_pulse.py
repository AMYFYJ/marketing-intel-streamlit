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
    summarize_channels,
)
from features.competitor_intelligence import KEYWORD_OPTIONS as VERTICAL_THEMES
from utils.formatting import display_labels, title_case_columns


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
        selected_keywords = c1.multiselect(
            "Industry Verticals",
            VERTICAL_THEMES,
            default=["Retail", "Beauty", "Gaming"],
            help="Each vertical is searched across the selected sources; the dashboards then break results down by marketing channel automatically.",
        )
        lookback_days = c2.slider("Lookback Days", min_value=1, max_value=30, value=7)
        max_items = c3.slider("Items per Source", min_value=5, max_value=50, value=20, step=5)
        sources = st.multiselect(
            "Sources",
            ["GDELT", "Reddit", "YouTube", "Google Trends export", "Pinterest export"],
            default=["GDELT", "Reddit"],
        )
        submitted = st.form_submit_button("Refresh Live Demand Signals")

    keywords = parse_keywords(selected_keywords)
    if not submitted:
        st.info("Pick industry verticals and refresh to fetch cached live demand signals. GDELT and Reddit need no API key; YouTube needs `YOUTUBE_API_KEY` in Streamlit secrets.")
        return
    if not keywords:
        st.warning("Select at least one industry vertical.")
        return

    youtube_key = _get_secret("YOUTUBE_API_KEY")
    items, statuses = _cached_demand_pulse(keywords, lookback_days, max_items, tuple(sources), youtube_key)
    _render_status(statuses, expanded=items.empty)
    if items.empty:
        failed = statuses[statuses["status"] == "failed"] if not statuses.empty else pd.DataFrame()
        if not failed.empty:
            st.error(
                "No demand items were returned because these sources failed: "
                + ", ".join(sorted(failed["source"].unique()))
                + ". See the source status table above for details."
            )
        else:
            st.warning("All sources responded but returned no items. Try broader keywords or a longer lookback.")
        return

    summary = compute_trend_summary(items, lookback_days=lookback_days)
    angles = recommend_campaign_angles(summary, items)
    _render_summary(items, summary, angles)
    _render_items(items)


def _render_status(statuses: pd.DataFrame, expanded: bool = False) -> None:
    with st.expander("Source Status", expanded=expanded):
        st.dataframe(title_case_columns(statuses), use_container_width=True, hide_index=True)


def _render_summary(items: pd.DataFrame, summary: pd.DataFrame, angles: pd.DataFrame) -> None:
    st.markdown("#### Demand Trend")
    daily = (
        items.dropna(subset=["published_at"])
        .assign(day=lambda df: df["published_at"].dt.floor("D"))
        .groupby(["day", "keyword"], as_index=False)
        .agg(mentions=("title", "count"))
    )
    st.plotly_chart(
        px.line(daily, x="day", y="mentions", color="keyword", markers=True, title="Mentions per Day by Keyword", labels=display_labels(["day", "mentions", "keyword"])),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    c1.plotly_chart(
        px.bar(
            summary,
            x="keyword",
            y="velocity",
            color="momentum",
            color_discrete_map={"Accelerating": "#2ca02c", "Steady": "#1f77b4", "Cooling": "#d62728"},
            title="Momentum: Recent Half vs Earlier Half (%)",
            labels=display_labels(["keyword", "velocity", "momentum"]),
        ),
        use_container_width=True,
    )
    c2.plotly_chart(
        px.scatter(
            summary,
            x="mentions",
            y="avg_sentiment",
            size=summary["recent_mentions"].clip(lower=1),
            color="source",
            hover_name="keyword",
            title="Mentions vs Sentiment",
            labels=display_labels(["mentions", "avg_sentiment", "source"]),
        ),
        use_container_width=True,
    )

    _render_channel_breakdown(items)

    st.markdown("#### Recommended Campaign Angles")
    st.dataframe(title_case_columns(angles), use_container_width=True, hide_index=True)


def _render_channel_breakdown(items: pd.DataFrame) -> None:
    st.markdown("#### Marketing Channel Breakdown")
    channels = summarize_channels(items)
    if channels.empty:
        st.info("No items in the current results mention a specific marketing channel (retail media, influencer, connected TV, paid search, ...). Broaden the verticals or lookback to surface channel-level chatter.")
        return
    classified = int(channels["mentions"].sum())
    st.caption(
        f"{classified} of {len(items)} fetched items mention a specific marketing channel; "
        "items are classified automatically from their text."
    )
    c1, c2 = st.columns(2)
    breakdown_fig = px.bar(
        channels,
        x="channel",
        y="mentions",
        color="keyword",
        title="Channel Mentions by Vertical",
        labels=display_labels(["channel", "mentions", "keyword"]),
    )
    breakdown_fig.update_xaxes(categoryorder="total descending")
    c1.plotly_chart(breakdown_fig, use_container_width=True)

    channel_sentiment = channels.groupby("channel", as_index=False).agg(mentions=("mentions", "sum"), avg_sentiment=("avg_sentiment", "mean"))
    c2.plotly_chart(
        px.scatter(
            channel_sentiment,
            x="mentions",
            y="avg_sentiment",
            size="mentions",
            color="channel",
            hover_name="channel",
            title="Channel Volume vs Sentiment",
            labels=display_labels(["mentions", "avg_sentiment", "channel"]),
        ),
        use_container_width=True,
    )


def _render_items(items: pd.DataFrame) -> None:
    st.markdown("#### Latest Signals")
    display = items.sort_values("published_at", ascending=False)[["source", "keyword", "channel", "title", "author", "published_at", "sentiment", "url"]].head(250)
    st.dataframe(
        title_case_columns(display),
        use_container_width=True,
        hide_index=True,
        column_config={
            "URL": st.column_config.LinkColumn("URL", display_text="Open"),
            "Sentiment": st.column_config.NumberColumn("Sentiment", format="%.2f"),
            "Published At": st.column_config.DatetimeColumn("Published At", format="YYYY-MM-DD HH:mm"),
        },
    )


def _get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None
