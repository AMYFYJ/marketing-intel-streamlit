from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data_sources.competitor_sources import (
    CompetitorQuery,
    analyze_creative_patterns,
    compute_share_of_voice,
    fetch_competitor_intelligence,
    parse_competitors,
)
from data_sources.trend_sources import parse_keywords


@st.cache_data(ttl=900, show_spinner=True)
def _cached_competitor_intelligence(
    competitors: tuple[str, ...],
    keywords: tuple[str, ...],
    country: str,
    max_items: int,
    sources: tuple[str, ...],
    meta_token: str | None,
    meta_version: str,
    youtube_key: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_competitor_intelligence(
        CompetitorQuery(competitors=competitors, keywords=keywords, country=country, max_items_per_source=max_items),
        sources=sources,
        meta_access_token=meta_token,
        meta_api_version=meta_version,
        youtube_api_key=youtube_key,
    )


def render() -> None:
    st.subheader("Competitor Ads + Creative Intelligence")
    st.caption("Monitor competitor creative signals from Meta Ad Library, TikTok Creative Center, YouTube, Reddit, and news sources.")

    with st.form("competitor_controls"):
        c1, c2, c3 = st.columns([2, 2, 1])
        raw_competitors = c1.text_area("Competitors", value="HubSpot, Salesforce, Klaviyo", height=88)
        raw_keywords = c2.text_area("Keywords or themes", value="AI marketing, customer data, automation", height=88)
        country = c3.selectbox("Market", ["US", "GB", "CA", "AU", "DE", "FR"], index=0)
        max_items = c3.slider("Items/source", min_value=5, max_value=50, value=15, step=5)
        sources = st.multiselect(
            "Sources",
            ["Meta Ad Library", "TikTok Creative Center", "YouTube", "Reddit", "GDELT"],
            default=["Meta Ad Library", "TikTok Creative Center", "Reddit", "GDELT"],
        )
        submitted = st.form_submit_button("Refresh competitor intelligence")

    if not submitted:
        st.info("Enter competitors and refresh. Meta API and YouTube are optional; TikTok Creative Center opens live creative-search links.")
        return

    competitors = parse_competitors(raw_competitors)
    keywords = parse_keywords(raw_keywords)
    if not competitors:
        st.warning("Enter at least one competitor.")
        return

    meta_token = _get_secret("META_ACCESS_TOKEN")
    meta_version = _get_secret("META_GRAPH_VERSION") or "v21.0"
    youtube_key = _get_secret("YOUTUBE_API_KEY")
    items, statuses = _cached_competitor_intelligence(competitors, keywords, country, max_items, tuple(sources), meta_token, meta_version, youtube_key)

    _render_status(statuses)
    if items.empty:
        st.warning("No competitor items were returned. Check source status, broaden keywords, or configure optional API keys.")
        return

    sov = compute_share_of_voice(items)
    patterns = analyze_creative_patterns(items)
    _render_charts(sov, patterns)
    _render_items(items)


def _render_status(statuses: pd.DataFrame) -> None:
    with st.expander("Source status and access notes", expanded=True):
        st.dataframe(statuses, use_container_width=True, hide_index=True)


def _render_charts(sov: pd.DataFrame, patterns: pd.DataFrame) -> None:
    st.markdown("#### Competitive Signal Mix")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(sov, x="competitor", y="share_of_voice", color="source", title="Share of voice by source"), use_container_width=True)
    c2.plotly_chart(px.bar(patterns.head(30), x="theme", y="items", color="cta", title="Creative themes and CTA patterns"), use_container_width=True)
    st.markdown("#### Creative Pattern Table")
    st.dataframe(patterns, use_container_width=True, hide_index=True)


def _render_items(items: pd.DataFrame) -> None:
    st.markdown("#### Latest Ads, Links, and Mentions")
    display = items.sort_values("published_at", ascending=False)[["source", "competitor", "keyword", "asset_type", "title", "theme", "cta", "published_at", "url"]].head(300)
    st.dataframe(display, use_container_width=True, hide_index=True)


def _get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None
