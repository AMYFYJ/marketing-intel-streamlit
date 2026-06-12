from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data_sources.competitor_sources import (
    CompetitorQuery,
    analyze_creative_patterns,
    compute_share_of_voice,
    exclude_live_link_rows,
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


COMPETITOR_OPTIONS = [
    "HubSpot",
    "Salesforce",
    "Klaviyo",
    "Mailchimp",
    "Braze",
    "Adobe Marketo",
    "ActiveCampaign",
    "Iterable",
    "Attentive",
    "Intercom",
    "Zoho",
    "Pipedrive",
]
KEYWORD_OPTIONS = [
    "marketing automation",
    "customer data",
    "email campaigns",
    "artificial intelligence marketing",
    "personalization",
    "customer engagement",
    "lead generation",
    "loyalty program",
    "ecommerce marketing",
    "analytics",
]


def render() -> None:
    st.subheader("Competitor Ads + Creative Intelligence")
    st.caption("Monitor competitor creative signals from Meta Ad Library, TikTok Creative Center, YouTube, Reddit, and news sources.")

    with st.form("competitor_controls"):
        c1, c2, c3 = st.columns([2, 2, 1])
        selected_competitors = c1.multiselect("Competitors", COMPETITOR_OPTIONS, default=["HubSpot", "Salesforce", "Klaviyo"])
        selected_keywords = c2.multiselect("Keywords or themes", KEYWORD_OPTIONS, default=["marketing automation", "customer data", "email campaigns"])
        country = c3.selectbox("Market", ["US", "GB", "CA", "AU", "DE", "FR"], index=0)
        max_items = c3.slider("Items/source", min_value=5, max_value=50, value=15, step=5)
        sources = st.multiselect(
            "Sources",
            ["Meta Ad Library", "TikTok Creative Center", "YouTube", "Reddit", "GDELT"],
            default=["Meta Ad Library", "TikTok Creative Center", "Reddit", "GDELT"],
        )
        submitted = st.form_submit_button("Refresh competitor intelligence")

    if not submitted:
        st.info("Select competitors and refresh. Meta API and YouTube are optional; TikTok Creative Center opens live creative-search links.")
        return

    competitors = parse_competitors(selected_competitors)
    keywords = parse_keywords(selected_keywords)
    if not competitors:
        st.warning("Select at least one competitor.")
        return

    meta_token = _get_secret("META_ACCESS_TOKEN")
    meta_version = _get_secret("META_GRAPH_VERSION") or "v21.0"
    youtube_key = _get_secret("YOUTUBE_API_KEY")
    items, statuses = _cached_competitor_intelligence(competitors, keywords, country, max_items, tuple(sources), meta_token, meta_version, youtube_key)

    # Live-search link rows stay visible in the items table (as on main) but
    # are excluded from share-of-voice and creative-pattern analytics.
    real_items = exclude_live_link_rows(items)

    _render_status(statuses, expanded=real_items.empty)
    if real_items.empty:
        failed = statuses[statuses["status"] == "failed"] if not statuses.empty else pd.DataFrame()
        if not failed.empty:
            st.error(
                "No competitor items were returned because these sources failed: "
                + ", ".join(sorted(failed["source"].unique()))
                + ". See the source status table above for details."
            )
        else:
            st.warning("No competitor ads or mentions were returned. Broaden keywords, or use the live search links below.")
        _render_items(items)
        return

    sov = compute_share_of_voice(real_items)
    patterns = analyze_creative_patterns(real_items)
    _render_charts(sov, patterns)
    _render_items(items)


def _render_status(statuses: pd.DataFrame, expanded: bool = False) -> None:
    with st.expander("Source status and access notes", expanded=expanded):
        st.dataframe(statuses, use_container_width=True, hide_index=True)


def _render_charts(sov: pd.DataFrame, patterns: pd.DataFrame) -> None:
    st.markdown("#### Competitive Signal Mix")
    c1, c2 = st.columns(2)

    sov_fig = px.bar(
        sov,
        x="competitor",
        y="share_of_voice",
        color="source",
        barmode="group",
        title="Share of voice by source",
        hover_data={"items": True, "share_of_voice": ":.1%"},
    )
    sov_fig.update_layout(yaxis_tickformat=".0%", yaxis_title="share of voice", legend_title_text="source")
    c1.plotly_chart(sov_fig, use_container_width=True)

    patterns_fig = px.bar(
        patterns.head(30),
        x="theme",
        y="items",
        color="cta",
        title="Creative themes and CTA patterns",
        hover_data={"competitor": True},
    )
    patterns_fig.update_xaxes(categoryorder="total descending")
    patterns_fig.update_layout(legend_title_text="CTA")
    c2.plotly_chart(patterns_fig, use_container_width=True)

    st.markdown("#### Creative Pattern Table")
    st.dataframe(
        patterns,
        use_container_width=True,
        hide_index=True,
        column_config={
            "items": st.column_config.NumberColumn("items", format="%d"),
            "avg_sentiment": st.column_config.NumberColumn("avg_sentiment", format="%.2f", help="VADER compound score from -1 (negative) to +1 (positive)."),
        },
    )


def _render_items(items: pd.DataFrame) -> None:
    if items.empty:
        return
    st.markdown("#### Latest Ads, Links, and Mentions")
    st.caption("Live-search link rows open platform search results directly; they are not counted in the charts above.")
    display = items.sort_values("published_at", ascending=False)[["source", "competitor", "keyword", "asset_type", "title", "theme", "cta", "published_at", "url"]].head(300)
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "url": st.column_config.LinkColumn("url", display_text="Open"),
            "published_at": st.column_config.DatetimeColumn("published_at", format="YYYY-MM-DD HH:mm"),
        },
    )


def _get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None
