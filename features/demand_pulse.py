from __future__ import annotations

from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import streamlit as st

from data_sources.competitor_sources import (
    LIVE_LINK_ASSET_TYPE,
    fetch_meta_ad_library,
    fetch_tiktok_creative_center_link,
)
from data_sources.trend_sources import (
    TrendQuery,
    compute_trend_summary,
    enrich_trend_items,
    fetch_demand_pulse,
    parse_keywords,
    recommend_campaign_angles,
    summarize_channels,
)
from features.competitor_intelligence import KEYWORD_OPTIONS as VERTICAL_THEMES
from utils.formatting import display_labels, title_case_columns

TIME_WINDOW_OPTIONS = {
    "Past 48 hours": 2,
    "Past week": 7,
    "Past 14 days": 14,
    "Past 30 days": 30,
}
DEFAULT_TIME_WINDOW = "Past week"
DEFAULT_MAX_ITEMS_PER_SOURCE = 20
ACTION_ORDER = {"Test now": 0, "Content idea": 1, "Monitor": 2, "Ignore/noisy": 3, "Fix source": 4}
NOISE_ORDER = {"Low": 0, "Medium": 1, "High": 2}
LANGUAGE_USE_CASES = {
    "Pain": "Objection-handling copy",
    "Question": "How-to or FAQ hook",
    "Comparison": "Comparison ad angle",
    "Purchase research": "Offer or landing-page copy",
    "General mention": "Trend headline",
}
DEMAND_BRIEF_HELP = {
    "Active keywords": "Distinct keywords with usable demand signals after parsing and filtering.",
    "Source coverage": "Selected public sources that returned usable results. Gaps usually mean missing API setup, missing exports, rate limits, or request failures.",
    "Rising topic": "Keyword with the strongest average urgency score, weighted toward fresh and specific signals.",
    "Urgency score": "0-100 directional score blending freshness, source confidence, engagement, keyword velocity, sentiment intensity, intent specificity, and noise risk.",
    "Sentiment shift": "Average simple positive-minus-negative language signal across the visible items. Use it as a tone check, not a brand-lift measure.",
    "Audience language": "The strongest phrase or question to reuse carefully in hooks, landing pages, FAQs, or sales enablement.",
    "Next move": "Highest-priority workflow recommendation across the visible signals: test, create content, monitor, ignore noise, or fix a source.",
    "Noise risk": "Count of broad or low-specificity signals that may not be useful enough for campaign planning.",
}
CHART_HELP = {
    "Velocity by Keyword": "Compares demand momentum by keyword and source. Recent items count more than older items.",
    "Sentiment vs Volume": "Shows which keywords have enough signal volume and whether public language is positive, negative, or neutral.",
    "Keyword + Source Heatmap": "Shows where urgency is concentrated by keyword and source so you can spot coverage gaps or strong channels.",
    "Freshness Distribution": "Shows how recent the visible signals are and which actions they map to.",
    "Action Studio": "Filters demand signals into exportable campaign briefs.",
    "Audience Language Workbench": "Ranks reusable customer language so copywriters and campaign owners can turn demand signals into hooks.",
    "Raw Signal Feed": "Most recent enriched demand signals before grouping.",
}


@st.cache_data(ttl=900, show_spinner=True)
def _cached_demand_pulse(
    keywords: tuple[str, ...],
    lookback_days: int,
    max_items_per_source: int,
    sources: tuple[str, ...],
    youtube_api_key: str | None,
    meta_access_token: str | None,
    meta_api_version: str,
    meta_market: str = "DE",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    items, statuses = fetch_demand_pulse(
        TrendQuery(keywords=keywords, lookback_days=lookback_days, max_items_per_source=max_items_per_source),
        sources=sources,
        youtube_api_key=youtube_api_key,
    )
    status_rows = statuses.to_dict("records") if not statuses.empty else []
    ad_frames: list[pd.DataFrame] = []
    link_rows: list[dict[str, str]] = []

    for keyword in keywords:
        if "Meta Ad Library" in sources:
            frame, status = fetch_meta_ad_library(
                keyword,
                competitor=keyword,
                access_token=meta_access_token,
                country=meta_market,
                api_version=meta_api_version,
                max_records=max_items_per_source,
            )
            status_rows.append(status)
            ads = frame[frame["asset_type"] != LIVE_LINK_ASSET_TYPE]
            if not ads.empty:
                ad_frames.append(
                    pd.DataFrame(
                        {
                            "source": "Meta Ad Library",
                            "keyword": keyword,
                            "title": ads["title"],
                            "url": ads["url"],
                            "published_at": ads["published_at"],
                            "snippet": ads["text"],
                            "author": ads["author"],
                            "engagement": 0.0,
                        }
                    )
                )
            for _, link in frame[frame["asset_type"] == LIVE_LINK_ASSET_TYPE].iterrows():
                link_rows.append({"source": "Meta Ad Library", "keyword": keyword, "url": link["url"]})
        if "TikTok Creative Center" in sources:
            frame, status = fetch_tiktok_creative_center_link(keyword, keyword)
            status_rows.append(status)
            link_rows.append({"source": "TikTok Creative Center", "keyword": keyword, "url": frame.loc[0, "url"]})

    if ad_frames:
        meta_items = enrich_trend_items(pd.concat(ad_frames, ignore_index=True), lookback_days)
        items = pd.concat([items, meta_items], ignore_index=True) if not items.empty else meta_items
    return items, pd.DataFrame(status_rows), pd.DataFrame(link_rows)


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
        max_items = c3.slider("Items per Source", min_value=10, max_value=100, value=50, step=10)
        meta_market = c2.selectbox(
            "Meta Ads Market",
            ["DE", "FR", "NL", "ES", "IT", "US", "GB", "CA", "AU"],
            index=0,
            help="Meta's Ad Library API returns commercial ads only for EU markets; outside the EU it covers political/issue ads only. EU markets act as a proxy for global ad volume.",
        )
        sources = st.multiselect(
            "Sources",
            ["GDELT", "Reddit", "Meta Ad Library", "TikTok Creative Center", "YouTube", "Google Trends export", "Pinterest export"],
            default=["GDELT", "Reddit", "Meta Ad Library", "TikTok Creative Center"],
            help="GDELT and Reddit need no keys and feed the charts. Meta Ad Library feeds the charts when META_ACCESS_TOKEN is set. TikTok has no public API and contributes live links only. YouTube needs YOUTUBE_API_KEY; the export sources read CSV files from data/.",
        )
        submitted = st.form_submit_button("Refresh Live Demand Signals")

    keywords = parse_keywords(selected_keywords)
    if not submitted:
        st.info("Pick industry verticals and refresh to fetch cached live demand signals. GDELT and Reddit need no API key; Meta Ad Library and YouTube use Streamlit secrets.")
        return
    if not keywords:
        st.warning("Select at least one industry vertical.")
        return

    youtube_key = _get_secret("YOUTUBE_API_KEY")
    meta_token = _get_secret("META_ACCESS_TOKEN")
    meta_version = _get_secret("META_GRAPH_VERSION") or "v21.0"
    items, statuses, live_links = _cached_demand_pulse(keywords, lookback_days, max_items, tuple(sources), youtube_key, meta_token, meta_version, meta_market)
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
        _render_live_links(live_links)
        return

    summary = compute_trend_summary(items, lookback_days=lookback_days)
    angles = recommend_campaign_angles(summary, items)
    _render_summary(items, summary, angles)
    _render_items(items)
    _render_live_links(live_links)


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


def _render_live_links(live_links: pd.DataFrame) -> None:
    if live_links.empty:
        return
    st.markdown("#### Live Creative Search")
    st.caption("Direct links into Meta Ad Library and TikTok Creative Center for each vertical; they are not counted in the charts above.")
    st.dataframe(
        title_case_columns(live_links.drop_duplicates()),
        use_container_width=True,
        hide_index=True,
        column_config={"URL": st.column_config.LinkColumn("URL", display_text="Open live search")},
    )


def _time_window_days(label: str) -> int:
    return int(TIME_WINDOW_OPTIONS.get(label, TIME_WINDOW_OPTIONS[DEFAULT_TIME_WINDOW]))


def _safe_external_url(value: object) -> str:
    text = str(value).strip()
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return text
    return "#"


def _language_snapshot(items: pd.DataFrame) -> list[tuple[str, str, str]]:
    if items.empty:
        return [
            ("Strongest phrase", "No phrase available", "Refresh or broaden keywords."),
            ("Pain language", "No objection language found", "Watch for pain-heavy signals."),
            ("Question hook", "No question language found", "Look for how-to demand."),
        ]

    ordered = _rank_language_items(items)
    strongest = ordered.iloc[0]
    pain = _first_matching_phrase(ordered, "Pain")
    question = _first_matching_phrase(ordered, "Question")
    return [
        (
            "Strongest phrase",
            str(strongest["audience_language"]),
            f"{strongest['keyword']} - {strongest['recommended_action']}",
        ),
        ("Pain language", pain, "Use for objection-handling copy."),
        ("Question hook", question, "Use for how-to, FAQ, or search-led creative."),
    ]


def _build_language_playbook(items: pd.DataFrame, limit: int = 12) -> pd.DataFrame:
    columns = ["Priority", "Action", "Keyword", "Intent", "Reusable phrase", "Recommended use", "Urgency", "Noise risk", "Source", "URL"]
    if items.empty:
        return pd.DataFrame(columns=columns)

    ranked = _rank_language_items(items)
    ranked = ranked[ranked["audience_language"].astype(str).str.strip().ne("")]
    if ranked.empty:
        return pd.DataFrame(columns=columns)

    selected = ranked.head(limit).copy()
    urgency = pd.to_numeric(selected["urgency_score"], errors="coerce").fillna(0).round().astype(int)
    return pd.DataFrame(
        {
            "Priority": selected["priority"].astype(str),
            "Action": selected["recommended_action"].astype(str),
            "Keyword": selected["keyword"].astype(str),
            "Intent": selected["intent"].astype(str),
            "Reusable phrase": selected["audience_language"].astype(str),
            "Recommended use": selected.apply(_language_use_case, axis=1),
            "Urgency": urgency,
            "Noise risk": selected["noise_risk"].astype(str),
            "Source": selected["source"].astype(str),
            "URL": selected["url"].astype(str),
        }
    )


def _rank_language_items(items: pd.DataFrame) -> pd.DataFrame:
    df = items.copy()
    defaults = {
        "priority": "Low",
        "recommended_action": "Monitor",
        "keyword": "",
        "intent": "General mention",
        "audience_language": "",
        "urgency_score": 0.0,
        "freshness_hours": 168.0,
        "noise_risk": "Medium",
        "source": "",
        "url": "",
    }
    for column, default in defaults.items():
        if column not in df:
            df[column] = default
    df["action_rank"] = df["recommended_action"].map(ACTION_ORDER).fillna(5)
    df["noise_rank"] = df["noise_risk"].map(NOISE_ORDER).fillna(3)
    df["urgency_numeric"] = pd.to_numeric(df["urgency_score"], errors="coerce").fillna(0)
    df["freshness_numeric"] = pd.to_numeric(df["freshness_hours"], errors="coerce").fillna(168)
    return df.sort_values(
        ["action_rank", "noise_rank", "urgency_numeric", "freshness_numeric"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def _first_matching_phrase(items: pd.DataFrame, intent: str) -> str:
    match = items[items["intent"].eq(intent)]
    if match.empty and intent == "Question":
        match = items[items["audience_language"].astype(str).str.contains("?", regex=False, na=False)]
    if match.empty:
        return f"No {intent.lower()} language found"
    return str(match.iloc[0]["audience_language"])


def _language_use_case(row: pd.Series) -> str:
    if row.get("recommended_action") == "Fix source":
        return "Fix source coverage first"
    if row.get("noise_risk") == "High":
        return "Hold until the signal is more specific"
    return LANGUAGE_USE_CASES.get(str(row.get("intent")), "Trend headline")


def _get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None
