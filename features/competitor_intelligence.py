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
from utils.formatting import display_labels, title_case_columns

SOURCES = [
    "Meta Ad Library",
    "TikTok Creative Center",
    "LinkedIn Ad Library",
    "X Ads Repository (EU Only)",
    "YouTube",
    "Reddit",
    "GDELT",
]
DEFAULT_SOURCES = ["Meta Ad Library", "TikTok Creative Center", "LinkedIn Ad Library", "Reddit", "GDELT"]
DEFAULT_MAX_ITEMS_PER_SOURCE = 15
DEFAULT_COMPETITORS: tuple[str, ...] = ()
DEFAULT_THEMES = ("beauty", "skincare", "makeup")
THEME_LABEL_SEPARATOR = ": "
THEME_OPTIONS_BY_CATEGORY = {
    "Beauty & Personal Care": (
        "beauty",
        "skincare",
        "makeup",
        "cosmetics",
        "hair care",
    ),
    "AI & Automation": (
        "AI marketing",
        "AI agents",
        "marketing automation",
    ),
    "Ads & Media": (
        "retail media",
        "paid social",
        "connected TV",
    ),
    "Commerce & Vertical Plays": (
        "ecommerce",
        "B2B marketing",
        "SaaS marketing",
    ),
    "Audience Pains": (
        "save time",
        "prove ROI",
        "creative fatigue",
    ),
    "Measurement & Analytics": (
        "attribution",
        "A/B testing",
        "incrementality",
    ),
}
THEME_OPTIONS = tuple(
    f"{category}{THEME_LABEL_SEPARATOR}{theme}"
    for category, themes in THEME_OPTIONS_BY_CATEGORY.items()
    for theme in themes
)
DEFAULT_THEME_OPTIONS = tuple(
    option
    for option in THEME_OPTIONS
    if option.split(THEME_LABEL_SEPARATOR, 1)[1] in DEFAULT_THEMES
)
THEME_DISPLAY_ACRONYMS = {
    "a/b": "A/B",
    "ai": "AI",
    "b2b": "B2B",
    "cpa": "CPA",
    "crm": "CRM",
    "gdpr": "GDPR",
    "roi": "ROI",
    "roas": "ROAS",
    "saas": "SaaS",
    "sms": "SMS",
    "tv": "TV",
}
SIGNAL_SUMMARY_HELP = {
    "Observed signals": "Total number of public items found.",
    "Active sources": "Sources that returned usable results or live links.",
    "Scan scope": "Shows whether the analysis is market-wide or narrowed to selected competitors.",
    "Share leader": "Competitor with the highest share of observed signals.",
    "Top message": "Most repeated detected message pattern, like AI, Trust, Launch, or Education.",
    "Top CTA": "Most common detected call to action, like Book demo, Free trial, or Download.",
    "Newest signal": "Most recent item found.",
    "Test-next ideas": "Signals strong enough to become creative test candidates.",
    "Avg strength": "Proxy score from 0-100 based on source confidence, freshness, CTA clarity, message match, sentiment, and engagement.",
}


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


# Recognizable advertisers across the verticals the app models; type in the
# selector to filter. Leaving the selector empty runs a keyword market scan.
COMPETITOR_OPTIONS = [
    # Retail and ecommerce
    "Amazon", "Walmart", "Target", "Temu", "Shein", "Etsy", "Wayfair", "IKEA",
    # Apparel and fitness
    "Nike", "Adidas", "Lululemon", "Gymshark", "On Running", "Alo Yoga", "Peloton", "Whoop",
    # Beauty
    "Sephora", "Ulta Beauty", "Glossier", "e.l.f. Beauty", "Fenty Beauty", "Rare Beauty",
    # Finance and fintech
    "American Express", "Capital One", "Chime", "Robinhood", "Coinbase", "Klarna", "SoFi",
    # Travel and hospitality
    "Booking.com", "Expedia", "Airbnb", "Hopper", "Kayak", "Marriott",
    # Food and delivery
    "McDonald's", "Chipotle", "DoorDash", "Uber Eats", "HelloFresh", "Liquid Death",
    # Education and entertainment
    "Duolingo", "Coursera", "MasterClass", "Netflix", "Spotify", "Roblox",
    # D2C and telecom
    "Casper", "Warby Parker", "Hims", "Mint Mobile",
    # SaaS and martech
    "HubSpot", "Salesforce", "Klaviyo", "Mailchimp", "Braze", "Adobe", "Shopify", "Canva", "Notion", "Intercom",
]
# Verticals and categories ad agencies commonly plan against.
KEYWORD_OPTIONS = [
    "Beauty",
    "Clothing",
    "Consumer Products",
    "Education",
    "Finance",
    "Fitness and Wellness",
    "Food and Beverage",
    "Gaming",
    "Home and Furniture",
    "Live Event Tickets",
    "Luxury",
    "Music",
    "Pets",
    "Retail",
    "SaaS",
    "Service Subscriptions",
    "Sports",
    "Streaming Services",
    "Toys",
    "Travel",
]


def _theme_keywords_from_options(options: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    keywords = []
    seen = set()
    for option in options:
        keyword = _theme_keyword_from_option(option)
        if keyword and keyword.lower() not in seen:
            keywords.append(keyword)
            seen.add(keyword.lower())
    return tuple(keywords)


def _theme_keyword_from_option(option: str) -> str:
    return str(option).split(THEME_LABEL_SEPARATOR, 1)[-1].strip()


def _theme_display_label(option: str) -> str:
    keyword = _theme_keyword_from_option(option)
    return " ".join(_title_case_theme_word(word) for word in keyword.split())


def _title_case_theme_word(word: str) -> str:
    return "-".join(_title_case_theme_part(part) for part in word.split("-"))


def _title_case_theme_part(part: str) -> str:
    if not part:
        return part
    acronym = THEME_DISPLAY_ACRONYMS.get(part.lower())
    if acronym:
        return acronym
    return part[:1].upper() + part[1:].lower()


def _query_signature(
    competitors: tuple[str, ...],
    keywords: tuple[str, ...],
    country: str,
    max_items: int,
    sources: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], str, int, tuple[str, ...]]:
    return competitors, keywords, country, int(max_items), sources


def render() -> None:
    st.subheader("Competitor Ads + Creative Intelligence")
    st.caption("Monitor competitor creative signals from Meta Ad Library, TikTok Creative Center, YouTube, Reddit, and news sources.")

    with st.form("competitor_controls"):
        c1, c2, c3 = st.columns([2, 2, 1])
        selected_competitors = c1.multiselect(
            "Competitors",
            COMPETITOR_OPTIONS,
            default=["Nike", "Sephora", "Duolingo"],
            help="Brands to search for across the selected sources. Leave empty to run a market scan on the keywords alone.",
        )
        selected_keywords = c2.multiselect(
            "Keywords or Themes",
            KEYWORD_OPTIONS,
            default=["Sports", "Beauty", "Education"],
            help="Industry verticals to scan. Combined with each competitor, or searched alone in a market scan.",
        )
        country = c3.selectbox(
            "Market",
            ["US", "GB", "CA", "AU", "DE", "FR"],
            index=0,
            help="Meta's Ad Library API returns commercial ads only for EU markets (DE, FR); elsewhere it covers political/issue ads only.",
        )
        max_items = c3.slider("Items/Source", min_value=5, max_value=50, value=15, step=5)
        sources = st.multiselect(
            "Sources",
            ["Meta Ad Library", "TikTok Creative Center", "YouTube", "Reddit", "GDELT"],
            default=["Meta Ad Library", "TikTok Creative Center", "Reddit", "GDELT"],
        )
        submitted = st.form_submit_button("Refresh Competitor Intelligence")

    if not submitted:
        st.info(
            "Select competitors and refresh, or leave competitors empty for a keyword-only market scan. "
            "Meta API and YouTube are optional; TikTok Creative Center opens live creative-search links."
        )
        return

    competitors = parse_competitors(selected_competitors)
    keywords = parse_keywords(selected_keywords)
    if not competitors and not keywords:
        st.warning("Select at least one competitor or keyword.")
        return
    if not competitors:
        st.info("No competitors selected — running a market scan on the keywords alone.")

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
    with st.expander("Source Status and Access Notes", expanded=expanded):
        st.dataframe(title_case_columns(statuses), use_container_width=True, hide_index=True)


def _render_charts(sov: pd.DataFrame, patterns: pd.DataFrame) -> None:
    st.markdown("#### Competitive Signal Mix")
    c1, c2 = st.columns(2)

    sov_fig = px.bar(
        sov,
        x="competitor",
        y="share_of_voice",
        color="source",
        barmode="group",
        title="Share of Voice by Source",
        hover_data={"items": True, "share_of_voice": ":.1%"},
        labels=display_labels(["competitor", "share_of_voice", "source", "items"]),
    )
    sov_fig.update_layout(yaxis_tickformat=".0%")
    c1.plotly_chart(sov_fig, use_container_width=True)

    patterns_fig = px.bar(
        patterns.head(30),
        x="theme",
        y="items",
        color="cta",
        title="Creative Themes and CTA Patterns",
        hover_data={"competitor": True},
        labels=display_labels(["theme", "items", "cta", "competitor"]),
    )
    patterns_fig.update_xaxes(categoryorder="total descending")
    c2.plotly_chart(patterns_fig, use_container_width=True)

    st.markdown("#### Creative Pattern Table")
    st.dataframe(
        title_case_columns(patterns),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Items": st.column_config.NumberColumn("Items", format="%d"),
            "Avg Sentiment": st.column_config.NumberColumn("Avg Sentiment", format="%.2f", help="VADER compound score from -1 (negative) to +1 (positive)."),
        },
    )


def _render_items(items: pd.DataFrame) -> None:
    if items.empty:
        return
    st.markdown("#### Latest Ads, Links, and Mentions")
    st.caption("Live-search link rows open platform search results directly; they are not counted in the charts above.")
    display = items.sort_values("published_at", ascending=False)[["source", "competitor", "keyword", "asset_type", "title", "theme", "cta", "published_at", "url"]].head(300)
    st.dataframe(
        title_case_columns(display),
        use_container_width=True,
        hide_index=True,
        column_config={
            "URL": st.column_config.LinkColumn("URL", display_text="Open"),
            "Published At": st.column_config.DatetimeColumn("Published At", format="YYYY-MM-DD HH:mm"),
        },
    )


def _platform_strategy_summary(items: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Competitor",
        "Top Source",
        "Top Format",
        "Top CTA",
        "Top Campaign Type",
        "Newest Signal",
        "Suggested Test",
    ]
    if items.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for competitor, group in items.groupby("competitor", sort=True):
        newest = group.sort_values("published_at", ascending=False).iloc[0]
        strongest = group.sort_values("signal_strength", ascending=False).iloc[0]
        rows.append(
            {
                "Competitor": competitor,
                "Top Source": _mode_label(group["source"]),
                "Top Format": _mode_label(group["creative_format"]),
                "Top CTA": _mode_label(group["cta"]),
                "Top Campaign Type": _mode_label(group["campaign_type"]),
                "Newest Signal": _truncate(newest["title"], 90),
                "Suggested Test": _suggested_test(strongest),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _mode_label(series: pd.Series) -> str:
    values = series.dropna().astype(str)
    if values.empty:
        return "No data"
    return str(values.value_counts().idxmax())


def _suggested_test(row: pd.Series) -> str:
    cta = str(row.get("cta", "No explicit CTA"))
    cta_phrase = cta if cta != "No explicit CTA" else "a clearer CTA"
    return f"Test {row.get('creative_format', 'creative')} for {row.get('campaign_type', 'campaign')} with {cta_phrase}."


def _truncate(value: object, length: int) -> str:
    text = " ".join(str(value).split())
    if len(text) <= length:
        return text
    return text[: max(length - 1, 0)].rstrip() + "..."


def _get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None
