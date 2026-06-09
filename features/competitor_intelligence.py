from __future__ import annotations

from html import escape
from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st

from data_sources.competitor_sources import (
    CompetitorQuery,
    MARKET_WIDE_COMPETITOR,
    build_strategy_recommendations,
    enrich_competitor_items,
    fetch_competitor_intelligence,
    summarize_competitive_signals,
)


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
COMPETITOR_OPTIONS = (
    "HubSpot",
    "Salesforce",
    "Klaviyo",
    "Sephora",
    "Ulta Beauty",
    "Glossier",
    "L'Oreal",
    "Maybelline",
    "Fenty Beauty",
    "Rare Beauty",
    "The Ordinary",
    "CeraVe",
    "Adobe Experience Cloud",
    "Marketo",
    "Oracle Eloqua",
    "Braze",
    "Iterable",
    "Mailchimp",
    "ActiveCampaign",
    "Constant Contact",
    "Customer.io",
    "Omnisend",
    "Attentive",
    "Shopify",
    "BigCommerce",
    "Wix",
    "Squarespace",
    "Meta Ads",
    "Google Ads",
    "TikTok for Business",
    "Amazon Ads",
    "LinkedIn Marketing Solutions",
    "Pinterest Ads",
    "The Trade Desk",
    "Criteo",
    "Roku Ads",
    "Snap Ads",
    "Reddit Ads",
    "Semrush",
    "Ahrefs",
    "Sprout Social",
    "Hootsuite",
    "Canva",
    "Buffer",
)
THEME_OPTIONS_BY_CATEGORY = {
    "Beauty & Personal Care": (
        "beauty",
        "skincare",
        "makeup",
        "cosmetics",
        "hair care",
        "fragrance",
        "clean beauty",
        "anti-aging",
        "sunscreen",
        "self-care",
        "wellness",
        "men's grooming",
    ),
    "Fashion & Apparel": (
        "fashion",
        "apparel",
        "streetwear",
        "luxury fashion",
        "activewear",
        "footwear",
        "jewelry",
        "handbags",
        "sustainable fashion",
        "back to school",
    ),
    "Health & Fitness": (
        "fitness",
        "workout",
        "nutrition",
        "weight loss",
        "supplements",
        "mental health",
        "sleep",
        "telehealth",
        "personal training",
        "running",
    ),
    "Food & Beverage": (
        "food delivery",
        "restaurants",
        "coffee",
        "snacks",
        "meal kits",
        "grocery",
        "plant-based",
        "protein",
        "hydration",
        "alcohol",
    ),
    "Home & Lifestyle": (
        "home decor",
        "furniture",
        "mattress",
        "home improvement",
        "cleaning",
        "pets",
        "parenting",
        "baby products",
        "gardening",
        "smart home",
    ),
    "Travel & Experiences": (
        "travel",
        "hotels",
        "flights",
        "cruises",
        "theme parks",
        "events",
        "luxury travel",
        "road trips",
        "local experiences",
    ),
    "Finance & Insurance": (
        "credit cards",
        "banking",
        "insurance",
        "investing",
        "personal finance",
        "mortgage",
        "tax software",
        "buy now pay later",
    ),
    "Entertainment & Tech": (
        "streaming",
        "gaming",
        "mobile phones",
        "laptops",
        "apps",
        "subscriptions",
        "consumer electronics",
        "smart watches",
    ),
    "AI & Automation": (
        "AI marketing",
        "AI agents",
        "marketing automation",
        "workflow automation",
        "campaign automation",
        "email automation",
        "sales automation",
        "generative AI",
        "predictive analytics",
        "personalization",
        "chatbots",
        "AI copywriting",
    ),
    "Customer Data & CRM": (
        "customer data",
        "customer data platform",
        "CRM",
        "first-party data",
        "identity resolution",
        "audience segmentation",
        "lead scoring",
        "data enrichment",
        "customer journey",
        "zero-party data",
    ),
    "Lifecycle & Retention": (
        "lifecycle marketing",
        "retention marketing",
        "loyalty",
        "email marketing",
        "SMS marketing",
        "push notifications",
        "abandoned cart",
        "winback campaigns",
        "onboarding",
        "customer engagement",
    ),
    "Ads & Media": (
        "paid social",
        "retail media",
        "performance marketing",
        "programmatic advertising",
        "connected TV",
        "search ads",
        "creator ads",
        "brand awareness",
        "lead generation",
        "conversion campaigns",
        "remarketing",
        "media mix",
    ),
    "Creative & Offers": (
        "free trial",
        "book demo",
        "download guide",
        "case study",
        "webinar",
        "product launch",
        "new feature",
        "limited time offer",
        "discount",
        "pricing",
        "social proof",
        "customer stories",
    ),
    "Measurement & Analytics": (
        "attribution",
        "marketing analytics",
        "incrementality",
        "ROAS",
        "CPA",
        "conversion tracking",
        "media measurement",
        "campaign reporting",
        "A/B testing",
        "experimentation",
        "forecasting",
        "budget optimization",
    ),
    "Trust, Privacy & Compliance": (
        "privacy",
        "security",
        "compliance",
        "GDPR",
        "data governance",
        "brand safety",
        "fraud prevention",
        "consent management",
        "cookieless targeting",
        "trust",
    ),
    "Commerce & Vertical Plays": (
        "ecommerce",
        "marketplaces",
        "B2B marketing",
        "SaaS marketing",
        "retail",
        "financial services",
        "healthcare marketing",
        "travel marketing",
        "local marketing",
        "mobile apps",
    ),
    "Audience Pains": (
        "save time",
        "reduce costs",
        "improve productivity",
        "increase revenue",
        "customer acquisition",
        "reduce churn",
        "data silos",
        "manual reporting",
        "creative fatigue",
        "campaign planning",
        "prove ROI",
    ),
}
DEFAULT_THEMES = ("beauty", "skincare", "makeup")
THEME_LABEL_SEPARATOR = ": "
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
PLOTLY_CONFIG = {"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d"]}
COLOR_SEQUENCE = ["#2563eb", "#059669", "#f97316", "#7c3aed", "#dc2626", "#0891b2", "#ca8a04"]
LIVE_RESULT_SIGNATURE_KEY = "competitor_intelligence_live_signature"
LIVE_RESULT_ITEMS_KEY = "competitor_intelligence_live_items"
LIVE_RESULT_STATUSES_KEY = "competitor_intelligence_live_statuses"


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
    st.subheader("Ad Creative Intelligence")
    st.caption(
        "Analyze public creative, messaging, and launch signals by ad content category, with optional competitor filters."
    )

    _inject_css()
    competitors, keywords, country, max_items, sources, submitted = _render_command_bar()

    if not keywords:
        st.warning("Choose at least one ad content category to analyze.")
        return
    if not sources:
        st.warning("Choose at least one source to scan.")
        return

    query_signature = _query_signature(competitors, keywords, country, max_items, tuple(sources))
    if submitted:
        meta_token = _get_secret("META_ACCESS_TOKEN")
        meta_version = _get_secret("META_GRAPH_VERSION") or "v21.0"
        youtube_key = _get_secret("YOUTUBE_API_KEY")
        raw_items, statuses = _cached_competitor_intelligence(
            competitors,
            keywords,
            country,
            max_items,
            tuple(sources),
            meta_token,
            meta_version,
            youtube_key,
        )
        _store_live_result(query_signature, raw_items, statuses)
        is_preview = False
    elif _has_live_result(query_signature):
        raw_items, statuses = _get_live_result()
        is_preview = False
    else:
        raw_items, statuses = _demo_competitive_snapshot(competitors, keywords, tuple(sources))
        is_preview = True

    items = enrich_competitor_items(raw_items, statuses)
    _render_run_status(is_preview, submitted)
    _render_source_strip(statuses, is_preview)

    if items.empty:
        _render_empty_state(statuses)
        return

    summary = summarize_competitive_signals(items, statuses)
    _render_summary(summary, is_preview)
    _render_proxy_note(is_preview)

    strategy_tab, creative_tab, action_tab, source_tab = st.tabs(
        ["Strategy Radar", "Creative Decode", "Action Board", "Source Detail"]
    )
    with strategy_tab:
        _render_strategy_radar(items)
    with creative_tab:
        _render_creative_decode(items)
    with action_tab:
        _render_action_board(items)
    with source_tab:
        _render_source_detail(items, statuses)


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


def _render_content_category_picker(container: st.delta_generator.DeltaGenerator) -> tuple[str, ...]:
    selected_options = container.multiselect(
        "Ad Content Categories",
        THEME_OPTIONS,
        default=list(DEFAULT_THEME_OPTIONS),
        format_func=_theme_display_label,
        placeholder="Search Categories Like Beauty",
        help=(
            "Options are ordered by category, but only the item name is shown. "
            "Start typing to search across all ad content categories."
        ),
    )
    return _theme_keywords_from_options(selected_options)


def _query_signature(
    competitors: tuple[str, ...],
    keywords: tuple[str, ...],
    country: str,
    max_items: int,
    sources: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], str, int, tuple[str, ...]]:
    return competitors, keywords, country, int(max_items), sources


def _store_live_result(
    query_signature: tuple[tuple[str, ...], tuple[str, ...], str, int, tuple[str, ...]],
    items: pd.DataFrame,
    statuses: pd.DataFrame,
) -> None:
    st.session_state[LIVE_RESULT_SIGNATURE_KEY] = query_signature
    st.session_state[LIVE_RESULT_ITEMS_KEY] = items
    st.session_state[LIVE_RESULT_STATUSES_KEY] = statuses


def _has_live_result(query_signature: tuple[tuple[str, ...], tuple[str, ...], str, int, tuple[str, ...]]) -> bool:
    return (
        st.session_state.get(LIVE_RESULT_SIGNATURE_KEY) == query_signature
        and LIVE_RESULT_ITEMS_KEY in st.session_state
        and LIVE_RESULT_STATUSES_KEY in st.session_state
    )


def _get_live_result() -> tuple[pd.DataFrame, pd.DataFrame]:
    return st.session_state[LIVE_RESULT_ITEMS_KEY], st.session_state[LIVE_RESULT_STATUSES_KEY]


def _render_command_bar() -> tuple[tuple[str, ...], tuple[str, ...], str, int, list[str], bool]:
    with st.container(border=True):
        st.markdown("#### Monitor Setup")
        c1, c2 = st.columns(2)
        selected_competitors = c1.multiselect(
            "Competitors (optional)",
            COMPETITOR_OPTIONS,
            default=list(DEFAULT_COMPETITORS),
            placeholder="Leave blank for market-wide scan",
            help="Optional. Choose brands to narrow the scan, or leave blank to analyze the selected ad content categories across the market.",
        )
        selected_keywords = _render_content_category_picker(c2)

        c3, c4, c5 = st.columns([0.85, 1.55, 1])
        country = c3.selectbox(
            "Market",
            ["US", "GB", "CA", "AU", "DE", "FR"],
            index=0,
            help="Limits market-specific sources, such as Meta Ad Library country targeting, when available.",
        )
        sources = c4.multiselect(
            "Sources",
            SOURCES,
            default=list(DEFAULT_SOURCES),
            placeholder="Choose Sources",
            help=(
                "Choose where to look for public ad and creative signals. "
                "Meta, TikTok, and LinkedIn are ad-library sources; X is an EU-only transparency repository. "
                "YouTube, Reddit, and GDELT add public video, discussion, and news signals."
            ),
        )
        max_items = c5.slider(
            "Items To Scan Per Source",
            min_value=5,
            max_value=50,
            value=DEFAULT_MAX_ITEMS_PER_SOURCE,
            step=5,
            help=(
                "Caps how many items each selected source can return for each ad content category "
                "and optional competitor. Higher values are broader but slower."
            ),
        )

        submitted = st.button("Run Live Analysis", type="primary", width="stretch")
        st.caption(
            "Live analysis queries the selected public sources for the current categories. "
            "After a run, those live results stay visible until you change the setup."
        )
    return tuple(selected_competitors), selected_keywords, country, int(max_items), sources, bool(submitted)


def _render_run_status(is_preview: bool, submitted: bool) -> None:
    if is_preview:
        st.info(
            "Preview mode: this is a sample analysis for the selected setup. "
            "Click **Run Live Analysis** to query the selected sources and update the signal analysis."
        )
    elif submitted:
        st.success("Live analysis complete for the current setup.")
    else:
        st.success("Showing saved live analysis for the current setup.")


def _render_source_strip(statuses: pd.DataFrame, is_preview: bool) -> None:
    st.markdown("#### Source Health")
    source_summary = _source_summary(statuses)
    columns = st.columns(max(len(source_summary), 1))
    for column, row in zip(columns, source_summary):
        status_class = _status_class(row["status"])
        detail = f"{row['ok']} usable · {row['issues']} gaps"
        column.markdown(
            f"""
            <div class="ci-source {status_class}">
                <div class="ci-source-name">{row['source']}</div>
                <div class="ci-source-status">{row['status']}</div>
                <div class="ci-source-detail">{detail}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if is_preview:
        st.caption("Preview source health is illustrative. Run Live Analysis to query configured public sources and optional APIs.")


def _render_summary(summary: dict[str, object], is_preview: bool) -> None:
    st.markdown("#### Signal Summary")
    scope_card = (
        ("Scan scope", "Market-wide", "No competitor filter")
        if str(summary["sov_leader"]) == MARKET_WIDE_COMPETITOR
        else ("Share leader", str(summary["sov_leader"]), "Highest observed share of voice")
    )
    cards = [
        ("Observed signals", f"{summary['items']:,}", "Creative, social, video, and news items"),
        ("Active sources", f"{summary['active_sources']}", f"{summary['source_gaps']} source gaps"),
        scope_card,
        ("Top message", str(summary["top_theme"]), "Most repeated message pattern"),
        ("Top CTA", str(summary["top_cta"]), "Most common conversion prompt"),
        ("Newest signal", str(summary["newest_signal"]), "Most recent observed item"),
        ("Test-next ideas", f"{summary['test_next']}", "High-priority creative candidates"),
        ("Avg strength", f"{summary['avg_signal_strength']:.0f}/100", "Proxy signal score"),
    ]
    columns = st.columns(4)
    for idx, (label, value, note) in enumerate(cards):
        tooltip = SIGNAL_SUMMARY_HELP[label]
        columns[idx % 4].markdown(
            f"""
            <div class="ci-card {'preview' if is_preview else ''}" tabindex="0" aria-label="{escape(label)}: {escape(tooltip)}">
                <div class="ci-card-label">{escape(label)}</div>
                <div class="ci-card-value">{escape(str(value))}</div>
                <div class="ci-card-note">{escape(note)}</div>
                <div class="ci-card-tooltip" role="tooltip">{escape(tooltip)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if idx == 3:
            columns = st.columns(4)


def _render_proxy_note(is_preview: bool) -> None:
    note = "Preview mode uses deterministic sample signals. " if is_preview else ""
    st.caption(
        f"{note}Competitive sources are observed signals, not proof of competitor spend, CTR, CPA, or ROAS."
    )


def _render_strategy_radar(items: pd.DataFrame) -> None:
    c1, c2 = st.columns([1, 1])
    with c1:
        _render_chart_header("Creative Format Mix", "See whether competitors are leaning into video, static, social, news, or unknown ad formats.")
        format_mix = _signal_count(items, ["creative_format", "source"])
        fig = px.bar(
            format_mix,
            x="creative_format",
            y="signals",
            color="source",
            barmode="stack",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"creative_format": "", "signals": "Signals", "source": "Source"},
        )
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    with c2:
        _render_chart_header("Platform x Creative Format Matrix", "Spot which selected sources are contributing each creative format.")
        format_matrix = _pivot_signal_count(items, index="source", columns="creative_format")
        if format_matrix.empty:
            st.info("No creative format data is available.")
        else:
            fig = px.imshow(format_matrix, aspect="auto", color_continuous_scale="Viridis", labels=dict(color="Signals"))
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, config=PLOTLY_CONFIG)

    c3, c4 = st.columns([1, 1])
    with c3:
        _render_chart_header("CTA Mix By Competitor", "Compare who is selling hard, educating, or running without an explicit CTA.")
        cta_mix = _signal_count(items, ["competitor", "cta"])
        fig = px.bar(
            cta_mix,
            x="competitor",
            y="signals",
            color="cta",
            barmode="stack",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"competitor": "", "signals": "Signals", "cta": "CTA"},
        )
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    with c4:
        _render_chart_header("Campaign Type Mix", "Understand whether competitors are running launches, promos, education, proof, lead gen, or brand campaigns.")
        campaign_mix = _signal_count(items, ["competitor", "campaign_type"])
        fig = px.bar(
            campaign_mix,
            x="competitor",
            y="signals",
            color="campaign_type",
            barmode="stack",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"competitor": "", "signals": "Signals", "campaign_type": "Campaign type"},
        )
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    c5, c6 = st.columns([1.15, 1])
    with c5:
        _render_chart_header("Creative Freshness Timeline", "See who has newer signals and whether recent activity clusters around specific campaign types.")
        timeline = items.sort_values("published_at")
        fig = px.scatter(
            timeline,
            x="published_at",
            y="competitor",
            size="signal_strength",
            color="campaign_type",
            symbol="creative_format",
            hover_name="title",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={
                "published_at": "",
                "competitor": "",
                "signal_strength": "Strength",
                "campaign_type": "Campaign type",
                "creative_format": "Creative format",
            },
        )
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    with c6:
        _render_chart_header("Top Repeated Creative Angles", "Identify the concepts competitors repeat most often, such as problem/solution, demo, promo, proof, or launch.")
        angle_mix = _signal_count(items, ["creative_angle", "campaign_type"]).head(14)
        fig = px.bar(
            angle_mix,
            x="signals",
            y="creative_angle",
            color="campaign_type",
            orientation="h",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"creative_angle": "", "signals": "Signals", "campaign_type": "Campaign type"},
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    _render_chart_header("Platform Strategy Summary", "Scan the dominant source, format, CTA, campaign type, and next test idea for each competitor.")
    st.dataframe(_platform_strategy_summary(items), width="stretch", hide_index=True)


def _render_chart_header(title: str, use_it_to: str) -> None:
    st.markdown(f"#### {title}")
    st.caption(f"Use it to: {use_it_to}")


def _signal_count(items: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return (
        items.groupby(columns, dropna=False, as_index=False)
        .agg(signals=("title", "count"), avg_strength=("signal_strength", "mean"))
        .sort_values(["signals", "avg_strength"], ascending=[False, False])
    )


def _pivot_signal_count(items: pd.DataFrame, index: str, columns: str) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame()
    matrix = (
        items.groupby([index, columns], as_index=False)
        .agg(signals=("title", "count"))
        .pivot(index=index, columns=columns, values="signals")
        .fillna(0)
    )
    return matrix


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


def _render_creative_decode(items: pd.DataFrame) -> None:
    st.markdown("#### Creative Decode")
    f1, f2, f3, f4 = st.columns(4)
    competitors = f1.multiselect("Competitor", sorted(items["competitor"].dropna().unique()), default=[])
    themes = f2.multiselect("Message pattern", sorted(items["theme"].dropna().unique()), default=[])
    actions = f3.multiselect("Action", sorted(items["recommended_action"].dropna().unique()), default=[])
    min_strength = f4.slider("Minimum strength", min_value=0, max_value=100, value=35, step=5)

    filtered = items.copy()
    if competitors:
        filtered = filtered[filtered["competitor"].isin(competitors)]
    if themes:
        filtered = filtered[filtered["theme"].isin(themes)]
    if actions:
        filtered = filtered[filtered["recommended_action"].isin(actions)]
    action_order = {"Test next": 0, "Open source": 1, "Watch": 2, "Fix source": 3, "Archive": 4}
    filtered["action_rank"] = filtered["recommended_action"].map(action_order).fillna(5)
    filtered = filtered[filtered["signal_strength"] >= min_strength].sort_values(
        ["action_rank", "signal_strength", "published_at"],
        ascending=[True, False, False],
    )

    if filtered.empty:
        st.info("No creative signals match the current filters.")
        return

    _render_creative_cards(filtered.head(12))

    st.markdown("#### Drilldown Table")
    columns = [
        "source",
        "competitor",
        "creative_format",
        "campaign_type",
        "theme",
        "cta",
        "creative_angle",
        "recommended_action",
        "signal_strength",
        "source_confidence_label",
        "freshness_days",
        "title",
        "url",
    ]
    st.dataframe(_display_items(filtered[columns].head(250)), width="stretch", hide_index=True)


def _render_creative_cards(items: pd.DataFrame) -> None:
    rows = [items.iloc[idx : idx + 3] for idx in range(0, len(items), 3)]
    for row in rows:
        columns = st.columns(3)
        for column, (_, item) in zip(columns, row.iterrows()):
            title = _truncate(item["title"], 105)
            text = _truncate(item["text"], 145)
            column.markdown(
                f"""
                <div class="ci-creative">
                    <div class="ci-pill">{item['recommended_action']} · {item['priority']}</div>
                    <div class="ci-creative-title">{title}</div>
                    <div class="ci-creative-meta">{item['competitor']} · {item['source']} · {_score(item['signal_strength'])}</div>
                    <div class="ci-creative-text">{text}</div>
                    <div class="ci-creative-tags">{item['creative_angle']} · {item['source_confidence_label']}</div>
                    <a class="ci-card-link" href="{item['url']}" target="_blank">Open signal</a>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_action_board(items: pd.DataFrame) -> None:
    st.markdown("#### Action Board")
    recommendations = build_strategy_recommendations(items)
    c1, c2 = st.columns([1, 1.2])
    with c1:
        action_mix = items.groupby("recommended_action", as_index=False).agg(signals=("title", "count"))
        fig = px.bar(
            action_mix,
            x="recommended_action",
            y="signals",
            color="recommended_action",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"recommended_action": "", "signals": "Signals"},
        )
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    with c2:
        competitor_actions = (
            items.groupby(["competitor", "recommended_action"], as_index=False)
            .agg(signals=("title", "count"), avg_strength=("signal_strength", "mean"))
            .sort_values("avg_strength", ascending=False)
        )
        fig = px.scatter(
            competitor_actions,
            x="signals",
            y="avg_strength",
            size="signals",
            color="recommended_action",
            hover_name="competitor",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"signals": "Signals", "avg_strength": "Average strength"},
        )
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    st.markdown("#### Creative Test Briefs")
    display = recommendations.head(60)
    st.dataframe(display, width="stretch", hide_index=True)
    st.download_button(
        "Download creative test briefs",
        data=_to_csv(display),
        file_name="competitor_creative_test_briefs.csv",
        mime="text/csv",
    )


def _render_source_detail(items: pd.DataFrame, statuses: pd.DataFrame) -> None:
    st.markdown("#### Source Access Notes")
    st.dataframe(statuses, width="stretch", hide_index=True)

    st.markdown("#### Raw Signal Feed")
    columns = [
        "published_at",
        "source",
        "competitor",
        "keyword",
        "asset_type",
        "creative_format",
        "campaign_type",
        "theme",
        "cta",
        "sentiment",
        "engagement",
        "title",
        "url",
    ]
    display = items.sort_values("published_at", ascending=False)[columns].head(500)
    st.dataframe(_display_items(display), width="stretch", hide_index=True)


def _render_empty_state(statuses: pd.DataFrame) -> None:
    st.warning("No competitor signals were returned for the selected query.")
    with st.expander("Source status and next checks", expanded=True):
        st.dataframe(statuses, width="stretch", hide_index=True)
    st.markdown(
        """
        Try broader competitors or watch-list items, reduce source count when rate limited, or configure optional
        `META_ACCESS_TOKEN` and `YOUTUBE_API_KEY` secrets for direct API-backed sources.
        """
    )


def _source_summary(statuses: pd.DataFrame) -> list[dict[str, object]]:
    if statuses.empty:
        return [{"source": "Sources", "status": "No status", "ok": 0, "issues": 0}]
    rows = []
    for source, group in statuses.groupby("source", sort=True):
        usable = group["status"].isin(["ok", "live link"])
        issues = int((~usable).sum())
        if usable.all():
            status = "Ready"
        elif usable.any():
            status = "Partial"
        else:
            status = str(group["status"].mode().iloc[0]).title()
        rows.append({"source": source, "status": status, "ok": int(usable.sum()), "issues": issues})
    return rows


def _demo_competitive_snapshot(
    competitors: tuple[str, ...],
    keywords: tuple[str, ...],
    sources: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_competitors = competitors[:3] or (MARKET_WIDE_COMPETITOR,)
    selected_keywords = keywords[:3] or DEFAULT_THEMES
    selected_sources = sources or tuple(DEFAULT_SOURCES)
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    templates = [
        ("Launch", "Shop now", "New {keyword} creative is showing up across paid social", "Shop now messaging highlights fresh {keyword} routines, product benefits, and social proof."),
        ("Trust", "Learn more", "{keyword} ads lean into proof and ingredient claims", "Learn more creative emphasizes trusted reviews, visible results, and simple product education."),
        ("Education", "Download", "Guide-style {keyword} ads explain routines and use cases", "Download-style messaging turns {keyword} product education into a low-friction content offer."),
        ("Discount", "Shop now", "Limited-time {keyword} offers are appearing in market", "Shop now ads promote limited-time {keyword} bundles, discounts, and seasonal offers."),
    ]
    for idx, competitor in enumerate(selected_competitors):
        for theme_idx, keyword in enumerate(selected_keywords):
            source = selected_sources[(idx + theme_idx) % len(selected_sources)]
            theme, cta, title_template, text_template = templates[(idx + theme_idx) % len(templates)]
            signal_prefix = keyword if competitor == MARKET_WIDE_COMPETITOR else competitor
            rows.append(
                {
                    "source": source,
                    "competitor": competitor,
                    "keyword": keyword if competitor == MARKET_WIDE_COMPETITOR else f"{competitor} {keyword}",
                    "asset_type": "Ad" if "Meta" in source else "Live search link" if _source_is_live_link(source) else "Mention",
                    "title": f"{signal_prefix}: {title_template.format(keyword=keyword)}",
                    "text": text_template.format(keyword=keyword),
                    "url": _demo_source_url(source, keyword),
                    "published_at": now - pd.Timedelta(days=(idx * 5 + theme_idx * 3)),
                    "author": competitor,
                    "platforms": source,
                    "engagement": float(8 + idx * 5 + theme_idx * 2),
                    "cta": cta,
                    "theme": theme,
                    "sentiment": 0.2 if theme in {"AI", "Launch"} else 0.0,
                }
            )
    status_rows = []
    for source in selected_sources:
        status_rows.append(
            {
                "source": source,
                "keyword": "preview",
                "status": "live link" if _source_is_live_link(source) else "ok",
                "detail": "Preview snapshot; refresh to query this source.",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(status_rows)


def _source_is_live_link(source: str) -> bool:
    return source in {"TikTok Creative Center", "LinkedIn Ad Library", "X Ads Repository (EU Only)"}


def _demo_source_url(source: str, keyword: str) -> str:
    query = keyword.replace(" ", "+")
    if "Meta" in source:
        return "https://www.facebook.com/ads/library/"
    if "TikTok" in source:
        return "https://ads.tiktok.com/business/creativecenter/"
    if "LinkedIn" in source:
        return "https://www.linkedin.com/ads/library/"
    if "X Ads Repository" in source:
        return "https://ads.twitter.com/ads-repository"
    return "https://www.google.com/search?q=" + query


def _format_items(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "published_at" in df:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "share_of_voice" in df:
        df["share_of_voice"] = df["share_of_voice"].map(lambda value: f"{value:.1%}")
    if "signal_strength" in df:
        df["signal_strength"] = df["signal_strength"].map(_score)
    if "freshness_days" in df:
        df["freshness_days"] = df["freshness_days"].map(lambda value: f"{value:.0f}d")
    if "sentiment" in df:
        df["sentiment"] = df["sentiment"].map(lambda value: f"{value:.2f}")
    return df


def _display_items(frame: pd.DataFrame) -> pd.DataFrame:
    return _format_items(frame).rename(
        columns={
            "cta": "CTA",
            "theme": "Message pattern",
            "creative_format": "Creative format",
            "campaign_type": "Campaign type",
            "creative_angle": "Creative angle",
            "recommended_action": "Recommended action",
            "signal_strength": "Signal strength",
            "source_confidence_label": "Source confidence",
            "freshness_days": "Freshness",
        }
    )


def _to_csv(frame: pd.DataFrame) -> str:
    output = StringIO()
    frame.to_csv(output, index=False)
    return output.getvalue()


def _status_class(status: str) -> str:
    value = status.lower()
    if value == "ready":
        return "ready"
    if value == "partial":
        return "partial"
    return "gap"


def _score(value: float) -> str:
    return f"{float(value):.0f}/100"


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


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .ci-card, .ci-source, .ci-creative {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 8px;
            background: rgba(248, 250, 252, 0.82);
        }
        .ci-card {
            min-height: 128px;
            padding: 0.85rem 0.9rem;
            margin-bottom: 0.8rem;
            position: relative;
            cursor: help;
        }
        .ci-card.preview {
            border-color: rgba(37, 99, 235, 0.22);
        }
        .ci-card:focus {
            outline: 2px solid rgba(37, 99, 235, 0.36);
            outline-offset: 2px;
        }
        .ci-card-label, .ci-source-name {
            color: #475569;
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .ci-card-value {
            color: #0f172a;
            font-size: 1.32rem;
            font-weight: 760;
            line-height: 1.15;
            margin-top: 0.35rem;
            word-break: break-word;
        }
        .ci-card-note, .ci-source-detail, .ci-creative-meta, .ci-creative-text, .ci-creative-tags {
            color: #475569;
            font-size: 0.86rem;
            line-height: 1.35;
        }
        .ci-card-note {
            margin-top: 0.45rem;
        }
        .ci-card-tooltip {
            position: absolute;
            left: 0.9rem;
            right: 0.9rem;
            top: calc(100% - 0.35rem);
            z-index: 20;
            visibility: hidden;
            opacity: 0;
            transform: translateY(0.25rem);
            border: 1px solid rgba(15, 23, 42, 0.12);
            border-radius: 8px;
            background: #0f172a;
            box-shadow: 0 12px 28px rgba(15, 23, 42, 0.18);
            color: #f8fafc;
            font-size: 0.82rem;
            line-height: 1.35;
            padding: 0.65rem 0.7rem;
            transition: opacity 120ms ease, transform 120ms ease, visibility 120ms ease;
        }
        .ci-card:hover .ci-card-tooltip, .ci-card:focus .ci-card-tooltip {
            visibility: visible;
            opacity: 1;
            transform: translateY(0);
        }
        .ci-source {
            min-height: 96px;
            padding: 0.75rem 0.85rem;
        }
        .ci-source.ready {
            border-left: 4px solid #059669;
        }
        .ci-source.partial {
            border-left: 4px solid #f97316;
        }
        .ci-source.gap {
            border-left: 4px solid #dc2626;
        }
        .ci-source-status {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 740;
            margin: 0.22rem 0;
        }
        .ci-creative {
            min-height: 270px;
            padding: 0.95rem;
            margin-bottom: 0.9rem;
        }
        .ci-pill {
            display: inline-block;
            border-radius: 999px;
            background: #e0f2fe;
            color: #075985;
            font-size: 0.78rem;
            font-weight: 720;
            padding: 0.18rem 0.55rem;
            margin-bottom: 0.65rem;
        }
        .ci-creative-title {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 760;
            line-height: 1.25;
            margin-bottom: 0.35rem;
        }
        .ci-creative-text {
            margin-top: 0.65rem;
        }
        .ci-creative-tags {
            margin-top: 0.7rem;
            font-weight: 650;
        }
        .ci-card-link {
            display: inline-block;
            color: #2563eb;
            font-size: 0.9rem;
            font-weight: 700;
            margin-top: 0.75rem;
            text-decoration: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
