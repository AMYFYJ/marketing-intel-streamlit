from __future__ import annotations

from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st

from data_sources.trend_sources import (
    TrendQuery,
    build_keyword_source_matrix,
    build_signal_opportunities,
    compute_trend_summary,
    enrich_demand_signals,
    fetch_demand_pulse,
    parse_keywords,
    summarize_demand_brief,
)


SOURCES = ["GDELT", "Reddit", "YouTube", "Google Trends export", "Pinterest export"]
DEFAULT_SOURCES = ["GDELT", "Reddit"]
PLOTLY_CONFIG = {"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d"]}
COLOR_SEQUENCE = ["#2563eb", "#059669", "#f97316", "#7c3aed", "#dc2626", "#0891b2", "#ca8a04"]


@st.cache_data(ttl=900, show_spinner=True)
def _cached_demand_pulse(
    keywords: tuple[str, ...],
    lookback_days: int,
    max_items_per_source: int,
    market: str,
    sources: tuple[str, ...],
    youtube_api_key: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    return fetch_demand_pulse(
        TrendQuery(
            keywords=keywords,
            lookback_days=lookback_days,
            max_items_per_source=max_items_per_source,
            market=market,
        ),
        sources=sources,
        youtube_api_key=youtube_api_key,
    )


def render() -> None:
    st.subheader("Demand Pulse")
    st.caption("Turn noisy public demand signals into campaign hooks, content priorities, and watchlist decisions.")

    _inject_css()
    raw_keywords, market, audience_focus, lookback_days, max_items, sources, submitted = _render_command_bar()
    keywords = parse_keywords(raw_keywords)
    if not keywords:
        st.warning("Enter at least one keyword.")
        return

    if submitted:
        youtube_key = _get_secret("YOUTUBE_API_KEY")
        raw_items, statuses = _cached_demand_pulse(
            keywords,
            lookback_days,
            max_items,
            market,
            tuple(sources),
            youtube_key,
        )
        is_preview = False
    else:
        raw_items, statuses = _demo_demand_snapshot(keywords, tuple(sources), audience_focus)
        is_preview = True

    items = enrich_demand_signals(raw_items, statuses)
    _render_source_strip(statuses, is_preview)

    if items.empty:
        _render_empty_state(statuses)
        return

    summary = compute_trend_summary(items)
    brief = summarize_demand_brief(items, statuses)
    _render_demand_brief(brief, audience_focus, is_preview)
    _render_signal_note(is_preview)

    radar_tab, action_tab, language_tab, source_tab = st.tabs(
        ["Signal Radar", "Action Studio", "Audience Language", "Source Detail"]
    )
    with radar_tab:
        _render_signal_radar(items, summary)
    with action_tab:
        _render_action_studio(items, summary)
    with language_tab:
        _render_audience_language(items)
    with source_tab:
        _render_source_detail(items, statuses)


def _render_command_bar() -> tuple[str, str, str, int, int, list[str], bool]:
    with st.form("demand_command_bar"):
        st.markdown("#### Signal Command Bar")
        c1, c2, c3 = st.columns([1.6, 1, 1])
        raw_keywords = c1.text_area("Keywords", value="AI marketing, retail media, TikTok ads", height=92)
        market = c2.selectbox("Market", ["US", "GB", "CA", "AU", "DE", "FR"], index=0)
        audience_focus = c2.selectbox(
            "Audience focus",
            ["Growth marketers", "Media buyers", "Ecommerce teams", "SaaS operators", "General market"],
            index=0,
        )
        lookback_days = c3.slider("Lookback days", min_value=1, max_value=30, value=7)
        max_items = c3.slider("Items/source", min_value=5, max_value=50, value=20, step=5)
        s1, s2 = st.columns([3, 1])
        sources = s1.multiselect("Sources", SOURCES, default=DEFAULT_SOURCES)
        submitted = s2.form_submit_button("Refresh demand signals", type="primary", width="stretch")
    return raw_keywords, market, audience_focus, int(lookback_days), int(max_items), sources, bool(submitted)


def _render_source_strip(statuses: pd.DataFrame, is_preview: bool) -> None:
    st.markdown("#### Source Health")
    rows = _source_summary(statuses)
    columns = st.columns(max(len(rows), 1))
    for column, row in zip(columns, rows):
        column.markdown(
            f"""
            <div class="dp-source {_status_class(row['status'])}">
                <div class="dp-source-name">{row['source']}</div>
                <div class="dp-source-status">{row['status']}</div>
                <div class="dp-source-detail">{row['ok']} usable - {row['issues']} gaps</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if is_preview:
        st.info("Preview snapshot shown. Refresh demand signals to query configured public sources and optional exports.")


def _render_demand_brief(brief: dict[str, object], audience_focus: str, is_preview: bool) -> None:
    st.markdown("#### Demand Brief")
    cards = [
        ("Active keywords", f"{brief['active_keywords']}", f"Audience focus: {audience_focus}"),
        ("Source coverage", str(brief["source_coverage"]), f"{brief['source_gaps']} source gaps"),
        ("Rising topic", str(brief["rising_topic"]), "Highest average urgency"),
        ("Urgency score", f"{brief['urgency_score']:.0f}/100", "Directional signal score"),
        ("Sentiment shift", str(brief["sentiment_shift"]), "Mean public tone"),
        ("Audience language", str(brief["audience_language"]), "Top phrase to reuse carefully"),
        ("Next move", str(brief["next_move"]), "Recommended campaign workflow"),
        ("Noise risk", f"{brief['noise_risk']}", f"{brief['test_now']} test-now ideas"),
    ]
    columns = st.columns(4)
    for idx, (label, value, note) in enumerate(cards):
        display_value = _truncate(value, 76) if label == "Audience language" else value
        columns[idx % 4].markdown(
            f"""
            <div class="dp-card {'preview' if is_preview else ''}">
                <div class="dp-card-label">{label}</div>
                <div class="dp-card-value">{display_value}</div>
                <div class="dp-card-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if idx == 3:
            columns = st.columns(4)


def _render_signal_note(is_preview: bool) -> None:
    prefix = "Preview mode uses deterministic sample signals. " if is_preview else ""
    st.caption(
        f"{prefix}Demand Pulse surfaces directional public signals, not exact market size, spend, or conversion forecasts."
    )


def _render_signal_radar(items: pd.DataFrame, summary: pd.DataFrame) -> None:
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("#### Velocity by Keyword")
        fig = px.bar(
            summary,
            x="keyword",
            y="velocity",
            color="source",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"keyword": "", "velocity": "Velocity", "source": "Source"},
        )
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    with c2:
        st.markdown("#### Sentiment vs Volume")
        rollup = _keyword_rollup(items)
        fig = px.scatter(
            rollup,
            x="signals",
            y="avg_sentiment",
            size="avg_urgency",
            color="primary_action",
            hover_name="keyword",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"signals": "Signals", "avg_sentiment": "Sentiment", "avg_urgency": "Urgency"},
        )
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    c3, c4 = st.columns([1, 1])
    with c3:
        st.markdown("#### Keyword + Source Heatmap")
        matrix = build_keyword_source_matrix(items)
        if matrix.empty:
            st.info("No keyword/source matrix is available.")
        else:
            fig = px.imshow(matrix, aspect="auto", color_continuous_scale="Viridis", labels=dict(color="Urgency"))
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, config=PLOTLY_CONFIG)

    with c4:
        st.markdown("#### Freshness Distribution")
        fig = px.histogram(
            items,
            x="freshness_hours",
            color="recommended_action",
            nbins=12,
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"freshness_hours": "Hours since published", "recommended_action": "Action"},
        )
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)


def _render_action_studio(items: pd.DataFrame, summary: pd.DataFrame) -> None:
    st.markdown("#### Action Studio")
    f1, f2, f3, f4 = st.columns(4)
    actions = f1.multiselect("Action", sorted(items["recommended_action"].dropna().unique()), default=[])
    intents = f2.multiselect("Intent", sorted(items["intent"].dropna().unique()), default=[])
    noise = f3.multiselect("Noise risk", sorted(items["noise_risk"].dropna().unique()), default=[])
    min_urgency = f4.slider("Minimum urgency", min_value=0, max_value=100, value=35, step=5)

    filtered = items.copy()
    if actions:
        filtered = filtered[filtered["recommended_action"].isin(actions)]
    if intents:
        filtered = filtered[filtered["intent"].isin(intents)]
    if noise:
        filtered = filtered[filtered["noise_risk"].isin(noise)]
    filtered = filtered[filtered["urgency_score"] >= min_urgency]

    if filtered.empty:
        st.info("No demand signals match the current filters.")
        return

    opportunities = build_signal_opportunities(filtered, summary)
    _render_opportunity_cards(opportunities.head(9))

    st.markdown("#### Exportable Campaign Briefs")
    display = opportunities.head(80)
    st.dataframe(display, width="stretch", hide_index=True)
    st.download_button(
        "Download demand briefs",
        data=_to_csv(display),
        file_name="demand_pulse_campaign_briefs.csv",
        mime="text/csv",
    )


def _render_opportunity_cards(opportunities: pd.DataFrame) -> None:
    rows = [opportunities.iloc[idx : idx + 3] for idx in range(0, len(opportunities), 3)]
    for row in rows:
        columns = st.columns(3)
        for column, (_, item) in zip(columns, row.iterrows()):
            column.markdown(
                f"""
                <div class="dp-opportunity">
                    <div class="dp-pill">{item['recommended_action']} - {item['priority']}</div>
                    <div class="dp-opportunity-title">{_truncate(item['campaign_hook'], 130)}</div>
                    <div class="dp-opportunity-meta">{item['keyword']} - {item['intent']}</div>
                    <div class="dp-opportunity-note">{item['rationale']}</div>
                    <a class="dp-card-link" href="{item['url']}" target="_blank">Open representative signal</a>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_audience_language(items: pd.DataFrame) -> None:
    st.markdown("#### Audience Language")
    language = (
        items.sort_values(["urgency_score", "freshness_hours"], ascending=[False, True])
        [["keyword", "intent", "recommended_action", "urgency_score", "noise_risk", "audience_language", "campaign_hook", "url"]]
        .head(250)
    )
    st.dataframe(_format_items(language), width="stretch", hide_index=True)

    st.markdown("#### Intent Mix")
    c1, c2 = st.columns(2)
    intent_mix = items.groupby("intent", as_index=False).agg(signals=("title", "count"), avg_urgency=("urgency_score", "mean"))
    c1.plotly_chart(
        px.bar(intent_mix, x="intent", y="signals", color="intent", color_discrete_sequence=COLOR_SEQUENCE),
        config=PLOTLY_CONFIG,
    )
    action_mix = items.groupby("recommended_action", as_index=False).agg(signals=("title", "count"), avg_urgency=("urgency_score", "mean"))
    c2.plotly_chart(
        px.bar(action_mix, x="recommended_action", y="signals", color="recommended_action", color_discrete_sequence=COLOR_SEQUENCE),
        config=PLOTLY_CONFIG,
    )


def _render_source_detail(items: pd.DataFrame, statuses: pd.DataFrame) -> None:
    st.markdown("#### Source Status")
    st.dataframe(statuses, width="stretch", hide_index=True)

    st.markdown("#### Raw Signal Feed")
    columns = [
        "published_at",
        "source",
        "keyword",
        "intent",
        "recommended_action",
        "urgency_score",
        "noise_risk",
        "sentiment",
        "title",
        "author",
        "url",
    ]
    display = items.sort_values(["published_at", "urgency_score"], ascending=[False, False])[columns].head(500)
    st.dataframe(_format_items(display), width="stretch", hide_index=True)


def _render_empty_state(statuses: pd.DataFrame) -> None:
    st.warning("No demand signals were returned for the selected query.")
    with st.expander("Source status and next checks", expanded=True):
        st.dataframe(statuses, width="stretch", hide_index=True)
    st.markdown(
        """
        Try broader or more specific keyword variants, reduce source count if rate limited, or add optional
        `YOUTUBE_API_KEY`, `data/google_trends_export.csv`, or `data/pinterest_trends_export.csv` inputs.
        """
    )


def _keyword_rollup(items: pd.DataFrame) -> pd.DataFrame:
    action_order = {"Test now": 0, "Content idea": 1, "Monitor": 2, "Ignore/noisy": 3, "Fix source": 4}
    rows = []
    for keyword, group in items.groupby("keyword"):
        action = (
            group["recommended_action"]
            .value_counts()
            .rename_axis("action")
            .reset_index(name="count")
            .assign(order=lambda frame: frame["action"].map(action_order).fillna(5))
            .sort_values(["order", "count"], ascending=[True, False])
            .iloc[0]["action"]
        )
        rows.append(
            {
                "keyword": keyword,
                "signals": int(len(group)),
                "avg_sentiment": float(group["sentiment"].mean()),
                "avg_urgency": float(group["urgency_score"].mean()),
                "primary_action": action,
            }
        )
    return pd.DataFrame(rows)


def _source_summary(statuses: pd.DataFrame) -> list[dict[str, object]]:
    if statuses.empty:
        return [{"source": "Sources", "status": "No status", "ok": 0, "issues": 0}]
    rows = []
    for source, group in statuses.groupby("source", sort=True):
        usable = group["status"].eq("ok")
        issues = int((~usable).sum())
        if usable.all():
            status = "Ready"
        elif usable.any():
            status = "Partial"
        else:
            status = str(group["status"].mode().iloc[0]).title()
        rows.append({"source": source, "status": status, "ok": int(usable.sum()), "issues": issues})
    return rows


def _demo_demand_snapshot(
    keywords: tuple[str, ...],
    sources: tuple[str, ...],
    audience_focus: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_keywords = keywords[:4] or ("AI marketing", "retail media", "TikTok ads")
    selected_sources = sources or tuple(DEFAULT_SOURCES)
    now = pd.Timestamp.now(tz="UTC")
    templates = [
        ("How should teams use AI marketing without creating expensive workflow problems?", "Question"),
        ("Retail media growth creates a strong opportunity for better measurement", "Purchase research"),
        ("TikTok ads versus short-form video alternatives for ecommerce teams", "Comparison"),
        ("Marketers complain that campaign reporting is slow and hard to trust", "Pain"),
    ]
    rows = []
    for idx, keyword in enumerate(selected_keywords):
        for offset in range(2):
            source = selected_sources[(idx + offset) % len(selected_sources)]
            title, _intent = templates[(idx + offset) % len(templates)]
            rows.append(
                {
                    "source": source,
                    "keyword": keyword,
                    "title": f"{keyword}: {title}",
                    "url": "https://www.google.com/search?q=" + keyword.replace(" ", "+"),
                    "published_at": now - pd.Timedelta(hours=idx * 12 + offset * 7),
                    "snippet": f"{audience_focus} are discussing {keyword} with new questions, risks, and campaign ideas.",
                    "author": source,
                    "engagement": float(5 + idx * 4 + offset * 3),
                    "sentiment": 0.35 if offset == 0 else -0.25 if idx == 3 else 0.0,
                    "recency_hours": float(idx * 12 + offset * 7),
                }
            )
    status_rows = [
        {
            "source": source,
            "keyword": "preview",
            "status": "ok",
            "detail": "Preview snapshot; refresh to query this source.",
        }
        for source in selected_sources
    ]
    return pd.DataFrame(rows, columns=empty_demo_columns()), pd.DataFrame(status_rows)


def empty_demo_columns() -> list[str]:
    return ["source", "keyword", "title", "url", "published_at", "snippet", "author", "engagement", "sentiment", "recency_hours"]


def _format_items(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "published_at" in df:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M")
    if "urgency_score" in df:
        df["urgency_score"] = df["urgency_score"].map(lambda value: f"{float(value):.0f}/100")
    if "sentiment" in df:
        df["sentiment"] = df["sentiment"].map(lambda value: f"{float(value):+.2f}")
    return df


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
        .dp-card, .dp-source, .dp-opportunity {
            border: 1px solid rgba(148, 163, 184, 0.28);
            border-radius: 8px;
            background: rgba(248, 250, 252, 0.82);
        }
        .dp-card {
            min-height: 128px;
            padding: 0.85rem 0.9rem;
            margin-bottom: 0.8rem;
        }
        .dp-card.preview {
            border-color: rgba(37, 99, 235, 0.22);
        }
        .dp-card-label, .dp-source-name {
            color: #475569;
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .dp-card-value {
            color: #0f172a;
            font-size: 1.28rem;
            font-weight: 760;
            line-height: 1.15;
            margin-top: 0.35rem;
            word-break: break-word;
        }
        .dp-card-note, .dp-source-detail, .dp-opportunity-meta, .dp-opportunity-note {
            color: #475569;
            font-size: 0.86rem;
            line-height: 1.35;
        }
        .dp-card-note {
            margin-top: 0.45rem;
        }
        .dp-source {
            min-height: 96px;
            padding: 0.75rem 0.85rem;
        }
        .dp-source.ready {
            border-left: 4px solid #059669;
        }
        .dp-source.partial {
            border-left: 4px solid #f97316;
        }
        .dp-source.gap {
            border-left: 4px solid #dc2626;
        }
        .dp-source-status {
            color: #0f172a;
            font-size: 1.05rem;
            font-weight: 740;
            margin: 0.22rem 0;
        }
        .dp-opportunity {
            min-height: 238px;
            padding: 0.95rem;
            margin-bottom: 0.9rem;
        }
        .dp-pill {
            display: inline-block;
            border-radius: 999px;
            background: #dcfce7;
            color: #166534;
            font-size: 0.78rem;
            font-weight: 720;
            padding: 0.18rem 0.55rem;
            margin-bottom: 0.65rem;
        }
        .dp-opportunity-title {
            color: #0f172a;
            font-size: 1.02rem;
            font-weight: 760;
            line-height: 1.25;
            margin-bottom: 0.4rem;
        }
        .dp-opportunity-note {
            margin-top: 0.65rem;
        }
        .dp-card-link {
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
