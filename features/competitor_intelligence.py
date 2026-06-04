from __future__ import annotations

from io import StringIO

import pandas as pd
import plotly.express as px
import streamlit as st

from data_sources.competitor_sources import (
    CompetitorQuery,
    analyze_creative_patterns,
    build_strategy_recommendations,
    build_theme_cta_matrix,
    compute_share_of_voice,
    enrich_competitor_items,
    fetch_competitor_intelligence,
    parse_competitors,
    summarize_competitive_signals,
)
from data_sources.trend_sources import parse_keywords


SOURCES = ["Meta Ad Library", "TikTok Creative Center", "YouTube", "Reddit", "GDELT"]
DEFAULT_SOURCES = ["Meta Ad Library", "TikTok Creative Center", "Reddit", "GDELT"]
PLOTLY_CONFIG = {"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d"]}
COLOR_SEQUENCE = ["#2563eb", "#059669", "#f97316", "#7c3aed", "#dc2626", "#0891b2", "#ca8a04"]


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
    st.subheader("Competitor Creative Intelligence")
    st.caption(
        "Track observed competitor creative, messaging, and launch signals across ad libraries, social discussion, video, and news."
    )

    _inject_css()
    raw_competitors, raw_keywords, country, max_items, sources, submitted = _render_command_bar()
    competitors = parse_competitors(raw_competitors)
    keywords = parse_keywords(raw_keywords)

    if not competitors:
        st.warning("Enter at least one competitor.")
        return

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
        is_preview = False
    else:
        raw_items, statuses = _demo_competitive_snapshot(competitors, keywords, tuple(sources))
        is_preview = True

    items = enrich_competitor_items(raw_items, statuses)
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


def _render_command_bar() -> tuple[str, str, str, int, list[str], bool]:
    with st.form("competitor_command_bar"):
        st.markdown("#### Command Bar")
        c1, c2, c3 = st.columns([1.6, 1.6, 1])
        raw_competitors = c1.text_area("Competitors", value="HubSpot, Salesforce, Klaviyo", height=92)
        raw_keywords = c2.text_area("Keywords or themes", value="AI marketing, customer data, automation", height=92)
        country = c3.selectbox("Market", ["US", "GB", "CA", "AU", "DE", "FR"], index=0)
        max_items = c3.slider("Items/source", min_value=5, max_value=50, value=15, step=5)

        s1, s2 = st.columns([3, 1])
        sources = s1.multiselect("Sources", SOURCES, default=DEFAULT_SOURCES)
        submitted = s2.form_submit_button("Refresh intelligence", type="primary", width="stretch")
    return raw_competitors, raw_keywords, country, int(max_items), sources, bool(submitted)


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
        st.info("Preview snapshot shown. Refresh intelligence to query configured public sources and optional APIs.")


def _render_summary(summary: dict[str, object], is_preview: bool) -> None:
    st.markdown("#### Signal Summary")
    cards = [
        ("Observed signals", f"{summary['items']:,}", "Creative, social, video, and news items"),
        ("Active sources", f"{summary['active_sources']}", f"{summary['source_gaps']} source gaps"),
        ("Share leader", str(summary["sov_leader"]), "Highest observed share of voice"),
        ("Top theme", str(summary["top_theme"]), "Most repeated message territory"),
        ("Top CTA", str(summary["top_cta"]), "Most common conversion prompt"),
        ("Newest signal", str(summary["newest_signal"]), "Most recent observed item"),
        ("Test-next ideas", f"{summary['test_next']}", "High-priority creative candidates"),
        ("Avg strength", f"{summary['avg_signal_strength']:.0f}/100", "Proxy signal score"),
    ]
    columns = st.columns(4)
    for idx, (label, value, note) in enumerate(cards):
        columns[idx % 4].markdown(
            f"""
            <div class="ci-card {'preview' if is_preview else ''}">
                <div class="ci-card-label">{label}</div>
                <div class="ci-card-value">{value}</div>
                <div class="ci-card-note">{note}</div>
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
    sov = compute_share_of_voice(items)
    patterns = analyze_creative_patterns(items)
    matrix = build_theme_cta_matrix(items)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("#### Share of Voice")
        fig = px.bar(
            sov,
            x="competitor",
            y="share_of_voice",
            color="source",
            barmode="group",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"competitor": "", "share_of_voice": "Share of observed signals", "source": "Source"},
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    with c2:
        st.markdown("#### Theme + CTA Mix")
        top_patterns = patterns.head(20).copy()
        fig = px.bar(
            top_patterns,
            x="items",
            y="theme",
            color="cta",
            orientation="h",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"items": "Signals", "theme": "", "cta": "CTA"},
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    c3, c4 = st.columns([1, 1.15])
    with c3:
        st.markdown("#### Creative Pattern Heatmap")
        if matrix.empty:
            st.info("No theme/CTA pattern data is available.")
        else:
            fig = px.imshow(matrix, aspect="auto", color_continuous_scale="Viridis", labels=dict(color="Signals"))
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, config=PLOTLY_CONFIG)

    with c4:
        st.markdown("#### Launch + Mention Timeline")
        timeline = items.sort_values("published_at")
        fig = px.scatter(
            timeline,
            x="published_at",
            y="competitor",
            size="signal_strength",
            color="theme",
            hover_name="title",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"published_at": "", "competitor": "", "signal_strength": "Strength"},
        )
        fig.update_layout(legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)


def _render_creative_decode(items: pd.DataFrame) -> None:
    st.markdown("#### Creative Decode")
    f1, f2, f3, f4 = st.columns(4)
    competitors = f1.multiselect("Competitor", sorted(items["competitor"].dropna().unique()), default=[])
    themes = f2.multiselect("Theme", sorted(items["theme"].dropna().unique()), default=[])
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
    st.dataframe(_format_items(filtered[columns].head(250)), width="stretch", hide_index=True)


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
        "theme",
        "cta",
        "sentiment",
        "engagement",
        "title",
        "url",
    ]
    display = items.sort_values("published_at", ascending=False)[columns].head(500)
    st.dataframe(_format_items(display), width="stretch", hide_index=True)


def _render_empty_state(statuses: pd.DataFrame) -> None:
    st.warning("No competitor signals were returned for the selected query.")
    with st.expander("Source status and next checks", expanded=True):
        st.dataframe(statuses, width="stretch", hide_index=True)
    st.markdown(
        """
        Try broader competitor names or themes, reduce source count when rate limited, or configure optional
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
    selected_competitors = competitors[:3] or ("HubSpot", "Salesforce", "Klaviyo")
    selected_keywords = keywords[:3] or ("AI marketing", "customer data", "automation")
    selected_sources = sources or tuple(DEFAULT_SOURCES)
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    templates = [
        ("AI", "Free trial", "Launch AI workflow automation for your marketing team", "Start free trial for AI workflow automation and save time on campaign tasks."),
        ("Trust", "Book demo", "See how leading teams unify customer data", "Book demo to explore trusted customer data workflows and proof from growing teams."),
        ("Education", "Download", "New guide for lifecycle automation", "Download the guide with tips for segmentation, lifecycle messaging, and campaign efficiency."),
        ("Launch", "Learn more", "Introducing a faster campaign planning workspace", "Learn more about the new launch built for marketers moving from ideas to execution."),
    ]
    for idx, competitor in enumerate(selected_competitors):
        for theme_idx, keyword in enumerate(selected_keywords):
            source = selected_sources[(idx + theme_idx) % len(selected_sources)]
            theme, cta, title, text = templates[(idx + theme_idx) % len(templates)]
            rows.append(
                {
                    "source": source,
                    "competitor": competitor,
                    "keyword": f"{competitor} {keyword}",
                    "asset_type": "Ad" if "Library" in source else "Live search link" if "TikTok" in source else "Mention",
                    "title": f"{competitor}: {title}",
                    "text": text,
                    "url": "https://www.facebook.com/ads/library/" if "Meta" in source else "https://ads.tiktok.com/business/creativecenter/" if "TikTok" in source else "https://www.google.com/search?q=" + competitor.replace(" ", "+"),
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
                "status": "live link" if source == "TikTok Creative Center" else "ok",
                "detail": "Preview snapshot; refresh to query this source.",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(status_rows)


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
        }
        .ci-card.preview {
            border-color: rgba(37, 99, 235, 0.22);
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
