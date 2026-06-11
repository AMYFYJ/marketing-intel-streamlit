from __future__ import annotations

from urllib.parse import urlparse

import pandas as pd
import plotly.express as px
import streamlit as st

from data_sources.demand_metrics import (
    baseline_verdict,
    build_action_queue,
    compute_baseline_stats,
    compute_momentum,
    compute_signal_confidence,
    confidence_verdict,
    lifecycle_verdict,
)
from data_sources.trend_sources import (
    TrendQuery,
    build_daily_series,
    empty_timeline_frame,
    fetch_demand_pulse,
    fetch_gdelt_timeline,
    parse_keywords,
)

SOURCES = ["GDELT", "Reddit", "YouTube", "Google Trends export", "Pinterest export"]
DEFAULT_SOURCES = ["GDELT", "Reddit"]
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
    "Action Studio": "Filters demand signals into exportable campaign briefs. Start broad, then tighten by action, intent, noise risk, or urgency.",
    "Exportable Campaign Briefs": "Downloadable action rows summarizing hook, rationale, keyword, intent, and representative source URL.",
    "Audience Language Workbench": "Ranks reusable customer language so copywriters and campaign owners can turn demand signals into hooks.",
    "Signal Phrase Feed": "Raw phrase-level view of audience language, hooks, urgency, noise, and source URLs.",
    "Intent Mix": "Shows whether demand is mostly pain, questions, comparisons, purchase research, or general mentions.",
    "Source Status": "Explains which selected sources worked and which need setup or broader queries.",
    "Raw Signal Feed": "Most recent enriched demand signals before they are grouped into briefs or language playbooks.",
}

_LIFECYCLE_BADGE = {
    "Accelerating": "🚀",
    "Emerging": "🌱",
    "Peaking": "⛰️",
    "Cooling": "❄️",
    "Dormant": "💤",
}


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


@st.cache_data(ttl=900, show_spinner=True)
def _cached_gdelt_timelines(keywords: tuple[str, ...], lookback_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    statuses: list[dict[str, str]] = []
    for keyword in keywords:
        frame, status = fetch_gdelt_timeline(keyword, lookback_days=lookback_days)
        frames.append(frame)
        statuses.append(status)
    non_empty = [frame for frame in frames if not frame.empty]
    timeline = pd.concat(non_empty, ignore_index=True) if non_empty else empty_timeline_frame()
    return timeline, pd.DataFrame(statuses)


def render() -> None:
    st.subheader("Demand Pulse")
    st.caption(
        "Track live category demand and turn it into timing decisions: where each topic sits in its "
        "lifecycle, whether the signal is trustworthy, how it compares to its own baseline, and what to do next."
    )

    with st.form("demand_pulse_controls"):
        c1, c2, c3 = st.columns([2, 1, 1])
        raw_keywords = c1.text_area(
            "Keywords",
            value="AI marketing, retail media, TikTok ads",
            height=88,
            help=(
                "Comma- or newline-separated topics to monitor. Each becomes its own tracked demand "
                "signal across the chosen sources. Use the specific terms your audience would search "
                "or post about — broad terms add noise, niche terms may return too few mentions to trust."
            ),
        )
        lookback_days = c2.slider(
            "Lookback days",
            min_value=1,
            max_value=30,
            value=TIME_WINDOW_OPTIONS[DEFAULT_TIME_WINDOW],
            help=(
                "How many recent days count as the 'current window'. This window is what momentum and "
                "the baseline comparison are measured over — a shorter window reacts faster but is noisier; "
                "a longer window is steadier but slower to flag a new spike."
            ),
        )
        max_items = c3.slider(
            "Items per source",
            min_value=5,
            max_value=50,
            value=DEFAULT_MAX_ITEMS_PER_SOURCE,
            step=5,
            help=(
                "Maximum item-level signals (articles, posts, videos) pulled from each source. Higher "
                "values give a larger sample for sentiment and confidence scoring, but take longer to fetch."
            ),
        )
        sources = st.multiselect(
            "Sources",
            SOURCES,
            default=DEFAULT_SOURCES,
            help=(
                "Which channels to listen to. More independent sources raise signal confidence "
                "(corroboration). GDELT (news) and Reddit (social) need no API key; YouTube needs "
                "`YOUTUBE_API_KEY`; the export options read trend CSVs from the data/ folder."
            ),
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
    timeline, timeline_status = _cached_gdelt_timelines(keywords, lookback_days)

    _render_status(statuses, timeline_status)
    if items.empty and timeline.empty:
        st.warning("No live demand items were returned. Try broader keywords or fewer source filters.")
        return

    # Fall back to a per-day series built from item timestamps when GDELT timelines are unavailable.
    timeline_for_analysis = timeline if not timeline.empty else build_daily_series(items, lookback_days)

    momentum = compute_momentum(timeline_for_analysis)
    baseline = compute_baseline_stats(timeline_for_analysis, lookback_days)
    confidence = compute_signal_confidence(items)
    actions = build_action_queue(momentum, confidence, baseline)
    # Share demand context with the Budget Optimizer's strategic-context panel.
    st.session_state["demand_context"] = {"momentum": momentum, "actions": actions}

    lifecycle_tab, signal_tab, baseline_tab, action_tab, raw_tab = st.tabs(
        ["Lifecycle & Momentum", "Signal Quality", "Baseline & Anomalies", "Action Queue", "Raw Signals"]
    )
    with lifecycle_tab:
        _render_lifecycle(timeline_for_analysis, momentum)
    with signal_tab:
        _render_signal_quality(items, confidence)
    with baseline_tab:
        _render_baseline(timeline_for_analysis, baseline, lookback_days)
    with action_tab:
        _render_action_queue(actions)
    with raw_tab:
        _render_items(items)


def _render_status(statuses: pd.DataFrame, timeline_status: pd.DataFrame) -> None:
    with st.expander("Source status", expanded=False):
        combined = pd.concat([statuses, timeline_status], ignore_index=True) if not timeline_status.empty else statuses
        st.dataframe(combined, use_container_width=True, hide_index=True)


def _render_lifecycle(timeline: pd.DataFrame, momentum: pd.DataFrame) -> None:
    st.subheader(
        "Trend Lifecycle & Momentum",
        help=(
            "**What it shows:** each topic's daily demand-volume curve, its momentum (recent half-window "
            "average vs the prior half), and a lifecycle stage badge.\n\n"
            "**Reads as:** 🌱 Emerging → 🚀 Accelerating → ⛰️ Peaking → ❄️ Cooling → 💤 Dormant. "
            "Momentum is the % change in average daily volume; acceleration tells you if that change is "
            "still speeding up.\n\n"
            "**Answers:** is interest still building, did you miss the peak, and is the window to act still open?"
        ),
    )
    st.caption(
        "Answers: where is each topic in its demand curve — is interest still building, "
        "did you miss the peak, and is the window to act still open?"
    )
    if momentum.empty:
        st.info("No time-series volume available to compute momentum. GDELT may be rate limited — try again shortly.")
        return

    badges = momentum.sort_values("momentum", ascending=False)
    cols = st.columns(min(len(badges), 4) or 1)
    for idx, (_, row) in enumerate(badges.iterrows()):
        badge = _LIFECYCLE_BADGE.get(row["lifecycle"], "")
        cols[idx % len(cols)].metric(
            f"{badge} {row['keyword']}",
            row["lifecycle"],
            f"{row['momentum'] * 100:+.0f}% momentum",
        )

    for _, row in badges.iterrows():
        badge = _LIFECYCLE_BADGE.get(row["lifecycle"], "")
        st.markdown(f"- {badge} **{row['keyword']}** — {lifecycle_verdict(row)}")

    if not timeline.empty:
        st.plotly_chart(
            px.line(timeline.sort_values("date"), x="date", y="volume", color="keyword", title="Daily demand volume by keyword"),
            use_container_width=True,
        )

    display = momentum[["keyword", "lifecycle", "momentum", "acceleration", "recent_avg", "prior_avg", "days"]].copy()
    display["momentum"] = display["momentum"].map(lambda v: f"{v * 100:+.0f}%")
    display["acceleration"] = display["acceleration"].map(lambda v: f"{v * 100:+.0f}%")
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_signal_quality(items: pd.DataFrame, confidence: pd.DataFrame) -> None:
    st.subheader(
        "Signal Quality & Corroboration",
        help=(
            "**What it shows:** a 0–1 confidence score per keyword, built from how many independent "
            "sources corroborate it (1 source ≈ 0.34, 3+ ≈ 1.0) and whether it clears the minimum "
            "mention threshold to rise above noise. The heatmap normalizes mention share within each "
            "source so channels of different sizes are comparable.\n\n"
            "**Reads as:** ✅ trusted (multi-source, enough mentions) vs ⚠️ thin or single-source. "
            "Sentiment is shown with its positive/negative split and sample size, never as a bare average.\n\n"
            "**Answers:** can you trust this signal, or is it a single-source echo / statistical noise?"
        ),
    )
    st.caption(
        "Answers: can you trust each signal — is it corroborated across independent channels "
        "or a single-source echo, and is the sample big enough to mean anything?"
    )
    if confidence.empty:
        st.info("No items available to assess signal quality.")
        return

    for _, row in confidence.iterrows():
        trusted = not bool(row["low_sample"]) and int(row["sources"]) >= 2
        marker = "✅" if trusted else "⚠️"
        st.markdown(f"- {marker} **{row['keyword']}** — {confidence_verdict(row)}")

    st.plotly_chart(
        px.bar(confidence, x="keyword", y="confidence", color="sources", title="Confidence by keyword (color = independent sources)", range_y=[0, 1]),
        use_container_width=True,
    )

    if not items.empty:
        heat = items.groupby(["keyword", "source"], as_index=False).agg(mentions=("title", "count"))
        heat["share"] = heat["mentions"] / heat.groupby("source")["mentions"].transform("sum")
        st.plotly_chart(
            px.density_heatmap(heat, x="source", y="keyword", z="share", title="Source-normalized mention share (each source sums to 1)"),
            use_container_width=True,
        )

    display = confidence[["keyword", "confidence", "sources", "source_list", "mentions", "positive", "negative", "avg_sentiment", "low_sample"]].copy()
    display["avg_sentiment"] = display["avg_sentiment"].map(lambda v: f"{v:+.2f}")
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_baseline(timeline: pd.DataFrame, baseline: pd.DataFrame, lookback_days: int) -> None:
    st.subheader(
        "Demand vs Baseline",
        help=(
            "**What it shows:** the current window's average volume against each keyword's own trailing "
            "baseline (median ± MAD band on the chart). The robust z-score measures how far the current "
            "level sits from that baseline in a way that resists outliers.\n\n"
            "**Reads as:** 📈 Above baseline (z ≥ 3.5, genuinely unusual), ➖ Within normal range, "
            "📉 Below baseline. Judging a spike against the keyword's own history is how you separate a "
            "real demand shift from ordinary seasonal fluctuation.\n\n"
            "**Answers:** is this spike real for this keyword, and does it justify a budget/bid move?"
        ),
    )
    st.caption(
        "Answers: is the current window unusual for this keyword's own history or just normal "
        "fluctuation — and does it justify a budget/bid move?"
    )
    if baseline.empty:
        st.info("No time-series volume available to compute a baseline. GDELT may be rate limited — try again shortly.")
        return

    for _, row in baseline.iterrows():
        marker = "📈" if row["classification"] == "Above baseline (unusual)" else ("📉" if row["classification"] == "Below baseline" else "➖")
        st.markdown(f"- {marker} **{row['keyword']}** — {baseline_verdict(row)}")

    if not timeline.empty:
        keyword = st.selectbox(
            "Keyword",
            sorted(timeline["keyword"].unique()),
            help="Pick which keyword's volume curve and baseline band to plot below.",
        )
        series = timeline[timeline["keyword"] == keyword].sort_values("date")
        stats = baseline[baseline["keyword"] == keyword]
        fig = px.line(series, x="date", y="volume", title=f"Daily volume vs trailing baseline — {keyword}")
        if not stats.empty:
            median = float(stats.iloc[0]["baseline_median"])
            mad = float(stats.iloc[0]["baseline_mad"])
            fig.add_hline(y=median, line_dash="dash", annotation_text="baseline median")
            fig.add_hrect(y0=median - mad, y1=median + mad, line_width=0, fillcolor="LightSalmon", opacity=0.2)
            if len(series) > lookback_days:
                window_start = series.iloc[-lookback_days]["date"]
                fig.add_vrect(x0=window_start, x1=series.iloc[-1]["date"], line_width=0, fillcolor="LightGreen", opacity=0.15, annotation_text="current window")
        st.plotly_chart(fig, use_container_width=True)

    display = baseline[["keyword", "current_avg", "prior_avg", "baseline_median", "robust_z", "vs_baseline_pct", "classification"]].copy()
    display["vs_baseline_pct"] = display["vs_baseline_pct"].map(lambda v: f"{v * 100:+.0f}%")
    display["robust_z"] = display["robust_z"].map(lambda v: f"{v:+.1f}")
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_action_queue(actions: pd.DataFrame) -> None:
    st.subheader(
        "Action Queue",
        help=(
            "**What it shows:** the top demand moves to make this week, ranked by a priority score that "
            "combines momentum, signal confidence, and how far the keyword sits above its baseline. Each "
            "row pairs a concrete paid-media action (increase budget, launch test creative, capture now, "
            "watchlist, deprioritize) with a plain-English rationale citing the numbers behind it.\n\n"
            "**Reads as:** the highlighted banner is the single highest-leverage move; the list is hard-"
            "capped at five so it stays a decision queue rather than another wall of alerts.\n\n"
            "**Answers:** what should you actually do this week, and why?"
        ),
    )
    st.caption(
        "Answers: what should you actually do this week, and what is the single "
        "highest-leverage demand move right now? Capped at five to stay decision-grade."
    )
    if actions.empty:
        st.info("No actions to recommend from the current signals.")
        return

    top = actions.iloc[0]
    top_badge = _LIFECYCLE_BADGE.get(top["lifecycle"], "")
    st.success(f"Highest-leverage move right now: {top_badge} **{top['keyword']} → {top['action']}** — {top['why']}")
    for _, row in actions.iloc[1:].iterrows():
        badge = _LIFECYCLE_BADGE.get(row["lifecycle"], "")
        st.markdown(f"**{badge} {row['keyword']} → {row['action']}**")
        st.caption(f"Priority {row['priority']:.2f} · {row['why']}")
    st.dataframe(actions, use_container_width=True, hide_index=True)


def _render_items(items: pd.DataFrame) -> None:
    st.subheader(
        "Latest Signals",
        help=(
            "**What it shows:** the raw item-level mentions behind every other dashboard — one row per "
            "article, post, or video, with its source, keyword, timestamp, sentiment, and a link.\n\n"
            "**Answers:** what specific content is driving the demand signal, so you can read the actual "
            "stories and verify a trend with your own eyes before acting."
        ),
    )
    if items.empty:
        st.info("No item-level signals available.")
        return
    display = items.sort_values("published_at", ascending=False)[["source", "keyword", "title", "author", "published_at", "sentiment", "url"]].head(250)
    st.dataframe(display, use_container_width=True, hide_index=True)


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
