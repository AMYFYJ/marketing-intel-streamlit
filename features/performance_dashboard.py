from __future__ import annotations

from dataclasses import dataclass
from io import StringIO

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_sources.campaign_data import (
    CampaignFilters,
    KAGGLE_DATASET_URL,
    add_recommendations,
    detect_anomalies,
    filter_campaigns,
    load_campaign_data,
    summarize_metrics,
    to_tuple,
    unique_sorted,
)


DIMENSION_OPTIONS = {
    "Platform": "platform",
    "Objective": "objective",
    "Industry": "industry",
    "Audience": "audience_segment",
    "Device": "device",
    "Creative format": "creative_format",
    "Placement": "placement",
    "Geo": "geo",
    "Budget tier": "budget_tier",
}

METRIC_OPTIONS = {
    "Spend": "spend",
    "Revenue": "revenue",
    "Profit": "profit",
    "ROAS": "roas",
    "CPA": "cpa",
    "CTR": "ctr",
    "CVR": "cvr",
    "Conversions": "conversions",
    "Clicks": "clicks",
    "Impressions": "impressions",
    "Frequency": "frequency",
}

AGGREGATE_COLUMNS = [
    "spend",
    "revenue",
    "profit",
    "impressions",
    "reach",
    "clicks",
    "landing_page_view",
    "add_to_cart",
    "conversions",
    "video_views",
]

PLOTLY_CONFIG = {
    "displayModeBar": True,
    "modeBarButtonsToRemove": ["lasso2d"],
}

COLOR_SEQUENCE = [
    "#2563eb",
    "#059669",
    "#f97316",
    "#7c3aed",
    "#dc2626",
    "#0891b2",
    "#ca8a04",
    "#db2777",
]


@dataclass(frozen=True)
class DashboardControls:
    filters: CampaignFilters
    primary_metric: str
    primary_metric_label: str
    trend_dimension: str
    segment_dimension: str
    heatmap_dimension: str
    granularity: str
    min_spend: float
    target_roas: float
    target_cpa: float
    top_n: int
    show_labels: bool


@st.cache_data(show_spinner=False)
def _cached_campaign_data() -> pd.DataFrame:
    return load_campaign_data()


def render() -> None:
    st.subheader("Paid Media Performance Intelligence")
    st.caption(
        "Interactive exploration of the Kaggle digital advertising campaign dataset, "
        "with period deltas, trends, segment diagnostics, funnel health, and campaign actions."
    )

    _inject_dashboard_css()
    data = _cached_campaign_data()
    _render_dataset_expander(data)

    controls = _render_controls(data)
    filtered = filter_campaigns(data, controls.filters)
    filtered = filtered[filtered["spend"] >= controls.min_spend].copy()
    if filtered.empty:
        st.warning("No campaigns match the current controls. Broaden the filters or lower the spend threshold.")
        return

    prepared = _prepare_campaigns(filtered, controls.target_roas, controls.target_cpa)
    previous = _previous_period_frame(data, controls)
    previous = previous[previous["spend"] >= controls.min_spend].copy()

    current_metrics = summarize_metrics(prepared)
    previous_metrics = summarize_metrics(previous) if not previous.empty else _empty_metrics()

    _render_metric_scoreboard(current_metrics, previous_metrics)
    _render_highlights(prepared, previous, controls)

    overview_tab, trends_tab, segments_tab, campaigns_tab = st.tabs(
        ["Overview", "Trend Explorer", "Segment Lens", "Campaign Lab"]
    )
    with overview_tab:
        _render_overview(prepared, controls)
    with trends_tab:
        _render_trend_explorer(prepared, controls)
    with segments_tab:
        _render_segment_lens(prepared, previous, controls)
    with campaigns_tab:
        _render_campaign_lab(prepared, controls)


def _render_dataset_expander(data: pd.DataFrame) -> None:
    min_date = data["date"].min().date()
    max_date = data["date"].max().date()
    with st.expander("Dataset source and setup", expanded=False):
        st.markdown(
            f"Primary dataset target: [Digital Advertising Campaign Performance Dataset]({KAGGLE_DATASET_URL}). "
            "Place the CSV in `data/` as `digital_advertising_campaign_performance.csv`, "
            "`digital_ad_campaigns.csv`, `paid_media_campaigns.csv`, or `campaign_performance.csv`."
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows loaded", f"{len(data):,}")
        c2.metric("Date coverage", f"{min_date} to {max_date}")
        c3.metric("Campaigns", f"{data['campaign_id'].nunique():,}")


def _render_controls(data: pd.DataFrame) -> DashboardControls:
    min_date = data["date"].min().date()
    max_date = data["date"].max().date()

    with st.container(border=True):
        st.markdown("#### Dashboard Controls")
        top_left, top_right = st.columns([1.15, 2.85])
        with top_left:
            default_start = max(min_date, (pd.Timestamp(max_date) - pd.Timedelta(days=89)).date())
            date_range = st.date_input(
                "Date range",
                value=(default_start, max_date),
                min_value=min_date,
                max_value=max_date,
                help="Previous-period deltas compare against the immediately preceding range of the same length.",
            )
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start_date, end_date = date_range
            else:
                start_date, end_date = default_start, max_date

            primary_metric_label = st.selectbox("Primary metric", list(METRIC_OPTIONS), index=2)
            granularity_label = st.radio("Trend grain", ["Daily", "Weekly", "Monthly"], horizontal=True, index=1)

        with top_right:
            f1, f2, f3 = st.columns(3)
            platforms = f1.multiselect("Platforms", unique_sorted(data, "platform"), default=[])
            objectives = f2.multiselect("Objectives", unique_sorted(data, "objective"), default=[])
            industries = f3.multiselect("Industries", unique_sorted(data, "industry"), default=[])

            f4, f5, f6 = st.columns(3)
            devices = f4.multiselect("Devices", unique_sorted(data, "device"), default=[])
            creatives = f5.multiselect("Creative formats", unique_sorted(data, "creative_format"), default=[])
            tiers = f6.multiselect("Budget tiers", unique_sorted(data, "budget_tier"), default=[])

            s1, s2, s3, s4 = st.columns(4)
            trend_label = s1.selectbox("Trend split", list(DIMENSION_OPTIONS), index=0)
            segment_label = s2.selectbox("Segment lens", list(DIMENSION_OPTIONS), index=0)
            heatmap_label = s3.selectbox("Heatmap rows", list(DIMENSION_OPTIONS), index=1)
            top_n = s4.slider("Top segments", min_value=5, max_value=25, value=12, step=1)

            b1, b2, b3, b4 = st.columns(4)
            min_spend = b1.number_input("Minimum row spend", min_value=0, max_value=100_000, value=0, step=100)
            target_roas = b2.number_input("Target ROAS", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
            target_cpa = b3.number_input("Target CPA", min_value=1, max_value=2_500, value=80, step=5)
            show_labels = b4.toggle("Show chart labels", value=False)

    granularity = {"Daily": "D", "Weekly": "W-MON", "Monthly": "MS"}[granularity_label]
    return DashboardControls(
        filters=CampaignFilters(
            start_date=pd.Timestamp(start_date),
            end_date=pd.Timestamp(end_date),
            platforms=to_tuple(platforms),
            objectives=to_tuple(objectives),
            industries=to_tuple(industries),
            devices=to_tuple(devices),
            creative_formats=to_tuple(creatives),
            budget_tiers=to_tuple(tiers),
        ),
        primary_metric=METRIC_OPTIONS[primary_metric_label],
        primary_metric_label=primary_metric_label,
        trend_dimension=DIMENSION_OPTIONS[trend_label],
        segment_dimension=DIMENSION_OPTIONS[segment_label],
        heatmap_dimension=DIMENSION_OPTIONS[heatmap_label],
        granularity=granularity,
        min_spend=float(min_spend),
        target_roas=float(target_roas),
        target_cpa=float(target_cpa),
        top_n=int(top_n),
        show_labels=bool(show_labels),
    )


def _prepare_campaigns(frame: pd.DataFrame, target_roas: float, target_cpa: float) -> pd.DataFrame:
    df = add_recommendations(detect_anomalies(frame)).copy()
    df["health_score"] = _health_score(df, target_roas, target_cpa)
    df["status"] = np.select(
        [
            df["roas"] >= target_roas,
            (df["cpa"] > target_cpa) & (df["conversions"] > 0),
            df["frequency"] >= 3.5,
            df["anomaly"],
        ],
        ["Above target", "High CPA", "Fatigue risk", "Anomaly"],
        default="Needs review",
    )
    df["action_reason"] = np.select(
        [
            df["recommendation"].eq("Scale"),
            df["recommendation"].eq("Watch"),
            df["recommendation"].eq("Optimize"),
            df["recommendation"].eq("Pause"),
        ],
        [
            "ROAS and profit are both strong.",
            "Efficient enough to monitor before scaling.",
            "Improve CPA, CVR, CTR, or audience quality.",
            "Spend is outpacing return.",
        ],
        default="Review delivery and conversion quality.",
    )
    return df


def _previous_period_frame(data: pd.DataFrame, controls: DashboardControls) -> pd.DataFrame:
    start = pd.Timestamp(controls.filters.start_date)
    end = pd.Timestamp(controls.filters.end_date)
    period_days = max((end - start).days, 0) + 1
    previous_end = start - pd.Timedelta(days=1)
    previous_start = previous_end - pd.Timedelta(days=period_days - 1)
    previous_filters = CampaignFilters(
        start_date=previous_start,
        end_date=previous_end,
        platforms=controls.filters.platforms,
        objectives=controls.filters.objectives,
        industries=controls.filters.industries,
        devices=controls.filters.devices,
        creative_formats=controls.filters.creative_formats,
        budget_tiers=controls.filters.budget_tiers,
    )
    return filter_campaigns(data, previous_filters)


def _render_metric_scoreboard(current: dict[str, float], previous: dict[str, float]) -> None:
    st.markdown("#### Executive Scoreboard")
    metrics = [
        ("Spend", "spend", _currency, "normal"),
        ("Revenue", "revenue", _currency, "normal"),
        ("Profit", "profit", _currency, "normal"),
        ("ROAS", "roas", _multiple, "normal"),
        ("CPA", "cpa", _currency, "inverse"),
        ("CTR", "ctr", _pct, "normal"),
        ("CVR", "cvr", _pct, "normal"),
        ("Conversions", "conversions", _number, "normal"),
    ]
    for row in (metrics[:4], metrics[4:]):
        columns = st.columns(4)
        for column, (label, key, formatter, delta_color) in zip(columns, row):
            column.metric(
                label,
                formatter(current.get(key, 0.0)),
                delta=_format_delta(current.get(key, 0.0), previous.get(key, 0.0), formatter),
                delta_color=delta_color,
            )


def _render_highlights(current: pd.DataFrame, previous: pd.DataFrame, controls: DashboardControls) -> None:
    st.markdown("#### Highlights")
    cards = _build_highlight_cards(current, previous, controls)
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        column.markdown(
            f"""
            <div class="mi-highlight">
                <div class="mi-highlight-label">{card['label']}</div>
                <div class="mi-highlight-value">{card['value']}</div>
                <div class="mi-highlight-note">{card['note']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _build_highlight_cards(current: pd.DataFrame, previous: pd.DataFrame, controls: DashboardControls) -> list[dict[str, str]]:
    segment = _aggregate_by(current, [controls.segment_dimension]).sort_values(controls.primary_metric, ascending=False)
    top_segment = segment.iloc[0] if not segment.empty else None

    movement = _segment_movement(current, previous, controls.segment_dimension)
    mover = movement.sort_values("profit_delta", ascending=False).head(1)

    fatigue = _aggregate_by(current, ["platform", "creative_format"])
    fatigue = fatigue[(fatigue["frequency"] >= 3.2) & (fatigue["ctr"] < fatigue["ctr"].median())].sort_values("spend", ascending=False)

    waste = current[(current["roas"] < controls.target_roas) & (current["cpa"] > controls.target_cpa)].sort_values("spend", ascending=False)
    scale = current[current["recommendation"].eq("Scale")].sort_values("profit", ascending=False)

    return [
        {
            "label": "Top segment",
            "value": str(top_segment[controls.segment_dimension]) if top_segment is not None else "None",
            "note": f"{controls.primary_metric_label}: {_format_metric_value(top_segment[controls.primary_metric], controls.primary_metric) if top_segment is not None else '0'}",
        },
        {
            "label": "Largest profit lift",
            "value": str(mover.iloc[0][controls.segment_dimension]) if not mover.empty else "No prior data",
            "note": _currency(float(mover.iloc[0]["profit_delta"])) if not mover.empty else "Previous period unavailable",
        },
        {
            "label": "Scale candidates",
            "value": f"{len(scale):,}",
            "note": f"{_currency(float(scale['profit'].sum()))} profit in filtered rows",
        },
        {
            "label": "Fatigue watch",
            "value": f"{len(fatigue):,}",
            "note": "High frequency with below-median CTR",
        },
        {
            "label": "Budget at risk",
            "value": _currency(float(waste["spend"].sum())),
            "note": f"{len(waste):,} rows below ROAS or above CPA target",
        },
    ]


def _render_overview(frame: pd.DataFrame, controls: DashboardControls) -> None:
    left, right = st.columns([1.35, 1])
    with left:
        st.markdown("#### Spend, Revenue, and Profit")
        trend = _time_series(frame, controls.granularity)
        trend_long = trend.melt(
            id_vars="period",
            value_vars=["spend", "revenue", "profit"],
            var_name="metric",
            value_name="value",
        )
        fig = px.area(
            trend_long,
            x="period",
            y="value",
            color="metric",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"period": "", "value": "Value", "metric": "Metric"},
        )
        fig.update_layout(hovermode="x unified", legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    with right:
        st.markdown("#### Funnel Health")
        _render_funnel(frame)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Action Mix")
        action = frame.groupby("recommendation", as_index=False).agg(
            rows=("campaign_id", "count"),
            spend=("spend", "sum"),
            profit=("profit", "sum"),
        )
        fig = px.bar(
            action,
            x="recommendation",
            y="spend",
            color="recommendation",
            text="spend" if controls.show_labels else None,
            color_discrete_map={"Scale": "#059669", "Watch": "#2563eb", "Optimize": "#f97316", "Pause": "#dc2626"},
            labels={"recommendation": "", "spend": "Spend"},
        )
        fig.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    with c2:
        st.markdown("#### Efficiency Frontier")
        frontier = _aggregate_by(frame, ["platform", "objective"])
        fig = px.scatter(
            frontier,
            x="cpa",
            y="roas",
            size="spend",
            color="platform",
            hover_name="objective",
            color_discrete_sequence=COLOR_SEQUENCE,
            labels={"cpa": "CPA", "roas": "ROAS", "spend": "Spend"},
        )
        fig.add_hline(y=controls.target_roas, line_dash="dash", line_color="#059669", annotation_text="Target ROAS")
        fig.add_vline(x=controls.target_cpa, line_dash="dash", line_color="#dc2626", annotation_text="Target CPA")
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)


def _render_trend_explorer(frame: pd.DataFrame, controls: DashboardControls) -> None:
    st.markdown("#### Metric Trend by Segment")
    split = controls.trend_dimension
    top_segments = (
        _aggregate_by(frame, [split])
        .sort_values(controls.primary_metric, ascending=False)
        .head(controls.top_n)[split]
        .tolist()
    )
    trend = _time_series(frame[frame[split].isin(top_segments)], controls.granularity, split)
    if trend.empty:
        st.info("No trend data is available for the current split.")
        return

    fig = px.line(
        trend,
        x="period",
        y=controls.primary_metric,
        color=split,
        markers=True,
        color_discrete_sequence=COLOR_SEQUENCE,
        labels={"period": "", controls.primary_metric: controls.primary_metric_label, split: split.replace("_", " ").title()},
    )
    fig.update_layout(hovermode="x unified", legend_title_text="", margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, config=PLOTLY_CONFIG)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Rolling KPI Pulse")
        rolling = _daily_rollup(frame)
        metric = controls.primary_metric
        if metric in rolling.columns:
            rolling["rolling_14"] = rolling[metric].rolling(window=14, min_periods=3).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=rolling["date"], y=rolling[metric], mode="lines", name=controls.primary_metric_label))
            fig.add_trace(go.Scatter(x=rolling["date"], y=rolling["rolling_14"], mode="lines", name="14-day average"))
            fig.update_layout(hovermode="x unified", margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, config=PLOTLY_CONFIG)

    with c2:
        st.markdown("#### Day-of-Week Pattern")
        weekday = _aggregate_by(frame.assign(weekday=frame["date"].dt.day_name()), ["weekday"])
        weekday["order"] = pd.Categorical(
            weekday["weekday"],
            categories=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            ordered=True,
        )
        weekday = weekday.sort_values("order")
        fig = px.bar(
            weekday,
            x="weekday",
            y=controls.primary_metric,
            color=controls.primary_metric,
            text=controls.primary_metric if controls.show_labels else None,
            color_continuous_scale="Bluered",
            labels={"weekday": "", controls.primary_metric: controls.primary_metric_label},
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)


def _render_segment_lens(current: pd.DataFrame, previous: pd.DataFrame, controls: DashboardControls) -> None:
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown("#### Segment Contribution")
        segment = _aggregate_by(current, [controls.segment_dimension]).sort_values(controls.primary_metric, ascending=False).head(controls.top_n)
        fig = px.bar(
            segment,
            x=controls.primary_metric,
            y=controls.segment_dimension,
            orientation="h",
            color="roas",
            text=controls.primary_metric if controls.show_labels else None,
            color_continuous_scale="RdYlGn",
            labels={controls.primary_metric: controls.primary_metric_label, controls.segment_dimension: ""},
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(coloraxis_colorbar_title="ROAS", margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    with right:
        st.markdown("#### Period Movement")
        movement = _segment_movement(current, previous, controls.segment_dimension).head(controls.top_n)
        if movement.empty:
            st.info("Previous-period data is not available for this date range.")
        else:
            fig = px.scatter(
                movement,
                x="spend_delta",
                y="profit_delta",
                size="current_spend",
                color="roas_delta",
                hover_name=controls.segment_dimension,
                color_continuous_scale="RdBu",
                labels={
                    "spend_delta": "Spend change",
                    "profit_delta": "Profit change",
                    "roas_delta": "ROAS change",
                    "current_spend": "Current spend",
                },
            )
            fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
            fig.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
            fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, config=PLOTLY_CONFIG)

    st.markdown("#### Cross-Dimension Heatmap")
    heatmap = _heatmap_frame(current, controls.heatmap_dimension, controls.segment_dimension, controls.primary_metric)
    if heatmap.empty:
        st.info("Choose two different dimensions with enough data to populate the heatmap.")
    else:
        fig = px.imshow(
            heatmap,
            aspect="auto",
            color_continuous_scale="Viridis",
            labels=dict(color=controls.primary_metric_label, x=controls.segment_dimension.replace("_", " ").title(), y=controls.heatmap_dimension.replace("_", " ").title()),
        )
        fig.update_layout(margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, config=PLOTLY_CONFIG)

    st.markdown("#### Segment Detail")
    detail = _aggregate_by(current, [controls.segment_dimension]).sort_values(controls.primary_metric, ascending=False)
    st.dataframe(
        _format_table(detail.head(controls.top_n * 2)),
        width="stretch",
        hide_index=True,
        column_config=_column_config(),
    )


def _render_campaign_lab(frame: pd.DataFrame, controls: DashboardControls) -> None:
    st.markdown("#### Campaign Triage")
    c1, c2, c3 = st.columns([1.4, 1, 1])
    search = c1.text_input("Search campaigns", value="", placeholder="Filter by campaign name, platform, objective, or industry")
    recommendation = c2.selectbox("Recommendation", ["All", "Scale", "Watch", "Optimize", "Pause"], index=0)
    sort_label = c3.selectbox("Sort by", ["Health score", "Profit", "Spend", "ROAS", "CPA", "CTR", "Conversions"], index=0)

    table = frame.copy()
    if search:
        query = search.lower().strip()
        searchable = table[["campaign_name", "platform", "objective", "industry", "creative_format"]].astype(str).agg(" ".join, axis=1).str.lower()
        table = table[searchable.str.contains(query, regex=False)]
    if recommendation != "All":
        table = table[table["recommendation"].eq(recommendation)]

    sort_column = {
        "Health score": "health_score",
        "Profit": "profit",
        "Spend": "spend",
        "ROAS": "roas",
        "CPA": "cpa",
        "CTR": "ctr",
        "Conversions": "conversions",
    }[sort_label]
    ascending = sort_column == "cpa"
    table = table.sort_values(sort_column, ascending=ascending)

    c4, c5, c6, c7 = st.columns(4)
    c4.metric("Visible rows", f"{len(table):,}")
    c5.metric("Visible spend", _currency(float(table["spend"].sum())))
    c6.metric("Visible profit", _currency(float(table["profit"].sum())))
    c7.metric("Visible ROAS", _multiple(float(table["revenue"].sum() / table["spend"].sum())) if table["spend"].sum() else "0.00x")

    columns = [
        "campaign_name",
        "platform",
        "objective",
        "industry",
        "audience_segment",
        "creative_format",
        "spend",
        "revenue",
        "profit",
        "roas",
        "cpa",
        "ctr",
        "cvr",
        "frequency",
        "health_score",
        "recommendation",
        "status",
        "action_reason",
        "anomaly",
    ]
    display = table[columns].head(500)
    st.dataframe(
        _format_table(display),
        width="stretch",
        hide_index=True,
        column_config=_column_config(),
    )

    csv = _to_csv(display)
    st.download_button(
        "Download visible campaign rows",
        data=csv,
        file_name="performance_campaign_triage.csv",
        mime="text/csv",
    )

    st.markdown("#### Planner Scenario")
    shift = st.slider("Scenario spend change", min_value=-50, max_value=100, value=20, step=5, format="%d%%")
    scenario = _scenario_projection(frame, shift / 100)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Projected spend", _currency(scenario["spend"]), delta=_currency(scenario["spend_delta"]))
    s2.metric("Projected revenue", _currency(scenario["revenue"]), delta=_currency(scenario["revenue_delta"]))
    s3.metric("Projected profit", _currency(scenario["profit"]), delta=_currency(scenario["profit_delta"]))
    s4.metric("Projected ROAS", _multiple(scenario["roas"]))


def _render_funnel(frame: pd.DataFrame) -> None:
    values = {
        "Impressions": frame["impressions"].sum(),
        "Reach": frame["reach"].sum(),
        "Clicks": frame["clicks"].sum(),
        "Landing page views": frame["landing_page_view"].sum(),
        "Add to carts": frame["add_to_cart"].sum(),
        "Conversions": frame["conversions"].sum(),
    }
    fig = go.Figure(
        go.Funnel(
            y=list(values.keys()),
            x=list(values.values()),
            textinfo="value+percent previous",
            marker={"color": ["#2563eb", "#0891b2", "#059669", "#ca8a04", "#f97316", "#dc2626"]},
        )
    )
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=350)
    st.plotly_chart(fig, config=PLOTLY_CONFIG)


def _time_series(frame: pd.DataFrame, granularity: str, dimension: str | None = None) -> pd.DataFrame:
    df = frame.copy()
    df["period"] = df["date"].dt.to_period(_period_alias(granularity)).dt.to_timestamp()
    groupers = ["period"] + ([dimension] if dimension else [])
    return _aggregate_by(df, groupers).sort_values(groupers)


def _daily_rollup(frame: pd.DataFrame) -> pd.DataFrame:
    return _aggregate_by(frame.assign(date=frame["date"].dt.floor("D")), ["date"]).sort_values("date")


def _aggregate_by(frame: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=dimensions + AGGREGATE_COLUMNS + ["roas", "cpa", "ctr", "cpc", "cvr", "frequency"])

    agg = (
        frame.groupby(dimensions, dropna=False, as_index=False)
        .agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            impressions=("impressions", "sum"),
            reach=("reach", "sum"),
            clicks=("clicks", "sum"),
            landing_page_view=("landing_page_view", "sum"),
            add_to_cart=("add_to_cart", "sum"),
            conversions=("conversions", "sum"),
            video_views=("video_views", "sum"),
            campaigns=("campaign_id", "nunique"),
        )
        .copy()
    )
    return _add_weighted_metrics(agg)


def _add_weighted_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["profit"] = df["revenue"] - df["spend"]
    df["roas"] = _divide(df["revenue"], df["spend"])
    df["cpa"] = _divide(df["spend"], df["conversions"])
    df["cpc"] = _divide(df["spend"], df["clicks"])
    df["ctr"] = _divide(df["clicks"], df["impressions"])
    df["cvr"] = _divide(df["conversions"], df["clicks"])
    df["frequency"] = _divide(df["impressions"], df["reach"])
    df["landing_page_rate"] = _divide(df["landing_page_view"], df["clicks"])
    df["cart_rate"] = _divide(df["add_to_cart"], df["landing_page_view"])
    df["video_view_rate"] = _divide(df["video_views"], df["impressions"])
    return df


def _segment_movement(current: pd.DataFrame, previous: pd.DataFrame, dimension: str) -> pd.DataFrame:
    if previous.empty:
        return pd.DataFrame()

    current_segment = _aggregate_by(current, [dimension]).add_prefix("current_")
    previous_segment = _aggregate_by(previous, [dimension]).add_prefix("previous_")
    merged = current_segment.merge(
        previous_segment,
        left_on=f"current_{dimension}",
        right_on=f"previous_{dimension}",
        how="outer",
    )
    merged[dimension] = merged[f"current_{dimension}"].fillna(merged[f"previous_{dimension}"])
    for metric in ["spend", "profit", "revenue", "conversions", "roas", "cpa", "ctr", "cvr"]:
        merged[f"current_{metric}"] = merged[f"current_{metric}"].fillna(0)
        merged[f"previous_{metric}"] = merged[f"previous_{metric}"].fillna(0)
        merged[f"{metric}_delta"] = merged[f"current_{metric}"] - merged[f"previous_{metric}"]
    return merged.sort_values("profit_delta", ascending=False)


def _heatmap_frame(frame: pd.DataFrame, row_dimension: str, column_dimension: str, metric: str) -> pd.DataFrame:
    if row_dimension == column_dimension:
        return pd.DataFrame()
    rows = _aggregate_by(frame, [row_dimension]).sort_values(metric, ascending=False).head(10)[row_dimension].tolist()
    columns = _aggregate_by(frame, [column_dimension]).sort_values(metric, ascending=False).head(10)[column_dimension].tolist()
    source = frame[frame[row_dimension].isin(rows) & frame[column_dimension].isin(columns)]
    grid = _aggregate_by(source, [row_dimension, column_dimension])
    if grid.empty:
        return pd.DataFrame()
    return grid.pivot(index=row_dimension, columns=column_dimension, values=metric).fillna(0)


def _health_score(frame: pd.DataFrame, target_roas: float, target_cpa: float) -> pd.Series:
    roas_score = np.clip(frame["roas"] / max(target_roas, 0.01), 0, 2.0) * 35
    cpa_score = np.clip(max(target_cpa, 0.01) / frame["cpa"].replace(0, np.nan), 0, 2.0).fillna(0) * 25
    cvr_score = _percentile_rank(frame["cvr"]) * 20
    ctr_score = _percentile_rank(frame["ctr"]) * 10
    profit_score = _percentile_rank(frame["profit"]) * 10
    penalty = np.where(frame["anomaly"], 12, 0) + np.where(frame["frequency"] > 4.0, 8, 0)
    return np.clip(roas_score + cpa_score + cvr_score + ctr_score + profit_score - penalty, 0, 100).round(1)


def _percentile_rank(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True).fillna(0)


def _scenario_projection(frame: pd.DataFrame, spend_shift: float) -> dict[str, float]:
    base = summarize_metrics(frame)
    multiplier = 1 + spend_shift
    # Mild diminishing returns on increases and mild efficiency loss on sharp cuts.
    response = multiplier ** 0.82 if multiplier >= 1 else multiplier ** 1.08
    projected_spend = max(base["spend"] * multiplier, 0)
    projected_revenue = max(base["revenue"] * response, 0)
    projected_profit = projected_revenue - projected_spend
    return {
        "spend": projected_spend,
        "revenue": projected_revenue,
        "profit": projected_profit,
        "roas": projected_revenue / projected_spend if projected_spend else 0,
        "spend_delta": projected_spend - base["spend"],
        "revenue_delta": projected_revenue - base["revenue"],
        "profit_delta": projected_profit - base["profit"],
    }


def _empty_metrics() -> dict[str, float]:
    return {
        "spend": 0.0,
        "revenue": 0.0,
        "profit": 0.0,
        "roas": 0.0,
        "cpa": 0.0,
        "ctr": 0.0,
        "cvr": 0.0,
        "conversions": 0.0,
        "impressions": 0.0,
        "clicks": 0.0,
    }


def _divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)


def _period_alias(granularity: str) -> str:
    if granularity == "W-MON":
        return "W"
    if granularity == "MS":
        return "M"
    return "D"


def _format_table(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    for column in ["spend", "revenue", "profit", "cpa", "cpc"]:
        if column in df:
            df[column] = df[column].map(_currency)
    for column in ["ctr", "cvr", "landing_page_rate", "cart_rate", "video_view_rate"]:
        if column in df:
            df[column] = df[column].map(_pct)
    for column in ["roas", "frequency"]:
        if column in df:
            df[column] = df[column].map(_multiple)
    for column in ["impressions", "reach", "clicks", "landing_page_view", "add_to_cart", "conversions", "video_views", "campaigns"]:
        if column in df:
            df[column] = df[column].map(_number)
    return df


def _column_config() -> dict[str, st.column_config.Column]:
    return {
        "campaign_name": st.column_config.TextColumn("Campaign"),
        "audience_segment": st.column_config.TextColumn("Audience"),
        "creative_format": st.column_config.TextColumn("Creative"),
        "health_score": st.column_config.ProgressColumn("Health", min_value=0, max_value=100, format="%.1f"),
        "action_reason": st.column_config.TextColumn("Why"),
        "anomaly": st.column_config.CheckboxColumn("Anomaly"),
    }


def _to_csv(frame: pd.DataFrame) -> str:
    output = StringIO()
    frame.to_csv(output, index=False)
    return output.getvalue()


def _format_delta(current: float, previous: float, formatter) -> str | None:
    if previous == 0:
        return None
    absolute = current - previous
    pct_change = absolute / abs(previous)
    return f"{formatter(absolute)} ({pct_change:+.1%})"


def _format_metric_value(value: float, metric: str) -> str:
    if metric in {"spend", "revenue", "profit", "cpa", "cpc"}:
        return _currency(value)
    if metric in {"ctr", "cvr", "landing_page_rate", "cart_rate", "video_view_rate"}:
        return _pct(value)
    if metric in {"roas", "frequency"}:
        return _multiple(value)
    return _number(value)


def _currency(value: float) -> str:
    return f"${value:,.0f}"


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _multiple(value: float) -> str:
    return f"{value:.2f}x"


def _number(value: float) -> str:
    return f"{value:,.0f}"


def _inject_dashboard_css() -> None:
    st.markdown(
        """
        <style>
        .mi-highlight {
            border: 1px solid rgba(148, 163, 184, 0.25);
            border-radius: 8px;
            padding: 0.85rem 0.9rem;
            min-height: 130px;
            background: rgba(248, 250, 252, 0.78);
        }
        .mi-highlight-label {
            color: #475569;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0;
            text-transform: uppercase;
        }
        .mi-highlight-value {
            color: #0f172a;
            font-size: 1.35rem;
            font-weight: 750;
            line-height: 1.18;
            margin-top: 0.35rem;
            word-break: break-word;
        }
        .mi-highlight-note {
            color: #475569;
            font-size: 0.86rem;
            line-height: 1.35;
            margin-top: 0.45rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
