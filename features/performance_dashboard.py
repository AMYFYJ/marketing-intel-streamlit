from __future__ import annotations

import pandas as pd
import plotly.express as px
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


@st.cache_data(show_spinner=False)
def _cached_campaign_data() -> pd.DataFrame:
    return load_campaign_data()


def render() -> None:
    st.subheader("Paid Media Performance Command Center")
    st.caption(
        "Uses the public Kaggle campaign-performance dataset when present in data/, "
        "with a deterministic fallback sample for deployment demos."
    )

    data = _cached_campaign_data()
    with st.expander("Dataset source and setup", expanded=False):
        st.markdown(
            f"Primary dataset target: [Digital Advertising Campaign Performance Dataset]({KAGGLE_DATASET_URL}). "
            "Place the CSV in `data/` as `digital_advertising_campaign_performance.csv`, "
            "`digital_ad_campaigns.csv`, `paid_media_campaigns.csv`, or `campaign_performance.csv`."
        )

    filters = _render_filters(data)
    filtered = filter_campaigns(data, filters)
    if filtered.empty:
        st.warning("No campaigns match the current filters. Broaden the selection to restore the dashboard.")
        return

    recommended = add_recommendations(detect_anomalies(filtered))
    metrics = summarize_metrics(recommended)
    _render_kpis(metrics)
    _render_charts(recommended)
    _render_campaign_tables(recommended)


def _render_filters(data: pd.DataFrame) -> CampaignFilters:
    st.markdown("#### Controls")
    min_date = data["date"].min().date()
    max_date = data["date"].max().date()
    left, right = st.columns([1, 3])
    with left:
        date_range = st.date_input("Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    with right:
        c1, c2, c3 = st.columns(3)
        platforms = c1.multiselect("Platforms", unique_sorted(data, "platform"), default=[])
        objectives = c2.multiselect("Objectives", unique_sorted(data, "objective"), default=[])
        industries = c3.multiselect("Industries", unique_sorted(data, "industry"), default=[])
        c4, c5, c6 = st.columns(3)
        devices = c4.multiselect("Devices", unique_sorted(data, "device"), default=[])
        creatives = c5.multiselect("Creative formats", unique_sorted(data, "creative_format"), default=[])
        tiers = c6.multiselect("Budget tiers", unique_sorted(data, "budget_tier"), default=[])

    return CampaignFilters(
        start_date=pd.Timestamp(start_date),
        end_date=pd.Timestamp(end_date),
        platforms=to_tuple(platforms),
        objectives=to_tuple(objectives),
        industries=to_tuple(industries),
        devices=to_tuple(devices),
        creative_formats=to_tuple(creatives),
        budget_tiers=to_tuple(tiers),
    )


def _render_kpis(metrics: dict[str, float]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spend", _currency(metrics["spend"]))
    c2.metric("Revenue", _currency(metrics["revenue"]))
    c3.metric("Profit", _currency(metrics["profit"]))
    c4.metric("ROAS", f"{metrics['roas']:.2f}x")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("CPA", _currency(metrics["cpa"]))
    c6.metric("CTR", _pct(metrics["ctr"]))
    c7.metric("CVR", _pct(metrics["cvr"]))
    c8.metric("Conversions", f"{metrics['conversions']:,.0f}")


def _render_charts(frame: pd.DataFrame) -> None:
    st.markdown("#### Performance Trends")
    daily = frame.groupby("date", as_index=False)[["spend", "revenue", "profit", "conversions"]].sum()
    daily_long = daily.melt(id_vars="date", value_vars=["spend", "revenue", "profit"], var_name="metric", value_name="value")
    st.plotly_chart(px.line(daily_long, x="date", y="value", color="metric", title="Spend, revenue, and profit over time"), use_container_width=True)

    c1, c2 = st.columns(2)
    platform = frame.groupby("platform", as_index=False).agg(spend=("spend", "sum"), revenue=("revenue", "sum"), profit=("profit", "sum"), roas=("roas", "mean"))
    c1.plotly_chart(px.bar(platform.sort_values("profit"), x="platform", y="profit", color="roas", title="Profit by platform"), use_container_width=True)

    objective = frame.groupby(["objective", "creative_format"], as_index=False).agg(spend=("spend", "sum"), revenue=("revenue", "sum"), cpa=("cpa", "median"), roas=("roas", "mean"))
    c2.plotly_chart(px.scatter(objective, x="cpa", y="roas", size="spend", color="objective", hover_name="creative_format", title="Objective efficiency map"), use_container_width=True)

    mix = frame.groupby(["platform", "recommendation"], as_index=False).agg(campaigns=("campaign_id", "count"))
    st.plotly_chart(px.bar(mix, x="platform", y="campaigns", color="recommendation", title="Recommendation mix by platform"), use_container_width=True)


def _render_campaign_tables(frame: pd.DataFrame) -> None:
    st.markdown("#### Campaign Actions")
    ranked = frame.sort_values(["recommendation", "profit", "roas"], ascending=[True, False, False])
    columns = [
        "campaign_name",
        "platform",
        "objective",
        "industry",
        "spend",
        "revenue",
        "profit",
        "roas",
        "cpa",
        "ctr",
        "cvr",
        "recommendation",
        "anomaly",
    ]
    st.dataframe(ranked[columns].head(250), use_container_width=True, hide_index=True)

    anomalies = frame[frame["anomaly"]].sort_values("spend", ascending=False).head(25)
    if not anomalies.empty:
        st.markdown("#### Anomaly Watchlist")
        st.dataframe(anomalies[columns], use_container_width=True, hide_index=True)


def _currency(value: float) -> str:
    return f"${value:,.0f}"


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"
