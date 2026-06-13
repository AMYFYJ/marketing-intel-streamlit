from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from data_sources.campaign_data import (
    CampaignFilters,
    KAGGLE_DATASET_URL,
    add_recommendations,
    aggregate_campaigns,
    detect_anomalies,
    filter_campaigns,
    load_campaign_data,
    summarize_metrics,
    to_tuple,
    unique_sorted,
)
from utils.formatting import display_labels, format_currency, format_pct, format_roas, title_case_columns

RECOMMENDATION_COLORS = {"Scale": "#2ca02c", "Watch": "#f0ad4e", "Optimize": "#1f77b4", "Pause": "#d62728"}
GRANULARITY_FREQ = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}
FUNNEL_LEAKAGE_STEPS = {
    "Reach / Impr.": ("reach", "impressions"),
    "Click / Impr.": ("clicks", "impressions"),
    "LPV / Click": ("landing_page_view", "clicks"),
    "Cart / LPV": ("add_to_cart", "landing_page_view"),
    "Conv. / Cart": ("conversions", "add_to_cart"),
    "Conv. / Click": ("conversions", "clicks"),
    "Conv. / Impr.": ("conversions", "impressions"),
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

CAMPAIGN_TABLE_COLUMN_CONFIG = {
    "Spend": st.column_config.NumberColumn("Spend", format="dollar"),
    "Revenue": st.column_config.NumberColumn("Revenue", format="dollar"),
    "Profit": st.column_config.NumberColumn("Profit", format="dollar"),
    "CPA": st.column_config.NumberColumn("CPA", format="dollar"),
    "ROAS": st.column_config.NumberColumn("ROAS", format="%.2fx"),
    "CTR": st.column_config.NumberColumn("CTR", format="percent"),
    "CVR": st.column_config.NumberColumn("CVR", format="percent"),
    "Date": st.column_config.DatetimeColumn("Date", format="YYYY-MM-DD"),
}


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
    with st.expander("Dataset Source and Setup", expanded=False):
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

    campaign_level = add_recommendations(aggregate_campaigns(filtered))
    flagged_days = detect_anomalies(filtered)

    _render_kpis(data, filters, filtered)
    _render_charts(filtered, campaign_level)
    _render_campaign_table(campaign_level)
    _render_anomaly_watchlist(flagged_days)


def _render_filters(data: pd.DataFrame) -> CampaignFilters:
    st.markdown("#### Controls")
    min_date = data["date"].min().date()
    max_date = data["date"].max().date()
    left, right = st.columns([1, 3])
    with left:
        date_range = st.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        st.warning("Select both a start and an end date — showing the full range until then.")
        start_date, end_date = min_date, max_date

    with right:
        c1, c2, c3 = st.columns(3)
        platforms = c1.multiselect("Platforms", unique_sorted(data, "platform"), default=[])
        objectives = c2.multiselect("Objectives", unique_sorted(data, "objective"), default=[])
        industries = c3.multiselect("Industries", unique_sorted(data, "industry"), default=[])
        c4, c5, c6 = st.columns(3)
        devices = c4.multiselect("Devices", unique_sorted(data, "device"), default=[])
        creatives = c5.multiselect("Creative Formats", unique_sorted(data, "creative_format"), default=[])
        tiers = c6.multiselect("Budget Tiers", unique_sorted(data, "budget_tier"), default=[])

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


def _previous_period_metrics(data: pd.DataFrame, filters: CampaignFilters) -> dict[str, float] | None:
    """Metrics for the window of equal length immediately before the selected range."""
    if filters.start_date is None or filters.end_date is None:
        return None
    span = filters.end_date - filters.start_date
    prev_end = filters.start_date - pd.Timedelta(days=1)
    prev_start = prev_end - span
    if prev_end < data["date"].min():
        return None
    previous = filter_campaigns(
        data,
        CampaignFilters(
            start_date=prev_start,
            end_date=prev_end,
            platforms=filters.platforms,
            objectives=filters.objectives,
            industries=filters.industries,
            devices=filters.devices,
            creative_formats=filters.creative_formats,
            budget_tiers=filters.budget_tiers,
        ),
    )
    if previous.empty:
        return None
    return summarize_metrics(previous)


def _render_kpis(data: pd.DataFrame, filters: CampaignFilters, filtered: pd.DataFrame) -> None:
    metrics = summarize_metrics(filtered)
    previous = _previous_period_metrics(data, filters)

    def delta(key: str, formatter) -> str | None:
        if previous is None:
            return None
        return formatter(metrics[key] - previous[key])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spend", format_currency(metrics["spend"]), delta=delta("spend", format_currency))
    c2.metric("Revenue", format_currency(metrics["revenue"]), delta=delta("revenue", format_currency))
    c3.metric("Profit", format_currency(metrics["profit"]), delta=delta("profit", format_currency))
    c4.metric("ROAS", format_roas(metrics["roas"]), delta=delta("roas", lambda v: f"{v:+.2f}x"))
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("CPA", format_currency(metrics["cpa"]), delta=delta("cpa", format_currency), delta_color="inverse")
    c6.metric("CTR", format_pct(metrics["ctr"]), delta=delta("ctr", lambda v: f"{v * 100:+.2f}pp"))
    c7.metric("CVR", format_pct(metrics["cvr"]), delta=delta("cvr", lambda v: f"{v * 100:+.2f}pp"))
    c8.metric("Conversions", f"{metrics['conversions']:,.0f}", delta=delta("conversions", lambda v: f"{v:+,.0f}"))
    if previous is not None:
        st.caption("Deltas compare against the preceding period of equal length with the same filters.")


def _render_charts(frame: pd.DataFrame, campaign_level: pd.DataFrame) -> None:
    st.markdown("#### Performance Trends")
    granularity = st.radio("Granularity", list(GRANULARITY_FREQ), index=1, horizontal=True, label_visibility="collapsed")
    freq = GRANULARITY_FREQ[granularity]
    trend = (
        frame.set_index("date")[["spend", "revenue", "profit"]]
        .resample(freq)
        .sum()
        .reset_index()
    )
    trend["roas"] = (trend["revenue"] / trend["spend"].replace(0, pd.NA)).astype(float)
    if granularity == "Daily":
        trend[["spend", "revenue", "profit"]] = trend[["spend", "revenue", "profit"]].rolling(7, min_periods=1).mean()
        trend["roas"] = trend["roas"].rolling(7, min_periods=1).mean()

    c1, c2 = st.columns(2)
    money_long = trend.melt(id_vars="date", value_vars=["spend", "revenue", "profit"], var_name="metric", value_name="value")
    money_title = "Spend, Revenue, and Profit" + (" (7-Day Rolling Avg)" if granularity == "Daily" else "")
    c1.plotly_chart(
        px.line(money_long, x="date", y="value", color="metric", title=money_title, labels=display_labels(["date", "value", "metric"])),
        use_container_width=True,
    )
    c2.plotly_chart(
        px.line(trend, x="date", y="roas", title=f"Blended ROAS ({granularity})", labels=display_labels(["date", "roas"])),
        use_container_width=True,
    )

    c3, c4 = st.columns(2)
    platform = frame.groupby("platform", as_index=False).agg(spend=("spend", "sum"), revenue=("revenue", "sum"), profit=("profit", "sum"))
    platform["roas"] = platform["revenue"] / platform["spend"].replace(0, pd.NA)
    c3.plotly_chart(
        px.bar(
            platform.sort_values("profit"),
            x="platform",
            y="profit",
            color="roas",
            title="Profit by Platform (Color = Weighted ROAS)",
            labels=display_labels(["platform", "profit", "roas"]),
        ),
        use_container_width=True,
    )

    objective = campaign_level.groupby(["objective", "creative_format"], as_index=False).agg(
        spend=("spend", "sum"), revenue=("revenue", "sum"), cpa=("cpa", "median"), roas=("roas", "mean")
    )
    c4.plotly_chart(
        px.scatter(
            objective,
            x="cpa",
            y="roas",
            size="spend",
            color="objective",
            hover_name="creative_format",
            title="Objective Efficiency Map",
            labels=display_labels(["cpa", "roas", "spend", "objective"]),
        ),
        use_container_width=True,
    )

    mix = campaign_level.groupby(["platform", "recommendation"], as_index=False).agg(campaigns=("campaign_id", "count"))
    st.plotly_chart(
        px.bar(
            mix,
            x="platform",
            y="campaigns",
            color="recommendation",
            color_discrete_map=RECOMMENDATION_COLORS,
            title="Recommendation Mix by Platform (Campaigns)",
            labels=display_labels(["platform", "campaigns", "recommendation"]),
        ),
        use_container_width=True,
    )


def _render_campaign_table(campaign_level: pd.DataFrame) -> None:
    st.markdown("#### Campaign Actions")
    options = ["Pause", "Optimize", "Watch", "Scale"]
    selected = st.multiselect("Recommendations", options, default=options, help="Filter the table by recommended action.")
    table = campaign_level[campaign_level["recommendation"].isin(selected)].copy()

    # Most actionable first: Pause campaigns by spend, then the rest by spend.
    action_rank = {"Pause": 0, "Optimize": 1, "Watch": 2, "Scale": 3}
    table["_rank"] = table["recommendation"].map(action_rank)
    table = table.sort_values(["_rank", "spend"], ascending=[True, False]).drop(columns="_rank")

    columns = [
        "campaign_name",
        "platform",
        "objective",
        "industry",
        "days_active",
        "spend",
        "revenue",
        "profit",
        "roas",
        "cpa",
        "ctr",
        "cvr",
        "recommendation",
    ]
    columns = [column for column in columns if column in table.columns]
    display = title_case_columns(table[columns])
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config=CAMPAIGN_TABLE_COLUMN_CONFIG,
    )
    st.download_button(
        "Download Campaign Actions (CSV)",
        display.to_csv(index=False).encode("utf-8"),
        file_name="campaign_actions.csv",
        mime="text/csv",
    )


def _ranked_segments(frame: pd.DataFrame, dimension: str, metric: str, limit: int) -> pd.DataFrame:
    segment = _aggregate_by(frame, [dimension])
    if segment.empty:
        return segment

    if metric == "cpa":
        segment = segment[(segment["conversions"] > 0) & (segment["cpa"] > 0)]
    elif metric == "cvr":
        segment = segment[segment["clicks"] > 0]

    return segment.sort_values(metric, ascending=(metric == "cpa")).head(limit)


def _top_movers(current: pd.DataFrame, previous: pd.DataFrame, dimension: str, metric: str, limit: int) -> pd.DataFrame:
    movement = _segment_movement(current, previous, dimension)
    delta_column = f"{metric}_delta"
    if movement.empty or delta_column not in movement:
        return pd.DataFrame(columns=[dimension, delta_column, "abs_delta"])

    if metric == "cpa":
        movement = movement[(movement["current_conversions"] > 0) | (movement["previous_conversions"] > 0)]

    movement = movement.copy()
    movement["abs_delta"] = movement[delta_column].abs()
    movement = movement[movement["abs_delta"] > 0]
    if movement.empty:
        return pd.DataFrame(columns=[dimension, delta_column, "abs_delta"])
    return movement.sort_values("abs_delta", ascending=False).head(limit)


def _funnel_leakage_matrix(frame: pd.DataFrame, dimension: str, limit: int) -> pd.DataFrame:
    segment = _aggregate_by(frame, [dimension]).sort_values("spend", ascending=False).head(limit)
    if segment.empty:
        return pd.DataFrame()
    matrix = pd.DataFrame(index=segment[dimension].astype(str))
    for label, (numerator, denominator) in FUNNEL_LEAKAGE_STEPS.items():
        matrix[label] = _divide(segment[numerator], segment[denominator]).to_numpy()
    return matrix


def _funnel_metric_help_html() -> str:
    return """
    <div class="mi-axis-help">
        <strong>Reach / Impr.</strong>: unique reach divided by impressions.
        <strong>Click / Impr.</strong>: clicks divided by impressions.
        <strong>LPV / Click</strong>: landing page views divided by clicks.
        <strong>Cart / LPV</strong>: add-to-carts divided by landing page views.
        <strong>Conv. / Cart</strong>: conversions divided by add-to-carts.
        <strong>Conv. / Click</strong>: conversions divided by clicks.
        <strong>Conv. / Impr.</strong>: conversions divided by impressions.
    </div>
    """


def _normalize_rate_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.apply(lambda column: column / column.max() if column.max() else column)


def _axis_tickformat(metric: str) -> str | None:
    if metric == "cpc":
        return "$,.2f"
    if metric in {"spend", "revenue", "profit", "cpa"}:
        return "$,.0f"
    if metric in {"ctr", "cvr"}:
        return ".2%"
    if metric in {"conversions", "impressions", "clicks"}:
        return ",.0f"
    if metric in {"roas", "frequency"}:
        return ".2f"
    return None


def _aggregate_by(frame: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=dimensions + AGGREGATE_COLUMNS + ["roas", "cpa", "ctr", "cpc", "cvr", "frequency"])

    grouped = (
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
    return _add_weighted_metrics(grouped)


def _add_weighted_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["profit"] = df["revenue"] - df["spend"]
    df["roas"] = _divide(df["revenue"], df["spend"])
    df["cpa"] = _divide(df["spend"], df["conversions"])
    df["cpc"] = _divide(df["spend"], df["clicks"])
    df["ctr"] = _divide(df["clicks"], df["impressions"])
    df["cvr"] = _divide(df["conversions"], df["clicks"])
    df["frequency"] = _divide(df["impressions"], df["reach"])
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


def _divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return (numerator / denominator.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)


def _render_anomaly_watchlist(flagged_days: pd.DataFrame) -> None:
    anomalies = flagged_days[flagged_days["anomaly"]]
    if anomalies.empty:
        return
    st.markdown("#### Anomaly Watchlist")
    st.caption("Campaign-days that deviate sharply from the campaign's own history. Unfavorable rows need attention first.")
    watchlist = anomalies.sort_values(["date", "spend"], ascending=[False, False]).head(20)
    columns = ["date", "campaign_name", "platform", "anomaly_metric", "anomaly_direction", "spend", "roas", "cpa", "ctr", "cvr"]
    columns = [column for column in columns if column in watchlist.columns]
    st.dataframe(
        title_case_columns(watchlist[columns]),
        use_container_width=True,
        hide_index=True,
        column_config=CAMPAIGN_TABLE_COLUMN_CONFIG,
    )
