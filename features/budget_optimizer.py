from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data_sources.connectors import available_connector_specs, normalize_uploaded_campaign_csv
from data_sources.synthetic_media_mix import DEFAULT_SYNTHETIC_ROWS, generate_synthetic_media_mix, validate_media_mix_frame
from utils.optimizer import (
    OptimizerConfig,
    allocate_budget,
    build_allocation_action_plan,
    summarize_allocation,
    summarize_goal_fit,
)


GOAL_PROFILES = {
    "Profitable growth": {
        "target_roas": 2.0,
        "target_cpa": 80,
        "risk_tolerance": 0.55,
        "help": "Balance profit, CPA, and channel risk.",
    },
    "Acquire customers efficiently": {
        "target_roas": 1.5,
        "target_cpa": 65,
        "risk_tolerance": 0.45,
        "help": "Prioritize cost control and conversion volume.",
    },
    "Scale winners": {
        "target_roas": 1.8,
        "target_cpa": 95,
        "risk_tolerance": 0.70,
        "help": "Move more budget into high-marginal-return channels.",
    },
    "Reduce waste": {
        "target_roas": 2.4,
        "target_cpa": 70,
        "risk_tolerance": 0.35,
        "help": "Tighten guardrails and expose inefficient spend.",
    },
}

PLOTLY_CONFIG = {"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d"]}
COLOR_SEQUENCE = ["#2563eb", "#059669", "#f97316", "#7c3aed", "#dc2626", "#0891b2"]


@st.cache_data(show_spinner=True)
def _cached_synthetic_media_mix(rows: int = DEFAULT_SYNTHETIC_ROWS) -> pd.DataFrame:
    return generate_synthetic_media_mix(rows=rows)


def render() -> None:
    st.subheader("Media Mix Budget Optimizer")
    st.caption("Plan budget around a business goal, target guardrails, risk tolerance, and expected channel tradeoffs.")

    uploaded = st.file_uploader("Optional campaign CSV override", type=["csv"], help="Upload real campaign data to replace the synthetic demo dataset for this session.")
    if uploaded is not None:
        source = normalize_uploaded_campaign_csv(pd.read_csv(uploaded))
        st.success(f"Using uploaded campaign data: {len(source):,} rows")
    else:
        source = _cached_synthetic_media_mix()
        st.info(f"Using synthetic optimizer data: {len(source):,} rows")

    errors = validate_media_mix_frame(source) if "marginal_roas" in source.columns else []
    if errors:
        st.error("The optimizer dataset failed validation: " + "; ".join(errors))
        return

    config = _render_controls(source)
    allocation = allocate_budget(source, config)
    if allocation.empty:
        st.warning("No eligible platforms remain after the current filters.")
        return

    summary = summarize_allocation(allocation)
    goal_fit = summarize_goal_fit(allocation, config)
    action_plan = build_allocation_action_plan(allocation, config)

    _render_decision_brief(summary, goal_fit, action_plan)

    plan_tab, tradeoff_tab, data_tab = st.tabs(["Allocation Plan", "Tradeoff Lens", "Data Readiness"])
    with plan_tab:
        _render_allocation_charts(allocation, action_plan)
    with tradeoff_tab:
        _render_tradeoff_lens(action_plan)
    with data_tab:
        _render_connector_specs()


def _render_controls(source: pd.DataFrame) -> OptimizerConfig:
    with st.container(border=True):
        st.markdown("#### Business Scenario")
        c1, c2, c3 = st.columns([1.4, 1, 1])
        business_goal = c1.selectbox(
            "Business goal",
            list(GOAL_PROFILES),
            index=0,
            help="Sets the default planning posture. Targets remain editable.",
        )
        profile = GOAL_PROFILES[business_goal]
        c1.caption(profile["help"])
        total_budget = c2.slider("Scenario budget", min_value=10_000, max_value=2_000_000, value=250_000, step=10_000)
        planning_horizon = c3.slider("Planning horizon", min_value=1, max_value=13, value=4, step=1)

        c4, c5, c6 = st.columns(3)
        target_roas = c4.slider("Target ROAS", min_value=0.5, max_value=6.0, value=float(profile["target_roas"]), step=0.1)
        target_cpa = c5.slider("Target CPA", min_value=10, max_value=500, value=int(profile["target_cpa"]), step=5)
        risk_tolerance = c6.slider(
            "Risk tolerance",
            min_value=0.0,
            max_value=1.0,
            value=float(profile["risk_tolerance"]),
            step=0.05,
            help="Higher values allow more volatile channels to receive budget.",
        )

        c7, c8 = st.columns(2)
        objectives = ["All"] + sorted(source["objective"].dropna().astype(str).unique().tolist())
        objective = c7.selectbox("Objective focus", objectives, index=0)
        excluded = c8.multiselect("Exclude platforms", sorted(source["platform"].dropna().astype(str).unique().tolist()))

    return OptimizerConfig(
        total_budget=float(total_budget),
        target_roas=float(target_roas),
        target_cpa=float(target_cpa),
        risk_tolerance=float(risk_tolerance),
        objective=None if objective == "All" else str(objective),
        excluded_platforms=tuple(excluded),
        business_goal=str(business_goal),
        planning_horizon_weeks=int(planning_horizon),
    )


def _render_decision_brief(
    summary: dict[str, float],
    goal_fit: dict[str, object],
    action_plan: pd.DataFrame,
) -> None:
    st.markdown("#### Decision Brief")
    status = str(goal_fit["status"])
    if status == "On track":
        st.success(str(goal_fit["summary"]))
    elif status == "Needs guardrails":
        st.warning(str(goal_fit["summary"]))
    else:
        st.error(str(goal_fit["summary"]))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Allocated budget", _currency(summary["budget"]))
    c2.metric("Expected profit", _currency(summary["expected_profit"]))
    c3.metric("Expected ROAS", f"{summary['expected_roas']:.2f}x", delta=f"{float(goal_fit['target_roas_gap']):+.2f}x vs target")
    c4.metric("Expected CPA", _currency(summary["expected_cpa"]), delta=f"{_currency(float(goal_fit['target_cpa_gap']))} target buffer")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Expected conversions", f"{summary['expected_conversions']:,.0f}")
    c6.metric("Budget to fund", _currency(float(goal_fit["budget_increase"])), delta=f"{float(goal_fit['increase_share']):.0%} of budget")
    c7.metric("Budget to limit", _currency(float(goal_fit["budget_limit"])), delta=f"{float(goal_fit['limit_share']):.0%} of budget", delta_color="inverse")
    c8.metric("Top platform", str(goal_fit["top_platform"]), delta=f"{float(goal_fit['top_platform_share']):.0%} allocation")

    st.caption(str(goal_fit["primary_constraint"]))
    _render_priority_moves(action_plan.head(3))


def _render_priority_moves(action_plan: pd.DataFrame) -> None:
    if action_plan.empty:
        return
    st.markdown("#### Priority Moves")
    columns = st.columns(len(action_plan))
    for column, (_, row) in zip(columns, action_plan.iterrows()):
        with column.container(border=True):
            st.markdown(f"**{int(row['priority'])}. {row['platform']}**")
            st.metric(str(row["decision"]), _currency(float(row["allocation"])), delta=f"{float(row['recommended_shift']):+.1%} vs current mix")
            st.caption(str(row["next_step"]))


def _render_allocation_charts(allocation: pd.DataFrame, action_plan: pd.DataFrame) -> None:
    st.markdown("#### Recommended Allocation")
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        px.bar(
            allocation,
            x="platform",
            y="allocation",
            color="action",
            color_discrete_sequence=COLOR_SEQUENCE,
            title="Budget allocation by platform",
            labels={"platform": "", "allocation": "Budget", "action": "Action"},
        ),
        config=PLOTLY_CONFIG,
    )
    c2.plotly_chart(
        px.scatter(
            allocation,
            x="expected_cpa",
            y="expected_roas",
            size="allocation",
            color="platform",
            hover_name="action",
            color_discrete_sequence=COLOR_SEQUENCE,
            title="Efficiency frontier",
            labels={"expected_cpa": "Expected CPA", "expected_roas": "Expected ROAS", "allocation": "Budget"},
        ),
        config=PLOTLY_CONFIG,
    )

    display = action_plan[
        [
            "priority",
            "platform",
            "decision",
            "allocation",
            "allocation_share",
            "recommended_shift",
            "expected_revenue",
            "expected_profit",
            "expected_roas",
            "expected_cpa",
            "marginal_roas",
            "risk_index",
            "saturation_index",
            "business_reason",
            "next_step",
        ]
    ].copy()
    display = _format_action_plan(display)
    st.dataframe(display, width="stretch", hide_index=True)
    st.download_button(
        "Download allocation action plan",
        data=action_plan.to_csv(index=False),
        file_name="budget_optimizer_action_plan.csv",
        mime="text/csv",
    )


def _render_tradeoff_lens(action_plan: pd.DataFrame) -> None:
    st.markdown("#### Tradeoff Lens")
    if action_plan.empty:
        st.info("No tradeoff data is available for this scenario.")
        return

    mix = action_plan[["platform", "historical_spend_share", "allocation_share"]].melt(
        id_vars="platform",
        var_name="mix",
        value_name="share",
    )
    mix["mix"] = mix["mix"].map(
        {
            "historical_spend_share": "Current spend mix",
            "allocation_share": "Recommended mix",
        }
    )

    c1, c2 = st.columns(2)
    c1.plotly_chart(
        px.bar(
            mix,
            x="platform",
            y="share",
            color="mix",
            barmode="group",
            color_discrete_sequence=COLOR_SEQUENCE,
            title="Current vs recommended mix",
            labels={"platform": "", "share": "Budget share", "mix": ""},
        ).update_yaxes(tickformat=".0%"),
        config=PLOTLY_CONFIG,
    )
    c2.plotly_chart(
        px.scatter(
            action_plan,
            x="risk_index",
            y="marginal_roas",
            size="allocation",
            color="decision",
            hover_name="platform",
            color_discrete_sequence=COLOR_SEQUENCE,
            title="Risk vs marginal return",
            labels={"risk_index": "Risk index", "marginal_roas": "Marginal ROAS", "allocation": "Budget"},
        ),
        config=PLOTLY_CONFIG,
    )

    shifts = action_plan.sort_values("recommended_shift", ascending=False)
    fig = px.bar(
        shifts,
        x="recommended_shift",
        y="platform",
        color="shift_direction",
        orientation="h",
        color_discrete_sequence=COLOR_SEQUENCE,
        title="Recommended shift from current mix",
        labels={"recommended_shift": "Share-point shift", "platform": "", "shift_direction": ""},
    )
    fig.update_xaxes(tickformat="+.0%")
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, config=PLOTLY_CONFIG)


def _render_connector_specs() -> None:
    with st.expander("Future real-data connectors", expanded=False):
        rows = [
            {
                "Connector": spec.name,
                "Status": spec.status,
                "Required secrets": ", ".join(spec.required_secrets) or "None",
                "Notes": spec.notes,
            }
            for spec in available_connector_specs()
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _currency(value: float) -> str:
    return f"${value:,.0f}"


def _format_action_plan(display: pd.DataFrame) -> pd.DataFrame:
    formatted = display.copy()
    for column in ["allocation", "expected_revenue", "expected_profit", "expected_cpa"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(_currency)
    for column in ["allocation_share", "recommended_shift"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value * 100:+.1f}%" if column == "recommended_shift" else f"{value * 100:.1f}%")
    for column in ["expected_roas", "marginal_roas"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.2f}x")
    for column in ["risk_index", "saturation_index"]:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.2f}")
    return formatted
