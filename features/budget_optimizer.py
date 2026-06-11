from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data_sources.connectors import available_connector_specs, normalize_uploaded_campaign_csv
from data_sources.synthetic_media_mix import DEFAULT_SYNTHETIC_ROWS, generate_synthetic_media_mix, validate_media_mix_frame
from utils.goal_planner import (
    GOAL_DEFINITIONS,
    GoalPlan,
    GoalSpec,
    budget_sweep,
    compute_platform_tilts,
    list_goal_labels,
    plan_for_goal,
    platform_response_curves,
)
from utils.optimizer import OptimizerConfig, allocate_budget, summarize_allocation


@st.cache_data(show_spinner=True)
def _cached_synthetic_media_mix(rows: int = DEFAULT_SYNTHETIC_ROWS) -> pd.DataFrame:
    return generate_synthetic_media_mix(rows=rows)


def render() -> None:
    st.subheader("Media Mix Budget Optimizer")
    st.caption("Uses a 250k-row synthetic media-mix dataset by default, with a CSV normalization path and future API connector stubs.")

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

    planner_tab, advanced_tab, connectors_tab = st.tabs(
        ["Goal Planner", "Advanced (ROAS/CPA)", "Connectors"]
    )
    with planner_tab:
        _render_goal_planner(source)
    with advanced_tab:
        _render_advanced(source)
    with connectors_tab:
        _render_connector_specs()


def _render_goal_planner(source: pd.DataFrame) -> None:
    st.markdown("#### Goal-Driven Media Plan")
    st.caption(
        "Tell the planner your budget and what you're optimizing for. It recommends which platforms "
        "to use, the campaign type and audience for each, how to split the budget, and the expected "
        "cost-per-goal and goal volume."
    )

    labels = list_goal_labels()
    c1, c2 = st.columns([1, 1])
    total_budget = c1.number_input(
        "Total budget ($)",
        min_value=1_000,
        max_value=10_000_000,
        value=250_000,
        step=10_000,
        help="The total media budget to allocate across platforms for this plan.",
    )
    goal_label = c2.selectbox(
        "Primary goal (what to maximize)",
        list(labels.keys()),
        index=0,
        help=(
            "The end metric you care about. The planner ranks platforms by how efficiently they "
            "deliver this goal (goal per dollar), not by generic ROAS."
        ),
    )
    goal_key = labels[goal_label]

    c3, c4 = st.columns(2)
    markets = ["All"] + sorted(source["geo"].dropna().astype(str).unique().tolist()) if "geo" in source.columns else ["All"]
    industries = ["All"] + sorted(source["industry"].dropna().astype(str).unique().tolist()) if "industry" in source.columns else ["All"]
    market = c3.selectbox("Market / geo", markets, index=0, help="Restrict the plan to one market, or keep All to plan globally.")
    industry = c4.selectbox("Industry", industries, index=0, help="Restrict benchmarks to one industry vertical, or keep All.")

    c5, c6 = st.columns(2)
    min_allocation = c5.number_input(
        "Min spend per platform ($)",
        min_value=0,
        max_value=100_000,
        value=2_500,
        step=500,
        help=(
            "Platforms that would receive less than this are dropped and their budget redistributed. "
            "Real campaigns need a viable minimum — a $500 line item can't exit the learning phase "
            "or produce statistically meaningful results."
        ),
    )
    max_platforms_label = c6.selectbox(
        "Max platforms",
        ["Auto", "2", "3", "4", "5", "6"],
        index=0,
        help="Hard-cap how many channels the plan may use. Auto lets the min-spend floor decide.",
    )
    max_platforms = None if max_platforms_label == "Auto" else int(max_platforms_label)

    goal = GOAL_DEFINITIONS[goal_key]
    if goal.is_proxy and goal.note:
        st.info(f"ℹ️ {goal.note}")

    platform_tilts = _render_strategic_context()

    plan = plan_for_goal(
        source,
        float(total_budget),
        goal_key,
        market=market,
        industry=industry,
        max_platforms=max_platforms,
        min_allocation=float(min_allocation),
        platform_tilts=platform_tilts,
    )
    if plan.allocation.empty:
        st.warning("No eligible platforms remain for this goal and filter combination. Broaden the market/industry filters.")
        return
    if plan.filter_relaxed:
        st.caption(
            f"Note: too little goal-aligned data for {goal.label.lower()}, so the plan uses all "
            "campaign objectives for these platforms rather than only the goal's preferred objective."
        )

    _render_goal_summary(plan.summary, goal)
    _render_goal_charts(plan.allocation, goal)
    _render_goal_recommendations(plan.allocation, goal)
    _render_returns_sensitivity(source, plan, goal_key, float(total_budget), market, industry, max_platforms, float(min_allocation), platform_tilts)


def _render_advanced(source: pd.DataFrame) -> None:
    st.markdown("#### Advanced ROAS / CPA Optimizer")
    st.caption("The original scenario optimizer: tune target ROAS, target CPA, and risk tolerance directly.")
    config = _render_controls(source)
    allocation = allocate_budget(source, config)
    if allocation.empty:
        st.warning("No eligible platforms remain after the current filters.")
        return

    summary = summarize_allocation(allocation)
    _render_summary(summary)
    _render_allocation_charts(allocation)


def _render_strategic_context() -> dict[str, float] | None:
    """Surface demand/competitor signals from the other tabs and offer a whitespace tilt."""
    demand = st.session_state.get("demand_context")
    competitor = st.session_state.get("competitor_context")
    tilts: dict[str, float] | None = None

    with st.expander("Strategic context from Demand Pulse & Competitor Intelligence", expanded=False):
        if not demand and not competitor:
            st.caption(
                "No live context yet. Refresh the Demand Pulse and Competitor Intelligence tabs in "
                "this session and their signals (demand momentum, competitor pressure) will appear "
                "here to inform the plan."
            )
            return None

        if demand:
            momentum = demand.get("momentum")
            if momentum is not None and not momentum.empty:
                st.markdown("**Demand angles to brief into creative** (from Demand Pulse)")
                display = momentum[["keyword", "lifecycle", "momentum"]].copy()
                display["momentum"] = display["momentum"].map(lambda v: f"{v * 100:+.0f}%")
                st.dataframe(display, use_container_width=True, hide_index=True)

        if competitor:
            items = competitor.get("items")
            candidate = compute_platform_tilts(items) if items is not None else {}
            if candidate:
                st.markdown("**Competitor pressure by ad platform** (from Competitor Intelligence)")
                pressure = pd.DataFrame(
                    [{"platform": p, "tilt": t, "read": "under-contested" if t > 1 else "crowded"} for p, t in candidate.items()]
                )
                st.dataframe(pressure.round(3), use_container_width=True, hide_index=True)
                if st.checkbox(
                    "Tilt allocation toward less-contested platforms",
                    value=False,
                    help=(
                        "Multiplies each platform's score by its whitespace tilt: platforms where "
                        "competitors are under-represented get a boost, crowded ones a small haircut."
                    ),
                ):
                    tilts = candidate
            else:
                st.caption("Competitor items could not be attributed to ad platforms (no Meta/TikTok/YouTube/Reddit sources).")
    return tilts


def _render_returns_sensitivity(
    source: pd.DataFrame,
    plan: GoalPlan,
    goal_key: str,
    total_budget: float,
    market: str,
    industry: str,
    max_platforms: int | None,
    min_allocation: float,
    platform_tilts: dict[str, float] | None,
) -> None:
    goal = plan.goal
    st.markdown("##### Returns & Sensitivity")
    st.caption(
        "Answers: where does each platform saturate, and what would more (or less) total budget "
        "actually buy? Flattening curves mean extra dollars stop converting proportionally."
    )

    curves = platform_response_curves(plan.allocation)
    sweep = budget_sweep(
        source,
        goal_key,
        total_budget,
        market=market,
        industry=industry,
        max_platforms=max_platforms,
        min_allocation=min_allocation,
        platform_tilts=platform_tilts,
    )

    c1, c2 = st.columns(2)
    if not curves.empty:
        c1.plotly_chart(
            px.line(
                curves,
                x="spend",
                y="expected_metric",
                color="platform",
                labels={"spend": "Spend ($)", "expected_metric": f"Expected {goal.volume_label.lower()}"},
                title="Response curves (0 → 2× planned spend per platform)",
            ),
            use_container_width=True,
        )
    if not sweep.empty:
        c2.plotly_chart(
            px.line(
                sweep,
                x="budget",
                y="expected_metric",
                markers=True,
                labels={"budget": "Total budget ($)", "expected_metric": f"Expected {goal.volume_label.lower()}"},
                title="Budget sweep (0.25× → 2× current budget)",
            ),
            use_container_width=True,
        )

    next_step = sweep[sweep["budget"] > total_budget].head(1)
    if not next_step.empty:
        row = next_step.iloc[0]
        extra_budget = float(row["budget"]) - total_budget
        current = sweep[sweep["budget"] == sweep[sweep["budget"] <= total_budget]["budget"].max()]
        extra_metric = float(row["expected_metric"]) - (float(current.iloc[0]["expected_metric"]) if not current.empty else 0.0)
        if extra_metric > 0:
            st.metric(
                f"Next +{_currency(extra_budget)} of budget buys",
                f"~{_format_metric(extra_metric, goal)} {goal.volume_label.lower()}",
                f"marginal {goal.cost_label.lower()}: {_format_cost(extra_budget / extra_metric, goal)}",
                delta_color="off",
            )


def _render_goal_summary(summary: dict[str, float], goal: GoalSpec) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Total budget", _currency(summary["budget"]))
    c2.metric(f"Expected {goal.volume_label.lower()}", _format_metric(summary["expected_metric"], goal))
    c3.metric(goal.cost_label, _format_cost(summary["blended_cost_per_metric"], goal))


def _render_goal_charts(allocation: pd.DataFrame, goal: GoalSpec) -> None:
    st.markdown("##### Recommended Allocation")
    chart = allocation.copy()
    chart["display_cost"] = chart["expected_cost_per_metric"].map(lambda v: _cost_value(v, goal))
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        px.bar(chart, x="platform", y="allocation", color="recommended_campaign_type", title="Budget by platform (color = recommended campaign type)"),
        use_container_width=True,
    )
    c2.plotly_chart(
        px.scatter(
            chart,
            x="display_cost",
            y="expected_metric",
            size="allocation",
            color="platform",
            hover_name="platform",
            labels={"display_cost": goal.cost_label, "expected_metric": f"Expected {goal.volume_label.lower()}"},
            title="Efficiency: cost-per-goal vs expected volume",
        ),
        use_container_width=True,
    )


def _render_goal_recommendations(allocation: pd.DataFrame, goal: GoalSpec) -> None:
    st.markdown("##### Per-Platform Playbook")
    for _, row in allocation.iterrows():
        st.markdown(
            f"**{row['platform']}** — {_currency(row['allocation'])} "
            f"({row['allocation_share'] * 100:.0f}% of budget)"
        )
        st.caption(
            f"Run **{row['recommended_campaign_type']}** to **{row['recommended_audience']}** audiences → "
            f"~{_format_metric(row['expected_metric'], goal)} {goal.volume_label.lower()} "
            f"at ~{_format_cost(row['expected_cost_per_metric'], goal)} each."
        )

    display = allocation.copy()
    display["allocation"] = display["allocation"].map(_currency)
    display["allocation_share"] = display["allocation_share"].map(lambda v: f"{v * 100:.1f}%")
    display["expected_metric"] = display["expected_metric"].map(lambda v: _format_metric(v, goal))
    display["expected_cost_per_metric"] = display["expected_cost_per_metric"].map(lambda v: _format_cost(v, goal))
    display = display[
        [
            "platform",
            "allocation",
            "allocation_share",
            "recommended_campaign_type",
            "recommended_audience",
            "expected_metric",
            "expected_cost_per_metric",
        ]
    ].rename(
        columns={
            "allocation": "Budget",
            "allocation_share": "Share",
            "recommended_campaign_type": "Campaign type",
            "recommended_audience": "Audience",
            "expected_metric": f"Expected {goal.volume_label.lower()}",
            "expected_cost_per_metric": goal.cost_label,
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def _cost_value(value: float, goal: GoalSpec) -> float:
    # Awareness is conventionally priced per 1,000 impressions (CPM), not per single impression.
    return value * 1000 if goal.key == "awareness" else value


def _format_cost(value: float, goal: GoalSpec) -> str:
    return _currency(_cost_value(value, goal)) if goal.key == "awareness" else f"${value:,.2f}"


def _format_metric(value: float, goal: GoalSpec) -> str:
    if goal.is_currency:
        return _currency(value)
    return f"{value:,.0f}"


def _render_controls(source: pd.DataFrame) -> OptimizerConfig:
    st.markdown("#### Scenario Controls")
    c1, c2, c3, c4 = st.columns(4)
    total_budget = c1.slider("Budget", min_value=10_000, max_value=2_000_000, value=250_000, step=10_000)
    target_roas = c2.slider("Target ROAS", min_value=0.5, max_value=6.0, value=2.0, step=0.1)
    target_cpa = c3.slider("Target CPA", min_value=10, max_value=500, value=80, step=5)
    risk_tolerance = c4.slider("Risk tolerance", min_value=0.0, max_value=1.0, value=0.55, step=0.05)

    c5, c6 = st.columns(2)
    objectives = ["All"] + sorted(source["objective"].dropna().astype(str).unique().tolist())
    objective = c5.selectbox("Objective", objectives, index=0)
    excluded = c6.multiselect("Exclude platforms", sorted(source["platform"].dropna().astype(str).unique().tolist()))

    return OptimizerConfig(
        total_budget=float(total_budget),
        target_roas=float(target_roas),
        target_cpa=float(target_cpa),
        risk_tolerance=float(risk_tolerance),
        objective=None if objective == "All" else str(objective),
        excluded_platforms=tuple(excluded),
    )


def _render_summary(summary: dict[str, float]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Allocated budget", _currency(summary["budget"]))
    c2.metric("Expected revenue", _currency(summary["expected_revenue"]))
    c3.metric("Expected profit", _currency(summary["expected_profit"]))
    c4.metric("Expected ROAS", f"{summary['expected_roas']:.2f}x")
    c5, c6 = st.columns(2)
    c5.metric("Expected conversions", f"{summary['expected_conversions']:,.0f}")
    c6.metric("Expected CPA", _currency(summary["expected_cpa"]))


def _render_allocation_charts(allocation: pd.DataFrame) -> None:
    st.markdown("#### Recommended Allocation")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(allocation, x="platform", y="allocation", color="action", title="Budget allocation by platform"), use_container_width=True)
    c2.plotly_chart(px.scatter(allocation, x="expected_cpa", y="expected_roas", size="allocation", color="platform", hover_name="action", title="Efficiency frontier"), use_container_width=True)

    display = allocation[
        [
            "platform",
            "allocation",
            "allocation_share",
            "expected_revenue",
            "expected_profit",
            "expected_roas",
            "expected_cpa",
            "marginal_roas",
            "risk_index",
            "saturation_index",
            "action",
        ]
    ].copy()
    display["allocation_share"] = display["allocation_share"].map(lambda value: f"{value * 100:.1f}%")
    st.dataframe(display, use_container_width=True, hide_index=True)


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
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _currency(value: float) -> str:
    return f"${value:,.0f}"
