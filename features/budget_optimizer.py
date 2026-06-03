from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data_sources.connectors import available_connector_specs, normalize_uploaded_campaign_csv
from data_sources.synthetic_media_mix import DEFAULT_SYNTHETIC_ROWS, generate_synthetic_media_mix, validate_media_mix_frame
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

    config = _render_controls(source)
    allocation = allocate_budget(source, config)
    if allocation.empty:
        st.warning("No eligible platforms remain after the current filters.")
        return

    summary = summarize_allocation(allocation)
    _render_summary(summary)
    _render_allocation_charts(allocation)
    _render_connector_specs()


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
