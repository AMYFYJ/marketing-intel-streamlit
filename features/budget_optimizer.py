from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data_sources.connectors import available_connector_specs, normalize_uploaded_campaign_csv
from data_sources.synthetic_media_mix import DEFAULT_SYNTHETIC_ROWS, generate_synthetic_media_mix, validate_media_mix_frame
from utils.formatting import display_labels, format_currency, format_roas, title_case_columns
from utils.optimizer import OptimizerConfig, allocate_budget, summarize_allocation

ALLOCATION_COLUMN_CONFIG = {
    "Allocation": st.column_config.NumberColumn("Allocation", format="dollar"),
    "Allocation Share": st.column_config.NumberColumn("Allocation Share", format="percent"),
    "Expected Revenue": st.column_config.NumberColumn("Expected Revenue", format="dollar"),
    "Expected Profit": st.column_config.NumberColumn("Expected Profit", format="dollar"),
    "Expected ROAS": st.column_config.NumberColumn("Expected ROAS", format="%.2fx"),
    "Expected CPA": st.column_config.NumberColumn("Expected CPA", format="dollar"),
    "Marginal ROAS": st.column_config.NumberColumn("Marginal ROAS", format="%.2f"),
    "Risk Index": st.column_config.NumberColumn("Risk Index", format="%.2f"),
    "Saturation Index": st.column_config.NumberColumn("Saturation Index", format="%.2f"),
}


@st.cache_data(show_spinner=True)
def _cached_synthetic_media_mix(rows: int = DEFAULT_SYNTHETIC_ROWS) -> pd.DataFrame:
    return generate_synthetic_media_mix(rows=rows)


def render() -> None:
    st.subheader("Media Mix Budget Optimizer")
    st.caption(
        "Allocates budget greedily along per-channel diminishing-return curves: each increment goes to the "
        "channel with the highest risk-adjusted marginal ROAS, and stops where targets can no longer be met."
    )

    uploaded = st.file_uploader("Optional Campaign CSV Override", type=["csv"], help="Upload real campaign data to replace the synthetic demo dataset for this session.")
    if uploaded is not None:
        source = normalize_uploaded_campaign_csv(pd.read_csv(uploaded))
        st.success(f"Using uploaded campaign data: {len(source):,} rows")
        _render_curve_source_badge(source)
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
    _render_summary(summary, config)
    _render_allocation_charts(allocation)
    _render_connector_specs()


def _render_curve_source_badge(source: pd.DataFrame) -> None:
    if "curve_source" not in source.columns:
        return
    fitted = sorted(source.loc[source["curve_source"] == "fitted", "platform"].astype(str).unique())
    assumed = sorted(source.loc[source["curve_source"] == "assumed", "platform"].astype(str).unique())
    if fitted and not assumed:
        st.success(f"Marginal-return curves fitted from daily spend variation for: {', '.join(fitted)}.")
    elif fitted:
        st.info(
            f"Curves fitted for {', '.join(fitted)}; assumed defaults for {', '.join(assumed)} "
            "(0.65x ROAS marginal, flat saturation). Upload daily-grain history per platform for fitted curves."
        )
    else:
        st.warning(
            "Curve parameters are assumed (0.65x ROAS marginal, flat saturation) because the upload lacks "
            "daily spend variation. Upload several days of per-platform data for fitted curves."
        )


def _render_controls(source: pd.DataFrame) -> OptimizerConfig:
    st.markdown("#### Scenario Controls")
    c1, c2, c3, c4 = st.columns(4)
    total_budget = c1.slider("Budget", min_value=10_000, max_value=2_000_000, value=250_000, step=10_000)
    target_roas = c2.slider("Target ROAS (Marginal Floor)", min_value=0.5, max_value=6.0, value=2.0, step=0.1)
    target_cpa = c3.slider("Target CPA (Marginal Cap)", min_value=10, max_value=500, value=80, step=5)
    risk_tolerance = c4.slider("Risk Tolerance", min_value=0.0, max_value=1.0, value=0.55, step=0.05)

    c5, c6 = st.columns(2)
    objectives = ["All"] + sorted(source["objective"].dropna().astype(str).unique().tolist())
    objective = c5.selectbox("Objective", objectives, index=0)
    excluded = c6.multiselect("Exclude Platforms", sorted(source["platform"].dropna().astype(str).unique().tolist()))

    return OptimizerConfig(
        total_budget=float(total_budget),
        target_roas=float(target_roas),
        target_cpa=float(target_cpa),
        risk_tolerance=float(risk_tolerance),
        objective=None if objective == "All" else str(objective),
        excluded_platforms=tuple(excluded),
    )


def _render_summary(summary: dict[str, float], config: OptimizerConfig) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Allocated Budget", format_currency(summary["budget"]))
    c2.metric("Expected Revenue", format_currency(summary["expected_revenue"]))
    c3.metric("Expected Profit", format_currency(summary["expected_profit"]))
    c4.metric("Expected ROAS", format_roas(summary["expected_roas"]))
    c5, c6, c7 = st.columns(3)
    c5.metric("Expected Conversions", f"{summary['expected_conversions']:,.0f}")
    c6.metric("Expected CPA", format_currency(summary["expected_cpa"]))
    c7.metric("Unallocated Budget", format_currency(summary["unallocated"]))
    if summary["unallocated"] > 0:
        st.warning(
            f"{format_currency(summary['unallocated'])} of the budget stays unallocated: no channel's marginal "
            f"return meets the {config.target_roas:.1f}x ROAS floor and {format_currency(config.target_cpa)} CPA cap "
            "at higher spend. Relax the targets or accept the smaller plan."
        )


def _render_allocation_charts(allocation: pd.DataFrame) -> None:
    st.markdown("#### Recommended Allocation")
    c1, c2 = st.columns(2)
    c1.plotly_chart(
        px.bar(allocation, x="platform", y="allocation", color="action", title="Budget Allocation by Platform", labels=display_labels(["platform", "allocation", "action"])),
        use_container_width=True,
    )
    funded = allocation[allocation["allocation"] > 0]
    c2.plotly_chart(
        px.scatter(
            funded,
            x="expected_cpa",
            y="expected_roas",
            size="allocation",
            color="platform",
            hover_name="platform",
            hover_data={"allocation": ":,.0f", "action": True},
            title="Efficiency Frontier (Funded Channels)",
            labels=display_labels(["expected_cpa", "expected_roas", "allocation", "platform", "action"]),
        ),
        use_container_width=True,
    )

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
    ]
    st.dataframe(title_case_columns(display), use_container_width=True, hide_index=True, column_config=ALLOCATION_COLUMN_CONFIG)
    st.caption(
        "Model: each channel follows a diminishing-return curve anchored at its historical spend share, with "
        "saturation depth from the data's diminishing-return index. Actions compare the recommended share with "
        "the historical share; 'Limit (Target)' means the ROAS/CPA targets cap that channel's spend."
    )


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
