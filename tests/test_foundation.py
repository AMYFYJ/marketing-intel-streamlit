from __future__ import annotations

import importlib

import pandas as pd


def test_app_modules_import() -> None:
    modules = [
        "streamlit_app",
        "features.performance_dashboard",
        "features.competitor_intelligence",
        "features.demand_pulse",
        "features.budget_optimizer",
        "utils.layout",
    ]
    for module in modules:
        assert importlib.import_module(module)


def test_frontier_cpc_axis_keeps_cents() -> None:
    performance_dashboard = importlib.import_module("features.performance_dashboard")

    assert performance_dashboard._axis_tickformat("cpc") == "$,.2f"
    assert performance_dashboard._axis_tickformat("cpa") == "$,.0f"


def test_performance_diagnostic_helpers_shape_ranked_outputs() -> None:
    performance_dashboard = importlib.import_module("features.performance_dashboard")
    campaign_data = importlib.import_module("data_sources.campaign_data")
    frame = campaign_data.generate_campaign_sample(rows=500, seed=21)

    leaderboard = performance_dashboard._ranked_segments(frame, "platform", "cpa", 5)
    assert len(leaderboard) <= 5
    assert leaderboard["conversions"].gt(0).all()
    assert leaderboard["cpa"].is_monotonic_increasing

    funnel = performance_dashboard._funnel_leakage_matrix(frame, "platform", 5)
    assert list(funnel.columns) == [
        "Reach / Impr.",
        "Click / Impr.",
        "LPV / Click",
        "Cart / LPV",
        "Conv. / Cart",
        "Conv. / Click",
        "Conv. / Impr.",
    ]
    assert len(funnel) <= 5

    normalized = performance_dashboard._normalize_rate_matrix(funnel)
    assert normalized.max().le(1).all()

    help_html = performance_dashboard._funnel_metric_help_html()
    for label in funnel.columns:
        assert label in help_html


def test_top_movers_rank_by_absolute_delta() -> None:
    performance_dashboard = importlib.import_module("features.performance_dashboard")
    campaign_data = importlib.import_module("data_sources.campaign_data")
    frame = campaign_data.generate_campaign_sample(rows=1_000, seed=34)

    max_date = frame["date"].max()
    current = frame[frame["date"] > max_date - pd.Timedelta(days=90)]
    previous = frame[
        (frame["date"] <= max_date - pd.Timedelta(days=90))
        & (frame["date"] > max_date - pd.Timedelta(days=180))
    ]

    movers = performance_dashboard._top_movers(current, previous, "platform", "profit", 5)

    assert len(movers) <= 5
    assert movers["abs_delta"].is_monotonic_decreasing
