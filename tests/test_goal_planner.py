from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_sources.campaign_data import PLATFORM_CREATIVE_FORMATS
from data_sources.synthetic_media_mix import generate_synthetic_media_mix
from utils.goal_planner import (
    GOAL_DEFINITIONS,
    budget_sweep,
    compute_platform_tilts,
    list_goal_labels,
    plan_for_goal,
    platform_response_curves,
    summarize_goal_plan,
)


def _frame() -> pd.DataFrame:
    return generate_synthetic_media_mix(rows=4_000, seed=21)


@pytest.mark.parametrize("goal_key", list(GOAL_DEFINITIONS.keys()))
def test_plan_for_goal_is_internally_consistent(goal_key: str) -> None:
    frame = _frame()
    plan = plan_for_goal(frame, total_budget=200_000, goal_key=goal_key)

    assert not plan.allocation.empty
    assert round(float(plan.allocation["allocation"].sum()), 2) == 200_000.00
    assert plan.allocation["expected_metric"].ge(0).all()
    assert plan.allocation["expected_cost_per_metric"].ge(0).all()
    assert round(float(plan.allocation["allocation_share"].sum()), 6) == 1.0
    assert set(plan.allocation["platform"]).issubset(set(frame["platform"]))
    assert plan.allocation["recommended_campaign_type"].ne("—").any()
    assert plan.allocation["recommended_audience"].ne("—").any()


def test_goal_metric_columns_all_exist_in_dataset() -> None:
    frame = _frame()
    for spec in GOAL_DEFINITIONS.values():
        assert spec.metric_column in frame.columns


def test_plan_summary_matches_allocation_totals() -> None:
    frame = _frame()
    plan = plan_for_goal(frame, total_budget=150_000, goal_key="conversions")
    summary = plan.summary

    assert round(summary["budget"], 2) == 150_000.00
    assert round(summary["expected_metric"], 4) == round(float(plan.allocation["expected_metric"].sum()), 4)
    expected_blended = summary["budget"] / summary["expected_metric"] if summary["expected_metric"] else 0.0
    assert round(summary["blended_cost_per_metric"], 6) == round(expected_blended, 6)
    assert summary["platforms"] == len(plan.allocation)


def test_summarize_goal_plan_handles_empty() -> None:
    empty = summarize_goal_plan(pd.DataFrame(columns=["allocation", "expected_metric"]), GOAL_DEFINITIONS["revenue"])
    assert empty["budget"] == 0.0
    assert empty["expected_metric"] == 0.0


def test_affinity_filter_relaxes_when_data_is_thin() -> None:
    # A frame whose only objective is unrelated to the Leads goal forces the soft-filter fallback.
    frame = _frame().copy()
    frame["objective"] = "Awareness"
    plan = plan_for_goal(frame, total_budget=100_000, goal_key="leads")

    assert plan.filter_relaxed
    assert not plan.allocation.empty


def test_market_filter_can_empty_plan_gracefully() -> None:
    frame = _frame()
    plan = plan_for_goal(frame, total_budget=100_000, goal_key="traffic", market="Atlantis")

    assert plan.allocation.empty
    assert plan.summary["expected_metric"] == 0.0


def test_invalid_inputs_raise() -> None:
    frame = _frame()
    with pytest.raises(ValueError):
        plan_for_goal(frame, total_budget=0, goal_key="revenue")
    with pytest.raises(ValueError):
        plan_for_goal(frame, total_budget=1_000, goal_key="not_a_goal")


def test_list_goal_labels_round_trips_to_keys() -> None:
    labels = list_goal_labels()
    for label, key in labels.items():
        assert GOAL_DEFINITIONS[key].label == label


def test_same_metric_goals_now_produce_different_plans() -> None:
    frame = _frame()
    conv = plan_for_goal(frame, total_budget=30_000, goal_key="conversions")
    leads = plan_for_goal(frame, total_budget=30_000, goal_key="leads")
    installs = plan_for_goal(frame, total_budget=30_000, goal_key="app_installs")

    conv_shares = conv.allocation.set_index("platform")["allocation_share"]
    lead_shares = leads.allocation.set_index("platform")["allocation_share"]
    install_shares = installs.allocation.set_index("platform")["allocation_share"]
    # Strict affinity + platform-objective fit should materially separate the share vectors.
    assert not conv_shares.round(3).equals(lead_shares.reindex(conv_shares.index).round(3))
    assert not lead_shares.round(3).equals(install_shares.reindex(lead_shares.index).round(3))


def test_recommended_creative_is_valid_for_platform() -> None:
    frame = _frame()
    for goal_key in GOAL_DEFINITIONS:
        plan = plan_for_goal(frame, total_budget=30_000, goal_key=goal_key)
        for _, row in plan.allocation.iterrows():
            creative = row["recommended_campaign_type"]
            if creative == "—":
                continue
            assert creative in PLATFORM_CREATIVE_FORMATS[row["platform"]], (
                f"{goal_key}: {creative} invalid for {row['platform']}"
            )


def test_upper_funnel_goals_never_recommend_retention_audiences() -> None:
    frame = _frame()
    for goal_key in ("awareness", "engagement", "traffic"):
        plan = plan_for_goal(frame, total_budget=30_000, goal_key=goal_key)
        for audience in plan.allocation["recommended_audience"]:
            for banned in ("Retargeting", "Lapsed Customer", "Lifecycle"):
                assert banned not in audience, f"{goal_key} recommended {audience}"


def test_min_allocation_floor_concentrates_budget() -> None:
    frame = _frame()
    plan = plan_for_goal(frame, total_budget=30_000, goal_key="conversions", min_allocation=2_500)

    assert plan.allocation["allocation"].ge(2_500).all()
    assert round(float(plan.allocation["allocation"].sum()), 2) == 30_000.00
    unfloored = plan_for_goal(frame, total_budget=30_000, goal_key="conversions")
    assert len(plan.allocation) < len(unfloored.allocation)


def test_max_platforms_renormalizes_to_full_budget() -> None:
    frame = _frame()
    plan = plan_for_goal(frame, total_budget=30_000, goal_key="conversions", max_platforms=3)

    assert len(plan.allocation) == 3
    assert round(float(plan.allocation["allocation"].sum()), 2) == 30_000.00


def test_platform_tilts_shift_allocation() -> None:
    frame = _frame()
    base = plan_for_goal(frame, total_budget=30_000, goal_key="conversions")
    target = base.allocation.iloc[-1]["platform"]
    tilted = plan_for_goal(frame, total_budget=30_000, goal_key="conversions", platform_tilts={target: 2.0})

    base_share = float(base.allocation.set_index("platform").loc[target, "allocation_share"])
    tilted_share = float(tilted.allocation.set_index("platform").loc[target, "allocation_share"])
    assert tilted_share > base_share
    assert round(float(tilted.allocation["allocation"].sum()), 2) == 30_000.00


def test_platform_response_curves_show_diminishing_returns() -> None:
    frame = _frame()
    plan = plan_for_goal(frame, total_budget=30_000, goal_key="conversions")
    curves = platform_response_curves(plan.allocation)

    for platform, group in curves.groupby("platform"):
        values = group.sort_values("spend")["expected_metric"].to_numpy()
        assert (np.diff(values) >= -1e-9).all(), f"{platform} curve not monotonic"
        gains = np.diff(values)
        # Marginal gain should shrink: late increments buy less than early ones.
        assert gains[-1] <= gains[1] + 1e-9, f"{platform} shows no diminishing returns"


def test_budget_sweep_marginal_cost_rises() -> None:
    frame = _frame()
    sweep = budget_sweep(frame, "conversions", base_budget=30_000)

    assert sweep["expected_metric"].is_monotonic_increasing
    marginal = sweep["marginal_cost_per_metric"].iloc[1:]
    assert marginal.iloc[-1] >= marginal.iloc[0]
    at_base = sweep[sweep["budget"] == 30_000].iloc[0]
    plan = plan_for_goal(frame, total_budget=30_000, goal_key="conversions")
    assert round(float(at_base["expected_metric"]), 2) == round(plan.summary["expected_metric"], 2)


def test_compute_platform_tilts_favors_whitespace() -> None:
    items = pd.DataFrame(
        {
            "source": ["Meta Ad Library"] * 8 + ["TikTok Creative Center"] * 1 + ["GDELT"] * 5,
            "title": ["x"] * 14,
        }
    )
    tilts = compute_platform_tilts(items)

    assert tilts["TikTok"] > 1.0  # under-contested
    assert tilts["Meta"] < 1.0  # crowded
    assert "GDELT" not in tilts and len(tilts) == 2

    assert compute_platform_tilts(pd.DataFrame()) == {}
