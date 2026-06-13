from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.campaign_data import load_campaign_data, summarize_metrics
from data_sources.competitor_sources import CompetitorQuery, fetch_competitor_intelligence
from data_sources.demand_metrics import (
    build_action_queue,
    compute_baseline_stats,
    compute_momentum,
    compute_signal_confidence,
)
from data_sources.synthetic_media_mix import generate_synthetic_media_mix, validate_media_mix_frame
from data_sources.trend_sources import TrendQuery, build_daily_series, fetch_demand_pulse
from utils.goal_planner import GOAL_DEFINITIONS, budget_sweep, compute_platform_tilts, plan_for_goal
from utils.optimizer import OptimizerConfig, allocate_budget, summarize_allocation


def main() -> None:
    campaign = load_campaign_data(fallback_rows=1_000)
    metrics = summarize_metrics(campaign)
    assert metrics["spend"] > 0

    synthetic = generate_synthetic_media_mix(rows=5_000)
    assert validate_media_mix_frame(synthetic) == []

    allocation = allocate_budget(synthetic, OptimizerConfig(total_budget=100_000))
    summary = summarize_allocation(allocation)
    assert round(summary["budget"] + summary["unallocated"], 2) == 100_000.00
    assert summary["budget"] > 0

    # Goal-driven planner runs offline across every supported goal.
    for goal_key in GOAL_DEFINITIONS:
        plan = plan_for_goal(synthetic, total_budget=100_000, goal_key=goal_key)
        assert round(float(plan.allocation["allocation"].sum()), 2) == 100_000.00
        assert plan.allocation["expected_metric"].ge(0).all()

    # Concentration floor, budget sweep, and competitor tilt all run offline.
    floored = plan_for_goal(synthetic, total_budget=30_000, goal_key="conversions", min_allocation=2_500)
    assert floored.allocation["allocation"].ge(2_500).all()
    sweep = budget_sweep(synthetic, "conversions", base_budget=30_000, multipliers=(0.5, 1.0, 2.0))
    assert sweep["expected_metric"].is_monotonic_increasing
    tilts = compute_platform_tilts(pd.DataFrame({"source": ["Meta Ad Library", "TikTok Creative Center"], "title": ["a", "b"]}))
    assert set(tilts) == {"Meta", "TikTok"}

    demand_items, demand_status = fetch_demand_pulse(
        TrendQuery(keywords=("AI marketing",), max_items_per_source=1),
        sources=("Google Trends export",),
        data_dir="data",
    )
    assert demand_items.empty
    assert not demand_status.empty

    # Demand-pulse analytics run offline from item timestamps (no network).
    pulse_items = pd.DataFrame(
        {
            "source": ["GDELT", "Reddit", "GDELT", "YouTube"],
            "keyword": ["AI marketing", "AI marketing", "retail media", "AI marketing"],
            "title": ["a", "b", "c", "d"],
            "published_at": pd.to_datetime(
                ["2026-06-09", "2026-06-09", "2026-06-08", "2026-06-07"], utc=True
            ),
            "sentiment": [1.0, -1.0, 0.0, 1.0],
        }
    )
    timeline = build_daily_series(pulse_items)
    momentum = compute_momentum(timeline)
    baseline = compute_baseline_stats(timeline, lookback_days=2)
    confidence = compute_signal_confidence(pulse_items)
    actions = build_action_queue(momentum, confidence, baseline)
    assert not confidence.empty
    assert len(actions) <= 5

    competitor_items, competitor_status = fetch_competitor_intelligence(
        CompetitorQuery(competitors=("HubSpot",), keywords=("AI marketing",), max_items_per_source=1),
        sources=("Meta Ad Library", "TikTok Creative Center"),
    )
    # Without API tokens both sources contribute live-search link rows.
    assert len(competitor_items) == 2
    assert not competitor_status.empty

    print("smoke test passed")


if __name__ == "__main__":
    main()
