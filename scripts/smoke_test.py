from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.campaign_data import load_campaign_data, summarize_metrics
from data_sources.competitor_sources import CompetitorQuery, fetch_competitor_intelligence
from data_sources.synthetic_media_mix import generate_synthetic_media_mix, validate_media_mix_frame
from data_sources.trend_sources import TrendQuery, fetch_demand_pulse
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

    demand_items, demand_status = fetch_demand_pulse(
        TrendQuery(keywords=("AI marketing",), max_items_per_source=1),
        sources=("Google Trends export",),
        data_dir="data",
    )
    assert demand_items.empty
    assert not demand_status.empty

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
