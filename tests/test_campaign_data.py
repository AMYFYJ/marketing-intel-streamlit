from __future__ import annotations

import pandas as pd

from data_sources.campaign_data import (
    PLATFORM_CREATIVE_FORMATS,
    CampaignFilters,
    add_recommendations,
    detect_anomalies,
    filter_campaigns,
    generate_campaign_sample,
    summarize_metrics,
)


def test_generate_campaign_sample_shape_and_constraints() -> None:
    frame = generate_campaign_sample(rows=1_000, seed=7)

    assert len(frame) == 1_000
    assert frame["spend"].gt(0).all()
    assert frame["impressions"].ge(frame["clicks"]).all()
    assert frame["clicks"].ge(frame["conversions"]).all()
    assert frame["ctr"].between(0, 1).all()
    assert frame["cvr"].between(0, 1).all()
    assert frame["roas"].ge(0).all()


def test_summarize_metrics_uses_weighted_totals() -> None:
    frame = pd.DataFrame(
        {
            "spend": [100.0, 300.0],
            "revenue": [250.0, 450.0],
            "clicks": [50, 150],
            "impressions": [1_000, 2_000],
            "conversions": [5, 15],
        }
    )

    metrics = summarize_metrics(frame)

    assert metrics["spend"] == 400.0
    assert metrics["revenue"] == 700.0
    assert metrics["profit"] == 300.0
    assert metrics["roas"] == 1.75
    assert metrics["cpa"] == 20.0
    assert round(metrics["ctr"], 6) == round(200 / 3_000, 6)
    assert metrics["cvr"] == 0.1


def test_filter_campaigns_by_platform_and_date() -> None:
    frame = generate_campaign_sample(rows=250, seed=10)
    platform = frame["platform"].iloc[0]
    start = frame["date"].min() + pd.Timedelta(days=120)
    end = frame["date"].max() - pd.Timedelta(days=120)

    filtered = filter_campaigns(frame, CampaignFilters(start_date=start, end_date=end, platforms=(platform,)))

    assert not filtered.empty
    assert set(filtered["platform"].unique()) == {platform}
    assert filtered["date"].min() >= start
    assert filtered["date"].max() <= end


def test_creative_formats_are_valid_for_each_platform() -> None:
    frame = generate_campaign_sample(rows=8_000, seed=42)
    for platform, group in frame.groupby("platform"):
        used = set(group["creative_format"].unique())
        assert used.issubset(set(PLATFORM_CREATIVE_FORMATS[platform])), (platform, used)


def test_benchmarks_are_realistic() -> None:
    frame = generate_campaign_sample(rows=20_000, seed=42)
    by_platform = frame.groupby("platform").agg(roas=("roas", "mean"), cpa=("cpa", "median"))

    # Believable paid-media ranges (no more 17x ROAS / $10 CPA fantasy numbers).
    assert by_platform["roas"].between(1.0, 5.0).all(), by_platform["roas"].to_dict()
    assert by_platform["cpa"].between(15, 200).all(), by_platform["cpa"].to_dict()
    blended = frame["revenue"].sum() / frame["spend"].sum()
    assert 2.0 <= blended <= 4.5, blended


def test_recommendations_and_anomaly_columns_are_added() -> None:
    frame = generate_campaign_sample(rows=500, seed=11)
    recommended = add_recommendations(detect_anomalies(frame))

    assert set(recommended["recommendation"].unique()).issubset({"Scale", "Watch", "Optimize", "Pause"})
    assert recommended["anomaly"].dtype == bool
