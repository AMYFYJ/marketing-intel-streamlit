from __future__ import annotations

import pandas as pd

from data_sources.campaign_data import (
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


def test_generate_campaign_sample_campaigns_persist_across_days() -> None:
    frame = generate_campaign_sample(rows=5_000, seed=7)
    days_per_campaign = frame.groupby("campaign_id")["date"].nunique()

    assert days_per_campaign.median() >= 21
    assert not frame.duplicated(subset=["campaign_id", "date"]).any()
    # Campaign attributes stay fixed over the campaign's life.
    assert (frame.groupby("campaign_id")["platform"].nunique() == 1).all()
    assert (frame.groupby("campaign_id")["objective"].nunique() == 1).all()


def test_generate_campaign_sample_is_calibrated_to_believable_economics() -> None:
    frame = generate_campaign_sample(rows=30_000, seed=42)
    metrics = summarize_metrics(frame)

    assert 2.0 <= metrics["roas"] <= 5.0
    assert 20.0 <= metrics["cpa"] <= 120.0
    platform_roas = frame.groupby("platform").apply(
        lambda g: g["revenue"].sum() / g["spend"].sum(), include_groups=False
    )
    assert platform_roas.max() <= 8.0
    # Search stays more efficient than LinkedIn, preserving believable ordering.
    assert platform_roas["Google"] > platform_roas["LinkedIn"]


def test_generate_campaign_sample_is_deterministic() -> None:
    left = generate_campaign_sample(rows=2_000, seed=11)
    right = generate_campaign_sample(rows=2_000, seed=11)

    pd.testing.assert_frame_equal(left, right)


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
    frame = generate_campaign_sample(rows=5_000, seed=10)
    start = frame["date"].min() + pd.Timedelta(days=120)
    end = frame["date"].max() - pd.Timedelta(days=120)
    in_window = frame[(frame["date"] >= start) & (frame["date"] <= end)]
    platform = in_window["platform"].mode().iloc[0]

    filtered = filter_campaigns(frame, CampaignFilters(start_date=start, end_date=end, platforms=(platform,)))

    assert not filtered.empty
    assert set(filtered["platform"].unique()) == {platform}
    assert filtered["date"].min() >= start
    assert filtered["date"].max() <= end


def test_recommendations_and_anomaly_columns_are_added() -> None:
    frame = generate_campaign_sample(rows=500, seed=11)
    recommended = add_recommendations(detect_anomalies(frame))

    assert set(recommended["recommendation"].unique()).issubset({"Scale", "Watch", "Optimize", "Pause"})
    assert recommended["anomaly"].dtype == bool
