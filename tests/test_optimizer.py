from __future__ import annotations

import pandas as pd
import pytest

from data_sources.connectors import available_connector_specs, missing_required_secrets, normalize_uploaded_campaign_csv
from data_sources.synthetic_media_mix import generate_synthetic_media_mix
from utils.optimizer import OptimizerConfig, allocate_budget, prepare_channel_summary, summarize_allocation


def test_prepare_channel_summary_filters_objective_and_exclusions() -> None:
    frame = generate_synthetic_media_mix(rows=2_000, seed=15)
    objective = frame["objective"].iloc[0]
    excluded = frame["platform"].iloc[0]

    summary = prepare_channel_summary(frame, objective=objective, excluded_platforms=(excluded,))

    assert not summary.empty
    assert excluded not in set(summary["platform"])


def test_allocate_budget_sums_to_requested_budget() -> None:
    frame = generate_synthetic_media_mix(rows=3_000, seed=16)
    allocation = allocate_budget(frame, OptimizerConfig(total_budget=250_000, target_roas=1.5, target_cpa=120, risk_tolerance=0.5))

    assert not allocation.empty
    assert round(float(allocation["allocation"].sum()), 2) == 250_000.00
    assert allocation["expected_revenue"].ge(0).all()
    assert allocation["expected_roas"].ge(0).all()
    assert set(allocation["action"].unique()).issubset({"Increase", "Maintain", "Limit"})


def test_summarize_allocation_uses_weighted_totals() -> None:
    allocation = pd.DataFrame(
        {
            "allocation": [100.0, 200.0],
            "expected_revenue": [300.0, 300.0],
            "expected_conversions": [3.0, 2.0],
        }
    )

    summary = summarize_allocation(allocation)

    assert summary["budget"] == 300.0
    assert summary["expected_revenue"] == 600.0
    assert summary["expected_profit"] == 300.0
    assert summary["expected_roas"] == 2.0
    assert summary["expected_cpa"] == 60.0


def test_allocate_budget_validates_config() -> None:
    frame = generate_synthetic_media_mix(rows=100, seed=17)
    with pytest.raises(ValueError):
        allocate_budget(frame, OptimizerConfig(total_budget=0))
    with pytest.raises(ValueError):
        allocate_budget(frame, OptimizerConfig(total_budget=100, risk_tolerance=1.5))


def test_connector_specs_and_csv_normalization() -> None:
    specs = available_connector_specs()
    meta = next(spec for spec in specs if spec.name == "Meta Marketing API")

    assert missing_required_secrets(["META_ACCESS_TOKEN"], meta) == ("META_AD_ACCOUNT_ID",)

    raw = pd.DataFrame(
        {
            "Date": ["2026-01-01"],
            "Channel": ["Meta"],
            "Campaign": ["Uploaded campaign"],
            "Cost": [100.0],
            "Impressions": [1000],
            "Clicks": [50],
            "Conversions": [5],
            "Revenue": [300.0],
        }
    )
    normalized = normalize_uploaded_campaign_csv(raw)

    assert normalized.loc[0, "platform"] == "Meta"
    assert normalized.loc[0, "roas"] == 3.0
    assert normalized.loc[0, "cpa"] == 20.0
