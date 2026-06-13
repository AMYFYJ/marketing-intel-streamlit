from __future__ import annotations

import numpy as np
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


def test_allocate_budget_accounts_for_full_budget() -> None:
    frame = generate_synthetic_media_mix(rows=3_000, seed=16)
    allocation = allocate_budget(frame, OptimizerConfig(total_budget=250_000, target_roas=1.5, target_cpa=120, risk_tolerance=0.5))

    assert not allocation.empty
    allocated = float(allocation["allocation"].sum())
    unallocated = float(allocation["unallocated"].iloc[0])
    assert round(allocated + unallocated, 2) == 250_000.00
    assert allocation["expected_revenue"].ge(0).all()
    assert allocation["expected_roas"].ge(0).all()
    assert set(allocation["action"].unique()).issubset({"Increase", "Maintain", "Decrease", "Limit (Target)"})


def test_allocation_mix_shifts_when_budget_scales() -> None:
    frame = generate_synthetic_media_mix(rows=20_000, seed=16)
    small = allocate_budget(frame, OptimizerConfig(total_budget=100_000)).set_index("platform")["allocation_share"]
    large = allocate_budget(frame, OptimizerConfig(total_budget=1_000_000)).set_index("platform")["allocation_share"]
    aligned = pd.concat([small.rename("small"), large.rename("large")], axis=1).fillna(0)

    assert not np.allclose(aligned["small"].to_numpy(), aligned["large"].to_numpy(), atol=0.01)


def test_expected_roas_declines_as_budget_grows() -> None:
    frame = generate_synthetic_media_mix(rows=20_000, seed=16)
    small = summarize_allocation(allocate_budget(frame, OptimizerConfig(total_budget=100_000)))
    large = summarize_allocation(allocate_budget(frame, OptimizerConfig(total_budget=1_000_000)))

    assert large["expected_roas"] < small["expected_roas"]


def test_risk_tolerance_changes_the_mix() -> None:
    frame = generate_synthetic_media_mix(rows=20_000, seed=16)
    cautious = allocate_budget(frame, OptimizerConfig(total_budget=500_000, risk_tolerance=0.0)).sort_values("platform")
    bold = allocate_budget(frame, OptimizerConfig(total_budget=500_000, risk_tolerance=1.0)).sort_values("platform")

    assert not np.allclose(cautious["allocation"].to_numpy(), bold["allocation"].to_numpy())


def test_tight_targets_leave_budget_unallocated() -> None:
    frame = generate_synthetic_media_mix(rows=20_000, seed=16)
    allocation = allocate_budget(frame, OptimizerConfig(total_budget=2_000_000, target_roas=3.5, target_cpa=35))

    assert float(allocation["unallocated"].iloc[0]) > 0
    assert "Limit (Target)" in set(allocation["action"])


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
    assert normalized.loc[0, "curve_source"] == "assumed"


def test_uploaded_csv_with_daily_grain_fits_curves() -> None:
    rng = np.random.default_rng(5)
    days = pd.date_range("2026-05-01", periods=14, freq="D")
    spend = rng.uniform(500, 2_000, len(days))
    revenue = 8.0 * spend**0.8 * rng.lognormal(0, 0.05, len(days))
    raw = pd.DataFrame(
        {
            "Date": days,
            "Channel": "Google",
            "Campaign": [f"Search {i}" for i in range(len(days))],
            "Cost": spend.round(2),
            "Impressions": 10_000,
            "Clicks": 400,
            "Conversions": 20,
            "Revenue": revenue.round(2),
        }
    )
    normalized = normalize_uploaded_campaign_csv(raw)

    assert (normalized["curve_source"] == "fitted").all()
    # Elasticity ~0.8 means marginal ROAS sits below average ROAS.
    avg_roas = normalized["revenue"].sum() / normalized["spend"].sum()
    assert normalized["marginal_roas"].iloc[0] < avg_roas
    assert 0.5 < normalized["diminishing_return_index"].iloc[0] <= 1.0


def test_small_upload_is_not_punished_for_dataset_size() -> None:
    # A one-week upload must not look saturated just because totals are small.
    raw = pd.DataFrame(
        {
            "Date": pd.date_range("2026-05-01", periods=4, freq="D").repeat(2),
            "Channel": ["Google", "Meta"] * 4,
            "Campaign": [f"C{i}" for i in range(8)],
            "Cost": [1200.0, 3000.0, 1150.0, 2900.0, 1180.0, 3050.0, 1210.0, 2980.0],
            "Impressions": [50_000, 400_000] * 4,
            "Clicks": [2_500, 3_200] * 4,
            "Conversions": [80, 60] * 4,
            "Revenue": [9_600.0, 5_400.0, 9_100.0, 5_300.0, 9_400.0, 5_500.0, 9_300.0, 5_350.0],
        }
    )
    normalized = normalize_uploaded_campaign_csv(raw)
    allocation = allocate_budget(normalized, OptimizerConfig(total_budget=100_000, target_roas=1.0, target_cpa=200))

    funded = allocation[allocation["allocation"] > 0]
    assert not funded.empty
    assert funded["expected_roas"].max() > 1.5
