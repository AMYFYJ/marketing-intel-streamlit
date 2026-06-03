from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_sources.campaign_data import (
    AUDIENCES,
    BUDGET_TIERS,
    CREATIVES,
    DEVICES,
    GEOS,
    INDUSTRIES,
    OBJECTIVES,
    PLATFORMS,
    generate_campaign_sample,
    recompute_metrics,
)

DEFAULT_SYNTHETIC_ROWS = 250_000


@dataclass(frozen=True)
class ChannelBenchmark:
    platform: str
    spend_share: float
    median_spend: float
    ctr: float
    cvr: float
    cpm: float
    aov: float
    roas: float


def build_channel_benchmarks(calibration_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    source = calibration_frame if calibration_frame is not None else generate_campaign_sample(rows=10_000, seed=42)
    source = recompute_metrics(source)
    total_spend = max(float(source["spend"].sum()), 1.0)

    grouped = source.groupby("platform", as_index=False).agg(
        spend=("spend", "sum"),
        median_spend=("spend", "median"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
        conversions=("conversions", "sum"),
        revenue=("revenue", "sum"),
    )
    grouped["spend_share"] = grouped["spend"] / total_spend
    grouped["ctr"] = grouped["clicks"] / grouped["impressions"].replace(0, np.nan)
    grouped["cvr"] = grouped["conversions"] / grouped["clicks"].replace(0, np.nan)
    grouped["cpm"] = grouped["spend"] / grouped["impressions"].replace(0, np.nan) * 1000
    grouped["aov"] = grouped["revenue"] / grouped["conversions"].replace(0, np.nan)
    grouped["roas"] = grouped["revenue"] / grouped["spend"].replace(0, np.nan)
    grouped = grouped.fillna({"ctr": 0.01, "cvr": 0.03, "cpm": 12.0, "aov": 120.0, "roas": 1.4})
    grouped["spend_share"] = grouped["spend_share"] / grouped["spend_share"].sum()
    return grouped[["platform", "spend_share", "median_spend", "ctr", "cvr", "cpm", "aov", "roas"]]


def generate_synthetic_media_mix(
    rows: int = DEFAULT_SYNTHETIC_ROWS,
    seed: int = 2026,
    calibration_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generate a deterministic, realistic media-mix dataset for optimization demos."""
    if rows <= 0:
        raise ValueError("rows must be positive")

    rng = np.random.default_rng(seed)
    benchmarks = build_channel_benchmarks(calibration_frame)
    platform_probs = benchmarks["spend_share"].to_numpy(dtype=float)
    platform_probs = platform_probs / platform_probs.sum()
    platform = rng.choice(benchmarks["platform"].to_numpy(), size=rows, p=platform_probs)
    bench = benchmarks.set_index("platform").loc[platform].reset_index(drop=True)

    dates = pd.date_range("2024-01-01", "2026-06-01", freq="D")
    date = rng.choice(dates, size=rows)
    month = pd.to_datetime(date).month.to_numpy()
    seasonality = 1 + 0.18 * np.sin((month - 1) / 12 * 2 * np.pi) + rng.normal(0, 0.04, rows)
    seasonality = np.clip(seasonality, 0.72, 1.34)

    objective = rng.choice(OBJECTIVES, size=rows, p=[0.14, 0.18, 0.17, 0.29, 0.07, 0.15])
    industry = rng.choice(INDUSTRIES, size=rows)
    audience = rng.choice(AUDIENCES, size=rows, p=[0.28, 0.18, 0.22, 0.14, 0.11, 0.07])
    device = rng.choice(DEVICES, size=rows, p=[0.70, 0.25, 0.05])
    creative = rng.choice(CREATIVES, size=rows, p=[0.22, 0.22, 0.16, 0.08, 0.22, 0.10])
    geo = rng.choice(GEOS, size=rows)
    budget_tier = rng.choice(BUDGET_TIERS, size=rows, p=[0.40, 0.35, 0.20, 0.05])

    tier_multiplier = pd.Series(budget_tier).map({"Test": 0.48, "Growth": 1.0, "Scale": 2.7, "Enterprise": 7.2}).to_numpy(dtype=float)
    spend = bench["median_spend"].to_numpy(dtype=float) * tier_multiplier * seasonality * rng.lognormal(0, 0.50, rows)
    spend = np.clip(spend, 50, 125_000).round(2)

    saturation = 1 / (1 + np.power(spend / np.maximum(bench["median_spend"].to_numpy(dtype=float) * 4.5, 1), 0.72))
    diminishing_return_index = np.clip(saturation + rng.normal(0, 0.03, rows), 0.18, 1.05)

    cpm = bench["cpm"].to_numpy(dtype=float) * rng.lognormal(0, 0.18, rows) * np.where(device == "Desktop", 1.12, 1.0)
    impressions = np.maximum((spend / np.maximum(cpm, 0.5) * 1000).astype(int), 100)

    ctr = bench["ctr"].to_numpy(dtype=float) * rng.lognormal(0, 0.26, rows)
    ctr *= np.where(creative == "Text Search", 1.65, np.where(np.isin(creative, ["Video", "Short-form Video"]), 1.10, 1.0))
    ctr = np.clip(ctr, 0.0005, 0.22)
    clicks = np.minimum(rng.binomial(impressions, ctr), impressions)

    cvr = bench["cvr"].to_numpy(dtype=float) * rng.lognormal(0, 0.25, rows)
    cvr *= np.where(audience == "Retargeting", 1.55, np.where(audience == "High Intent", 1.24, 1.0))
    cvr *= diminishing_return_index
    cvr = np.clip(cvr, 0.0005, 0.40)
    conversions = np.minimum(rng.binomial(np.maximum(clicks, 1), cvr), clicks)

    aov = bench["aov"].to_numpy(dtype=float) * rng.lognormal(0, 0.30, rows)
    revenue = (conversions * aov * seasonality * rng.uniform(0.86, 1.14, rows)).round(2)
    incremental_revenue = (revenue * np.clip(diminishing_return_index * rng.uniform(0.72, 1.02, rows), 0.08, 1.0)).round(2)
    marginal_roas = incremental_revenue / np.maximum(spend, 1)

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(date),
            "week": pd.to_datetime(date).to_period("W").astype(str),
            "campaign_id": [f"SYN-{idx + 1:07d}" for idx in range(rows)],
            "platform": platform,
            "objective": objective,
            "industry": industry,
            "audience_segment": audience,
            "device": device,
            "creative_format": creative,
            "geo": geo,
            "budget_tier": budget_tier,
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "revenue": revenue,
            "incremental_revenue": incremental_revenue,
            "marginal_roas": marginal_roas,
            "seasonality_index": seasonality.round(3),
            "diminishing_return_index": diminishing_return_index.round(3),
            "retargeting_share": np.where(audience == "Retargeting", rng.uniform(0.55, 0.95, rows), rng.uniform(0.0, 0.35, rows)).round(3),
        }
    )
    return recompute_metrics(frame)


def validate_media_mix_frame(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    required = {
        "date",
        "campaign_id",
        "platform",
        "objective",
        "spend",
        "impressions",
        "clicks",
        "conversions",
        "revenue",
        "profit",
        "roas",
        "cpa",
        "marginal_roas",
        "diminishing_return_index",
    }
    missing = required.difference(frame.columns)
    if missing:
        errors.append(f"missing columns: {sorted(missing)}")
        return errors
    if not frame["campaign_id"].is_unique:
        errors.append("campaign_id values must be unique")
    if not frame["spend"].gt(0).all():
        errors.append("spend must be positive")
    if not frame["impressions"].ge(frame["clicks"]).all():
        errors.append("clicks cannot exceed impressions")
    if not frame["clicks"].ge(frame["conversions"]).all():
        errors.append("conversions cannot exceed clicks")
    if not frame["revenue"].ge(0).all():
        errors.append("revenue cannot be negative")
    if not frame["marginal_roas"].ge(0).all():
        errors.append("marginal_roas cannot be negative")
    if not frame["diminishing_return_index"].between(0, 1.1).all():
        errors.append("diminishing_return_index must stay in a plausible range")
    return errors
