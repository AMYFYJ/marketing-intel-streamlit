from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/juniornsa/digital-advertising-campaign-performance-dataset"

SOURCE_CANDIDATES = (
    "digital_advertising_campaign_performance.csv",
    "digital_ad_campaigns.csv",
    "paid_media_campaigns.csv",
    "campaign_performance.csv",
)

PLATFORMS = ["Meta", "Google", "TikTok", "LinkedIn", "YouTube", "Pinterest", "Snapchat", "Reddit"]
OBJECTIVES = ["Awareness", "Traffic", "Lead Gen", "Sales", "App Install", "Retargeting"]
INDUSTRIES = ["Retail", "SaaS", "Beauty", "Finance", "Travel", "Healthcare", "Education", "Gaming"]
AUDIENCES = ["Prospecting", "Lookalike", "Retargeting", "High Intent", "Lifecycle", "Lapsed Customer"]
DEVICES = ["Mobile", "Desktop", "Tablet"]
CREATIVES = ["Video", "Static Image", "Carousel", "Collection", "Short-form Video", "Text Search"]
PLACEMENTS = ["Feed", "Stories", "Search", "Reels/Shorts", "Display", "Marketplace", "In-stream"]
GEOS = ["US", "Canada", "UK", "Australia", "Germany", "France", "Brazil", "Mexico"]
BUDGET_TIERS = ["Test", "Growth", "Scale", "Enterprise"]

NUMERIC_COLUMNS = [
    "spend",
    "impressions",
    "clicks",
    "conversions",
    "revenue",
    "ctr",
    "cpc",
    "cvr",
    "cpa",
    "roas",
    "profit",
    "frequency",
    "reach",
    "video_views",
    "add_to_cart",
    "landing_page_view",
    "quality_score",
]


@dataclass(frozen=True)
class CampaignFilters:
    start_date: pd.Timestamp | None = None
    end_date: pd.Timestamp | None = None
    platforms: tuple[str, ...] = ()
    objectives: tuple[str, ...] = ()
    industries: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    creative_formats: tuple[str, ...] = ()
    budget_tiers: tuple[str, ...] = ()


def load_campaign_data(data_dir: str | Path = "data", fallback_rows: int = 30_000) -> pd.DataFrame:
    """Load the preferred public campaign dataset or a deterministic fallback sample."""
    data_path = _find_source_file(Path(data_dir))
    if data_path:
        raw = pd.read_csv(data_path)
        frame = standardize_campaign_frame(raw)
        return recompute_metrics(frame)
    return generate_campaign_sample(fallback_rows)


def _find_source_file(data_dir: Path) -> Path | None:
    for filename in SOURCE_CANDIDATES:
        candidate = data_dir / filename
        if candidate.exists():
            return candidate
    return None


def standardize_campaign_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns={column: _snake(column) for column in frame.columns}).copy()

    aliases = {
        "campaign": "campaign_name",
        "campaign_title": "campaign_name",
        "channel": "platform",
        "ad_platform": "platform",
        "campaign_objective": "objective",
        "target_audience": "audience_segment",
        "audience": "audience_segment",
        "creative_type": "creative_format",
        "ad_format": "creative_format",
        "cost": "spend",
        "amount_spent": "spend",
        "conversion": "conversions",
        "sales": "conversions",
        "sales_revenue": "revenue",
        "return_on_ad_spend": "roas",
    }
    renamed = renamed.rename(columns={src: dst for src, dst in aliases.items() if src in renamed.columns})

    required_defaults = {
        "date": pd.Timestamp("2025-01-01"),
        "campaign_id": [f"CMP-{idx:06d}" for idx in range(len(renamed))],
        "campaign_name": [f"Campaign {idx + 1}" for idx in range(len(renamed))],
        "platform": "Unknown",
        "objective": "Unknown",
        "industry": "Unknown",
        "audience_segment": "Unknown",
        "device": "Unknown",
        "creative_format": "Unknown",
        "placement": "Unknown",
        "geo": "Unknown",
        "budget_tier": "Unknown",
        "retargeting": False,
        "frequency": 1.0,
        "reach": 0,
        "video_views": 0,
        "add_to_cart": 0,
        "landing_page_view": 0,
        "quality_score": 5.0,
    }
    for column, default in required_defaults.items():
        if column not in renamed.columns:
            renamed[column] = default

    for column in ["spend", "impressions", "clicks", "conversions", "revenue"]:
        if column not in renamed.columns:
            renamed[column] = 0

    renamed["date"] = pd.to_datetime(renamed["date"], errors="coerce").fillna(pd.Timestamp("2025-01-01"))
    for column in NUMERIC_COLUMNS:
        if column in renamed.columns:
            renamed[column] = pd.to_numeric(renamed[column], errors="coerce").fillna(0)
    renamed["retargeting"] = renamed["retargeting"].astype(str).str.lower().isin(["true", "1", "yes", "retargeting"])
    return renamed


def _snake(value: str) -> str:
    return (
        value.strip()
        .replace("%", "pct")
        .replace("/", "_")
        .replace("-", "_")
        .replace(" ", "_")
        .lower()
    )


PLATFORM_CTR = {"Meta": 0.011, "Google": 0.037, "TikTok": 0.014, "LinkedIn": 0.008, "YouTube": 0.010, "Pinterest": 0.012, "Snapchat": 0.013, "Reddit": 0.009}
PLATFORM_CPM = {"Meta": 12, "Google": 18, "TikTok": 9, "LinkedIn": 34, "YouTube": 14, "Pinterest": 10, "Snapchat": 8, "Reddit": 11}
# Calibrated so blended ROAS lands near 3x with believable platform ordering.
PLATFORM_TARGET_ROAS = {"Meta": 3.2, "Google": 4.2, "TikTok": 2.8, "LinkedIn": 1.4, "YouTube": 2.2, "Pinterest": 2.6, "Snapchat": 2.3, "Reddit": 2.0}
# Campaign-level ROAS dispersion per platform; differentiates optimizer risk indices.
PLATFORM_ROAS_VOLATILITY = {"Meta": 0.35, "Google": 0.22, "TikTok": 0.55, "LinkedIn": 0.30, "YouTube": 0.40, "Pinterest": 0.45, "Snapchat": 0.60, "Reddit": 0.50}
OBJECTIVE_CVR = {"Awareness": 0.004, "Traffic": 0.008, "Lead Gen": 0.025, "Sales": 0.018, "App Install": 0.015, "Retargeting": 0.035}
# Spend-weighted mean of these factors is ~1.0 so platform ROAS targets hold.
OBJECTIVE_ROAS_FACTOR = {"Awareness": 0.55, "Traffic": 0.80, "Lead Gen": 1.00, "Sales": 1.25, "App Install": 0.90, "Retargeting": 1.55}
TIER_DAILY_SPEND = {"Test": 35, "Growth": 140, "Scale": 450, "Enterprise": 1400}
WEEKDAY_SPEND_FACTOR = np.array([1.05, 1.06, 1.07, 1.06, 1.02, 0.88, 0.86])

SAMPLE_DATE_RANGE = (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-31"))
CAMPAIGN_LIFESPAN_DAYS = (21, 240)


def generate_campaign_sample(rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """Generate persistent campaigns with one row per campaign per active day.

    Campaigns keep fixed attributes over a 30-400 day lifespan; daily rows add
    weekday/seasonality/noise dynamics plus per-campaign performance drift, so
    period comparisons and campaign-level trends are meaningful.
    """
    if rows <= 0:
        raise ValueError("rows must be positive")
    rng = np.random.default_rng(seed)
    start_date, end_date = SAMPLE_DATE_RANGE
    total_days = (end_date - start_date).days + 1

    # Spawn campaigns until cumulative campaign-days cover the requested rows,
    # then trim the last lifespan so len(frame) == rows exactly.
    min_life, max_life = CAMPAIGN_LIFESPAN_DAYS
    n_estimate = max(rows // ((min_life + max_life) // 2) + 8, 4)
    lifespans = rng.integers(min_life, max_life + 1, size=n_estimate)
    while int(lifespans.sum()) < rows:
        lifespans = np.concatenate([lifespans, rng.integers(min_life, max_life + 1, size=n_estimate)])
    cutoff = int(np.searchsorted(np.cumsum(lifespans), rows)) + 1
    lifespans = lifespans[:cutoff].copy()
    lifespans[-1] -= int(lifespans.sum()) - rows
    n = len(lifespans)

    platform = rng.choice(PLATFORMS, size=n, p=[0.26, 0.23, 0.16, 0.09, 0.10, 0.06, 0.05, 0.05])
    objective = rng.choice(OBJECTIVES, size=n, p=[0.16, 0.20, 0.18, 0.25, 0.08, 0.13])
    industry = rng.choice(INDUSTRIES, size=n)
    audience = rng.choice(AUDIENCES, size=n, p=[0.30, 0.18, 0.20, 0.12, 0.12, 0.08])
    device = rng.choice(DEVICES, size=n, p=[0.68, 0.27, 0.05])
    creative = rng.choice(CREATIVES, size=n, p=[0.24, 0.24, 0.15, 0.08, 0.20, 0.09])
    placement = rng.choice(PLACEMENTS, size=n)
    geo = rng.choice(GEOS, size=n)
    budget_tier = rng.choice(BUDGET_TIERS, size=n, p=[0.42, 0.34, 0.19, 0.05])
    start_day = rng.integers(0, np.maximum(total_days - lifespans, 1))

    # Campaign-level baselines (mean-corrected lognormal noise keeps platform averages on target).
    ctr_mod = np.where(creative == "Text Search", 1.8, np.where(creative == "Video", 1.12, 1.0))
    ctr_campaign = np.array([PLATFORM_CTR[p] for p in platform]) * ctr_mod * rng.lognormal(-0.02, 0.20, n)
    cvr_mod = np.where(audience == "Retargeting", 1.65, np.where(audience == "High Intent", 1.32, 1.0))
    cvr_campaign = np.array([OBJECTIVE_CVR[o] for o in objective]) * cvr_mod * rng.lognormal(-0.02, 0.20, n)
    cpm_campaign = np.array([PLATFORM_CPM[p] for p in platform], dtype=float) * rng.lognormal(0, 0.15, n)
    base_daily_spend = np.array([TIER_DAILY_SPEND[t] for t in budget_tier], dtype=float) * rng.lognormal(0, 0.35, n)

    volatility = np.array([PLATFORM_ROAS_VOLATILITY[p] for p in platform])
    performance_mult = rng.lognormal(-(volatility**2) / 2, volatility)
    roas_campaign = np.array([PLATFORM_TARGET_ROAS[p] for p in platform]) * np.array([OBJECTIVE_ROAS_FACTOR[o] for o in objective]) * performance_mult
    expected_cpa_campaign = cpm_campaign / (1000 * ctr_campaign * cvr_campaign)
    aov_campaign = roas_campaign * expected_cpa_campaign
    monthly_drift = rng.normal(0, 0.04, n).clip(-0.10, 0.10)
    quality_score = rng.normal(6.8, 1.2, n).clip(1, 10).round(1)

    # Expand to campaign-day rows.
    camp = np.repeat(np.arange(n), lifespans)
    day_offset = np.arange(rows) - np.repeat(np.cumsum(lifespans) - lifespans, lifespans)
    date = start_date + pd.to_timedelta(start_day[camp] + day_offset, unit="D")
    month = date.month.to_numpy()
    seasonality = 1 + 0.18 * np.sin((month - 1) / 12 * 2 * np.pi)
    weekday_factor = WEEKDAY_SPEND_FACTOR[date.weekday.to_numpy()]

    spend = (base_daily_spend[camp] * weekday_factor * seasonality * rng.lognormal(0, 0.22, rows)).clip(10, 25_000).round(2)
    cpm_day = cpm_campaign[camp] * rng.lognormal(0, 0.06, rows)
    impressions = np.maximum((spend / cpm_day * 1000).astype(int), 200)

    ctr_day = np.clip(ctr_campaign[camp] * rng.lognormal(0, 0.08, rows), 0.0008, 0.15)
    clicks = np.minimum(rng.binomial(impressions, ctr_day), impressions)

    drift_mult = np.clip((1 + monthly_drift[camp]) ** (day_offset / 30), 0.5, 2.0)
    cvr_day = np.clip(cvr_campaign[camp] * drift_mult * rng.lognormal(0, 0.08, rows), 0.0005, 0.30)
    conversions = np.minimum(rng.binomial(np.maximum(clicks, 1), cvr_day), clicks)

    revenue = (conversions * aov_campaign[camp] * rng.lognormal(0, 0.10, rows)).round(2)

    reach = np.maximum((impressions / rng.uniform(1.2, 4.5, rows)).astype(int), 1)
    video_views = np.where(np.isin(creative[camp], ["Video", "Short-form Video"]), (impressions * rng.uniform(0.16, 0.62, rows)).astype(int), 0)
    add_to_cart = np.minimum((conversions * rng.uniform(1.2, 3.5, rows)).astype(int), clicks)
    landing_page_view = np.minimum((clicks * rng.uniform(0.62, 0.96, rows)).astype(int), clicks)

    frame = pd.DataFrame(
        {
            "date": date,
            "campaign_id": np.array([f"CMP-{idx + 1:05d}" for idx in range(n)])[camp],
            "campaign_name": np.array([f"{platform[idx]} {objective[idx]} {industry[idx]} {idx + 1}" for idx in range(n)])[camp],
            "platform": platform[camp],
            "objective": objective[camp],
            "industry": industry[camp],
            "audience_segment": audience[camp],
            "device": device[camp],
            "creative_format": creative[camp],
            "placement": placement[camp],
            "geo": geo[camp],
            "budget_tier": budget_tier[camp],
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "revenue": revenue,
            "retargeting": np.isin(audience[camp], ["Retargeting", "Lifecycle"]),
            "frequency": (impressions / reach).round(2),
            "reach": reach,
            "video_views": video_views,
            "add_to_cart": add_to_cart,
            "landing_page_view": landing_page_view,
            "quality_score": quality_score[camp],
        }
    )
    return recompute_metrics(frame)


def recompute_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    for column in ["spend", "impressions", "clicks", "conversions", "revenue"]:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0)

    df["ctr"] = _safe_divide(df["clicks"], df["impressions"])
    df["cpc"] = _safe_divide(df["spend"], df["clicks"])
    df["cvr"] = _safe_divide(df["conversions"], df["clicks"])
    df["cpa"] = _safe_divide(df["spend"], df["conversions"])
    df["roas"] = _safe_divide(df["revenue"], df["spend"])
    df["profit"] = df["revenue"] - df["spend"]
    return df


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def filter_campaigns(frame: pd.DataFrame, filters: CampaignFilters) -> pd.DataFrame:
    df = frame.copy()
    if filters.start_date is not None:
        df = df[df["date"] >= pd.Timestamp(filters.start_date)]
    if filters.end_date is not None:
        df = df[df["date"] <= pd.Timestamp(filters.end_date)]
    for attr, column in [
        ("platforms", "platform"),
        ("objectives", "objective"),
        ("industries", "industry"),
        ("devices", "device"),
        ("creative_formats", "creative_format"),
        ("budget_tiers", "budget_tier"),
    ]:
        values = getattr(filters, attr)
        if values:
            df = df[df[column].isin(values)]
    return df


def summarize_metrics(frame: pd.DataFrame) -> dict[str, float]:
    spend = float(frame["spend"].sum())
    revenue = float(frame["revenue"].sum())
    clicks = float(frame["clicks"].sum())
    impressions = float(frame["impressions"].sum())
    conversions = float(frame["conversions"].sum())
    return {
        "spend": spend,
        "revenue": revenue,
        "profit": revenue - spend,
        "roas": revenue / spend if spend else 0.0,
        "cpa": spend / conversions if conversions else 0.0,
        "ctr": clicks / impressions if impressions else 0.0,
        "cvr": conversions / clicks if clicks else 0.0,
        "conversions": conversions,
        "impressions": impressions,
        "clicks": clicks,
    }


def add_recommendations(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    platform_median_cpa = df.groupby("platform")["cpa"].transform("median").replace(0, np.nan)
    conditions = [
        (df["roas"] >= 3.0) & (df["profit"] > 0),
        (df["roas"] >= 1.5) & (df["cpa"] <= platform_median_cpa.fillna(df["cpa"].median()) * 1.15),
        (df["roas"] < 0.8) | (df["profit"] < 0),
    ]
    choices = ["Scale", "Watch", "Pause"]
    df["recommendation"] = np.select(conditions, choices, default="Optimize")
    return df


def detect_anomalies(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    flags = pd.Series(False, index=df.index)
    for metric in ["cpa", "roas", "ctr", "cvr"]:
        med = df.groupby("platform")[metric].transform("median")
        mad = (df[metric] - med).abs().groupby(df["platform"]).transform("median").replace(0, np.nan)
        robust_z = ((df[metric] - med).abs() / mad).replace([np.inf, -np.inf], np.nan).fillna(0)
        flags = flags | (robust_z > 5.0)
    df["anomaly"] = flags
    return df


def unique_sorted(frame: pd.DataFrame, column: str) -> list[str]:
    return sorted(str(value) for value in frame[column].dropna().unique())


def to_tuple(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(str(value) for value in values if str(value))
