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

# Which creative formats are actually buyable on each platform. Single source of truth shared by the
# data generators (so synthetic rows are realistic) and the goal planner (so it never recommends an
# impossible combo like Text Search on TikTok).
PLATFORM_CREATIVE_FORMATS: dict[str, tuple[str, ...]] = {
    "Meta": ("Video", "Static Image", "Carousel", "Collection", "Short-form Video"),
    "Google": ("Text Search", "Static Image", "Video"),
    "TikTok": ("Short-form Video", "Video"),
    "LinkedIn": ("Static Image", "Carousel", "Video"),
    "YouTube": ("Video", "Short-form Video"),
    "Pinterest": ("Static Image", "Carousel", "Collection", "Video"),
    "Snapchat": ("Short-form Video", "Video", "Static Image", "Collection"),
    "Reddit": ("Static Image", "Video", "Carousel"),
}

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


def load_campaign_data(data_dir: str | Path = "data", fallback_rows: int = 10_000) -> pd.DataFrame:
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


def sample_creative_by_platform(platform: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pick a creative format for each row from only the formats valid on that row's platform.

    Replaces independent creative sampling so the data never contains impossible combos like
    Text Search on TikTok. Uses ``PLATFORM_CREATIVE_FORMATS`` as the single source of truth.
    """
    creative = np.empty(len(platform), dtype=object)
    for name, formats in PLATFORM_CREATIVE_FORMATS.items():
        mask = platform == name
        count = int(mask.sum())
        if count:
            creative[mask] = rng.choice(np.array(formats, dtype=object), size=count)
    return creative


def generate_campaign_sample(rows: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", "2026-01-31", freq="D")

    platform = rng.choice(PLATFORMS, size=rows, p=[0.26, 0.23, 0.16, 0.09, 0.10, 0.06, 0.05, 0.05])
    objective = rng.choice(OBJECTIVES, size=rows, p=[0.16, 0.20, 0.18, 0.25, 0.08, 0.13])
    industry = rng.choice(INDUSTRIES, size=rows)
    audience = rng.choice(AUDIENCES, size=rows, p=[0.30, 0.18, 0.20, 0.12, 0.12, 0.08])
    device = rng.choice(DEVICES, size=rows, p=[0.68, 0.27, 0.05])
    creative = sample_creative_by_platform(platform, rng)
    placement = rng.choice(PLACEMENTS, size=rows)
    budget_tier = rng.choice(BUDGET_TIERS, size=rows, p=[0.42, 0.34, 0.19, 0.05])

    platform_ctr = {"Meta": 0.011, "Google": 0.026, "TikTok": 0.014, "LinkedIn": 0.010, "YouTube": 0.010, "Pinterest": 0.012, "Snapchat": 0.013, "Reddit": 0.009}
    objective_cvr = {"Awareness": 0.009, "Traffic": 0.018, "Lead Gen": 0.024, "Sales": 0.016, "App Install": 0.018, "Retargeting": 0.034}
    platform_cpm = {"Meta": 13, "Google": 23, "TikTok": 13, "LinkedIn": 30, "YouTube": 14, "Pinterest": 12, "Snapchat": 11, "Reddit": 12}
    tier_spend = {"Test": 450, "Growth": 1800, "Scale": 5800, "Enterprise": 17000}

    base_spend = np.array([tier_spend[tier] for tier in budget_tier], dtype=float)
    spend = rng.lognormal(mean=np.log(base_spend), sigma=0.45).clip(80, 90_000).round(2)
    cpm = np.array([platform_cpm[item] for item in platform], dtype=float) * rng.lognormal(0, 0.18, rows)
    impressions = np.maximum((spend / cpm * 1000).astype(int), 500)

    ctr_base = np.array([platform_ctr[item] for item in platform], dtype=float)
    ctr_mod = np.where(creative == "Text Search", 1.25, np.where(creative == "Video", 1.12, 1.0))
    ctr = np.clip(ctr_base * ctr_mod * rng.lognormal(0, 0.30, rows), 0.001, 0.18)
    clicks = np.minimum(rng.binomial(impressions, ctr), impressions)

    cvr_base = np.array([objective_cvr[item] for item in objective], dtype=float)
    cvr_mod = np.where(audience == "Retargeting", 1.65, np.where(audience == "High Intent", 1.32, 1.0))
    cvr = np.clip(cvr_base * cvr_mod * rng.lognormal(0, 0.28, rows), 0.001, 0.35)
    conversions = np.minimum(rng.binomial(np.maximum(clicks, 1), cvr), clicks)

    aov_by_industry = {
        "Retail": 64,
        "SaaS": 170,
        "Beauty": 52,
        "Finance": 240,
        "Travel": 170,
        "Healthcare": 130,
        "Education": 110,
        "Gaming": 34,
    }
    aov = np.array([aov_by_industry[item] for item in industry], dtype=float) * rng.lognormal(0, 0.33, rows)
    revenue = (conversions * aov * rng.uniform(0.82, 1.18, rows)).round(2)

    reach = np.maximum((impressions / rng.uniform(1.2, 4.5, rows)).astype(int), 1)
    video_views = np.where(np.isin(creative, ["Video", "Short-form Video"]), (impressions * rng.uniform(0.16, 0.62, rows)).astype(int), 0)
    add_to_cart = np.minimum((conversions * rng.uniform(1.2, 3.5, rows)).astype(int), clicks)
    landing_page_view = np.minimum((clicks * rng.uniform(0.62, 0.96, rows)).astype(int), clicks)

    frame = pd.DataFrame(
        {
            "date": rng.choice(dates, size=rows),
            "campaign_id": [f"CMP-{idx + 1:06d}" for idx in range(rows)],
            "campaign_name": [f"{platform[idx]} {objective[idx]} {industry[idx]} {idx + 1}" for idx in range(rows)],
            "platform": platform,
            "objective": objective,
            "industry": industry,
            "audience_segment": audience,
            "device": device,
            "creative_format": creative,
            "placement": placement,
            "geo": rng.choice(GEOS, size=rows),
            "budget_tier": budget_tier,
            "spend": spend,
            "impressions": impressions,
            "clicks": clicks,
            "conversions": conversions,
            "revenue": revenue,
            "retargeting": np.isin(audience, ["Retargeting", "Lifecycle"]),
            "frequency": (impressions / reach).round(2),
            "reach": reach,
            "video_views": video_views,
            "add_to_cart": add_to_cart,
            "landing_page_view": landing_page_view,
            "quality_score": rng.normal(6.8, 1.2, rows).clip(1, 10).round(1),
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
