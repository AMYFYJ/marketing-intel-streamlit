from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from data_sources.campaign_data import recompute_metrics, standardize_campaign_frame

# Fallback assumptions when uploads lack the daily grain needed to fit curves.
ASSUMED_MARGINAL_ROAS_RATIO = 0.65
ASSUMED_SATURATION_INDEX = 0.72


@dataclass(frozen=True)
class ConnectorSpec:
    name: str
    status: str
    required_secrets: tuple[str, ...]
    notes: str


def available_connector_specs() -> list[ConnectorSpec]:
    return [
        ConnectorSpec("CSV upload", "ready", (), "Upload a campaign CSV and normalize it into the shared metric schema."),
        ConnectorSpec("Google Ads API", "stub", ("GOOGLE_ADS_DEVELOPER_TOKEN", "GOOGLE_ADS_CLIENT_ID", "GOOGLE_ADS_CLIENT_SECRET"), "Future authenticated connector for real campaign data."),
        ConnectorSpec("Meta Marketing API", "stub", ("META_ACCESS_TOKEN", "META_AD_ACCOUNT_ID"), "Future connector only if account access becomes available."),
        ConnectorSpec("TikTok Ads API", "stub", ("TIKTOK_ACCESS_TOKEN", "TIKTOK_ADVERTISER_ID"), "Future authenticated connector for TikTok campaign reporting."),
        ConnectorSpec("LinkedIn Ads API", "stub", ("LINKEDIN_ACCESS_TOKEN", "LINKEDIN_ACCOUNT_ID"), "Future authenticated connector for B2B campaign reporting."),
    ]


def normalize_uploaded_campaign_csv(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize an uploaded campaign CSV and attach marginal economics.

    When a platform has enough daily spend variation, the marginal ROAS and
    saturation index are fitted from a log-log spend/revenue elasticity;
    otherwise documented fallback assumptions are used. The per-row
    `curve_source` column records which path applied.
    """
    normalized = recompute_metrics(standardize_campaign_frame(frame))
    return estimate_marginal_economics(normalized)


def estimate_marginal_economics(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    if "marginal_roas" in df.columns and "diminishing_return_index" in df.columns:
        if "curve_source" not in df.columns:
            df["curve_source"] = "provided"
        return df

    df["marginal_roas"] = (df["roas"] * ASSUMED_MARGINAL_ROAS_RATIO).clip(lower=0)
    df["diminishing_return_index"] = ASSUMED_SATURATION_INDEX
    df["curve_source"] = "assumed"

    daily = df[(df["spend"] > 0) & (df["revenue"] > 0)].groupby(["platform", "date"], as_index=False).agg(
        spend=("spend", "sum"), revenue=("revenue", "sum")
    )
    for platform, group in daily.groupby("platform"):
        if len(group) < 4 or group["spend"].round(2).nunique() < 3:
            continue
        # Power-law response revenue = a * spend^beta; beta is the elasticity,
        # so marginal ROAS = beta * average ROAS and beta itself proxies saturation.
        beta = float(np.polyfit(np.log(group["spend"]), np.log(group["revenue"]), 1)[0])
        beta = float(np.clip(beta, 0.10, 1.0))
        platform_roas = float(group["revenue"].sum() / group["spend"].sum())
        mask = df["platform"] == platform
        df.loc[mask, "marginal_roas"] = beta * platform_roas
        df.loc[mask, "diminishing_return_index"] = float(np.clip(beta, 0.18, 1.0))
        df.loc[mask, "curve_source"] = "fitted"

    if "incremental_revenue" not in df.columns:
        ratio = (df["marginal_roas"] / df["roas"].replace(0, np.nan)).clip(0, 1).fillna(ASSUMED_MARGINAL_ROAS_RATIO)
        df["incremental_revenue"] = df["revenue"] * ratio
    return df


def missing_required_secrets(configured: Sequence[str], spec: ConnectorSpec) -> tuple[str, ...]:
    configured_set = set(configured)
    return tuple(secret for secret in spec.required_secrets if secret not in configured_set)
