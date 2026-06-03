from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd

from data_sources.campaign_data import recompute_metrics, standardize_campaign_frame


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
    normalized = recompute_metrics(standardize_campaign_frame(frame))
    if "marginal_roas" not in normalized.columns:
        normalized["marginal_roas"] = (normalized["roas"] * 0.65).clip(lower=0)
    if "diminishing_return_index" not in normalized.columns:
        normalized["diminishing_return_index"] = 0.72
    if "incremental_revenue" not in normalized.columns:
        normalized["incremental_revenue"] = normalized["revenue"] * 0.65
    return normalized


def missing_required_secrets(configured: Sequence[str], spec: ConnectorSpec) -> tuple[str, ...]:
    configured_set = set(configured)
    return tuple(secret for secret in spec.required_secrets if secret not in configured_set)
