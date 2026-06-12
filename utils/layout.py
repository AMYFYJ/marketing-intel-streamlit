from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from data_sources.campaign_data import SOURCE_CANDIDATES

OPTIONAL_SECRETS = ("YOUTUBE_API_KEY", "META_ACCESS_TOKEN", "META_GRAPH_VERSION")
TREND_EXPORTS = ("google_trends_export.csv", "pinterest_trends_export.csv")


def render_app_header() -> None:
    st.title("Marketing Intelligence Dashboard")
    st.caption("Paid media performance, competitor intelligence, demand signals, and budget planning in one Streamlit app.")


def render_data_sources() -> None:
    st.subheader("Data Sources")
    st.markdown(
        """
        **Campaign performance**
        - Preferred dataset: Kaggle Digital Advertising Campaign Performance Dataset.
        - Local CSV names supported: `digital_advertising_campaign_performance.csv`, `digital_ad_campaigns.csv`, `paid_media_campaigns.csv`, or `campaign_performance.csv`.
        - If no CSV is present, the app uses a deterministic fallback sample calibrated to believable paid-media benchmarks (~3x blended ROAS).

        **Competitor and demand intelligence**
        - GDELT and Reddit provide no-key public trend signals.
        - YouTube Data API is used when `YOUTUBE_API_KEY` is configured.
        - Meta Ad Library API is used when `META_ACCESS_TOKEN` is configured; otherwise the app provides public search links.
        - TikTok Creative Center is covered with live deep links because a stable public API is not available here.
        - Sentiment uses VADER compound scores.

        **Budget optimizer**
        - Uses a 250k-row synthetic media-mix dataset generated in memory.
        - Allocates along per-channel diminishing-return curves; CSV upload fits marginal returns from daily spend variation when available.
        """
    )
    _render_live_configuration()


def _render_live_configuration() -> None:
    st.markdown("#### Current Configuration")
    rows = [
        {
            "Item": f"Secret: {name}",
            "Status": "configured" if _get_secret(name) else "not set",
            "Detail": "Used for live API calls." if _get_secret(name) else "Optional — the app falls back to public links or other sources.",
        }
        for name in OPTIONAL_SECRETS
    ]

    data_dir = Path("data")
    campaign_csv = next((name for name in SOURCE_CANDIDATES if (data_dir / name).exists()), None)
    rows.append(
        {
            "Item": "Campaign CSV in data/",
            "Status": campaign_csv or "not found",
            "Detail": "Loaded as the Performance dataset." if campaign_csv else "Using the deterministic fallback sample.",
        }
    )
    for export in TREND_EXPORTS:
        present = (data_dir / export).exists()
        rows.append(
            {
                "Item": f"Trend export: {export}",
                "Status": "found" if present else "not found",
                "Detail": "Available as a Demand Pulse source." if present else "Optional manual export for Demand Pulse.",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _get_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None
