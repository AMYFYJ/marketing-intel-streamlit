from __future__ import annotations

import pandas as pd

from data_sources.competitor_sources import (
    LIVE_LINK_ASSET_TYPE,
    CompetitorQuery,
    analyze_creative_patterns,
    compute_share_of_voice,
    detect_cta,
    detect_theme,
    exclude_live_link_rows,
    fetch_competitor_intelligence,
    fetch_meta_ad_library,
    fetch_tiktok_creative_center_link,
    parse_competitors,
)


class FakeResponse:
    status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "data": [
                {
                    "id": "1",
                    "page_name": "Acme",
                    "ad_snapshot_url": "https://facebook.com/ad/1",
                    "ad_delivery_start_time": "2026-01-01",
                    "publisher_platforms": ["facebook", "instagram"],
                    "ad_creative_bodies": ["Start free trial for new AI workflow automation"],
                    "ad_creative_link_titles": ["Try free"],
                }
            ]
        }


def test_parse_competitors_deduplicates() -> None:
    assert parse_competitors("HubSpot, Salesforce\nHubSpot") == ("HubSpot", "Salesforce")


def test_meta_ad_library_without_token_returns_public_search_link() -> None:
    frame, status = fetch_meta_ad_library("HubSpot AI", "HubSpot", access_token=None)

    assert status["status"] == "not configured"
    assert "facebook.com/ads/library" in status["detail"]
    assert len(frame) == 1
    assert frame.loc[0, "asset_type"] == LIVE_LINK_ASSET_TYPE
    assert "facebook.com/ads/library" in frame.loc[0, "url"]


def test_meta_ad_library_parses_api_payload() -> None:
    def fake_get(*args, **kwargs):
        return FakeResponse()

    frame, status = fetch_meta_ad_library("Acme AI", "Acme", access_token="token", request_get=fake_get)

    assert status["status"] == "ok"
    assert len(frame) == 1
    assert frame.loc[0, "source"] == "Meta Ad Library"
    assert frame.loc[0, "platforms"] == "facebook, instagram"


def test_tiktok_creative_center_returns_live_link_item() -> None:
    frame, status = fetch_tiktok_creative_center_link("Acme AI", "Acme")

    assert status["status"] == "live link"
    assert len(frame) == 1
    assert frame.loc[0, "asset_type"] == LIVE_LINK_ASSET_TYPE
    assert "creativecenter" in frame.loc[0, "url"]


def test_detect_cta_and_theme() -> None:
    text = "Start free trial for AI workflow automation"

    assert detect_cta(text) == "Free trial"
    assert detect_theme(text) == "AI"


def test_detect_cta_requires_word_boundaries() -> None:
    # "sale" must not match inside "Salesforce".
    assert detect_cta("Salesforce announces quarterly results") == "No explicit CTA"
    assert detect_cta("Huge sale this weekend") == "Shop now"


def test_detect_theme_requires_word_boundaries() -> None:
    # "ai" must not match inside "email", "Spain", "domain", or "maintain".
    assert detect_theme("Spain retail report from domain.com") != "AI"
    assert detect_theme("maintaining brand safety") != "AI"
    assert detect_theme("AI marketing tools compared") == "AI"


def test_share_of_voice_and_creative_patterns_exclude_live_links() -> None:
    link_frame, _ = fetch_tiktok_creative_center_link("Acme AI", "Acme")
    real_rows = pd.DataFrame(
        [
            {
                "source": "Reddit",
                "competitor": "Acme",
                "keyword": "Acme AI",
                "asset_type": "Social mention",
                "title": "Acme launches AI automation suite",
                "text": "free trial available",
                "url": "https://example.com/post",
                "published_at": pd.Timestamp.now(tz="UTC"),
                "author": "user",
                "platforms": "Reddit",
                "engagement": 4.0,
            },
            {
                "source": "Reddit",
                "competitor": "Beta",
                "keyword": "Beta AI",
                "asset_type": "Social mention",
                "title": "Beta review thread",
                "text": "",
                "url": "https://example.com/post2",
                "published_at": pd.Timestamp.now(tz="UTC"),
                "author": "user",
                "platforms": "Reddit",
                "engagement": 1.0,
            },
        ]
    )
    items = pd.concat([link_frame, real_rows], ignore_index=True)
    items["sentiment"] = 0.1
    items["cta"] = items["text"].map(detect_cta)
    items["theme"] = items["title"].map(detect_theme)

    sov = compute_share_of_voice(items)
    patterns = analyze_creative_patterns(items)

    assert set(sov["competitor"]) == {"Acme", "Beta"}
    assert "TikTok Creative Center" not in set(sov["source"])
    assert not patterns.empty
    assert int(sov["items"].sum()) == 2


def test_exclude_live_link_rows_keeps_real_items() -> None:
    link_frame, _ = fetch_tiktok_creative_center_link("Acme AI", "Acme")
    assert exclude_live_link_rows(link_frame).empty


def test_fetch_competitor_intelligence_combines_link_sources() -> None:
    items, statuses = fetch_competitor_intelligence(
        CompetitorQuery(competitors=("Acme",), keywords=("AI",), max_items_per_source=5),
        sources=("Meta Ad Library", "TikTok Creative Center"),
    )

    assert len(items) == 2
    assert set(items["asset_type"]) == {LIVE_LINK_ASSET_TYPE}
    assert set(statuses["source"]) == {"Meta Ad Library", "TikTok Creative Center"}
    assert set(items["competitor"]) == {"Acme"}


def test_fetch_competitor_intelligence_without_token_returns_meta_link() -> None:
    items, statuses = fetch_competitor_intelligence(
        CompetitorQuery(competitors=("Acme",), keywords=("AI",), max_items_per_source=5),
        sources=("Meta Ad Library",),
    )

    assert len(items) == 1
    assert items.loc[0, "asset_type"] == LIVE_LINK_ASSET_TYPE
    assert statuses.loc[0, "status"] == "not configured"
