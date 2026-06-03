from __future__ import annotations

from data_sources.competitor_sources import (
    CompetitorQuery,
    analyze_creative_patterns,
    compute_share_of_voice,
    detect_cta,
    detect_theme,
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


def test_meta_ad_library_without_token_returns_public_search_status() -> None:
    frame, status = fetch_meta_ad_library("HubSpot AI", "HubSpot", access_token=None)

    assert frame.empty
    assert status["status"] == "not configured"
    assert "facebook.com/ads/library" in status["detail"]


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
    assert "creativecenter" in frame.loc[0, "url"]


def test_detect_cta_and_theme() -> None:
    text = "Start free trial for AI workflow automation"

    assert detect_cta(text) == "Free trial"
    assert detect_theme(text) == "AI"


def test_share_of_voice_and_creative_patterns() -> None:
    frame, _ = fetch_tiktok_creative_center_link("Acme AI", "Acme")
    other, _ = fetch_tiktok_creative_center_link("Beta AI", "Beta")
    items = frame._append(other, ignore_index=True)
    items["sentiment"] = [0.1, 0.2]
    items["cta"] = items["text"].map(detect_cta)
    items["theme"] = items["text"].map(detect_theme)

    sov = compute_share_of_voice(items)
    patterns = analyze_creative_patterns(items)

    assert set(sov["competitor"]) == {"Acme", "Beta"}
    assert not patterns.empty


def test_fetch_competitor_intelligence_combines_link_sources() -> None:
    items, statuses = fetch_competitor_intelligence(
        CompetitorQuery(competitors=("Acme",), keywords=("AI",), max_items_per_source=5),
        sources=("Meta Ad Library", "TikTok Creative Center"),
    )

    assert len(items) == 1
    assert set(statuses["source"]) == {"Meta Ad Library", "TikTok Creative Center"}
    assert items.loc[0, "competitor"] == "Acme"
