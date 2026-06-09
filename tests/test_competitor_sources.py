from __future__ import annotations

import pandas as pd

from data_sources.competitor_sources import (
    CompetitorQuery,
    MARKET_WIDE_COMPETITOR,
    analyze_creative_patterns,
    build_strategy_recommendations,
    build_theme_cta_matrix,
    compute_share_of_voice,
    detect_cta,
    detect_theme,
    enrich_competitor_items,
    empty_competitor_frame,
    fetch_competitor_intelligence,
    fetch_linkedin_ad_library_link,
    fetch_meta_ad_library,
    fetch_tiktok_creative_center_link,
    fetch_x_ads_repository_link,
    parse_competitors,
    summarize_competitive_signals,
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


def test_linkedin_ad_library_returns_live_link_item() -> None:
    frame, status = fetch_linkedin_ad_library_link("Acme AI", "Acme")

    assert status["status"] == "live link"
    assert len(frame) == 1
    assert frame.loc[0, "source"] == "LinkedIn Ad Library"
    assert "linkedin.com/ads/library" in frame.loc[0, "url"]
    assert frame.loc[0, "platforms"] == "LinkedIn"


def test_x_ads_repository_returns_eu_only_live_link_item() -> None:
    frame, status = fetch_x_ads_repository_link("Acme AI", "Acme")

    assert status["status"] == "live link"
    assert len(frame) == 1
    assert frame.loc[0, "source"] == "X Ads Repository (EU Only)"
    assert "ads.twitter.com/ads-repository" in frame.loc[0, "url"]
    assert "EU Digital Services Act" in frame.loc[0, "text"]


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
        sources=("Meta Ad Library", "TikTok Creative Center", "LinkedIn Ad Library", "X Ads Repository (EU Only)"),
    )

    assert len(items) == 3
    assert set(statuses["source"]) == {
        "Meta Ad Library",
        "TikTok Creative Center",
        "LinkedIn Ad Library",
        "X Ads Repository (EU Only)",
    }
    assert items.loc[0, "competitor"] == "Acme"


def test_fetch_competitor_intelligence_supports_market_wide_theme_scan() -> None:
    items, statuses = fetch_competitor_intelligence(
        CompetitorQuery(competitors=(), keywords=("beauty",), max_items_per_source=5),
        sources=("TikTok Creative Center",),
    )

    assert len(items) == 1
    assert items.loc[0, "competitor"] == MARKET_WIDE_COMPETITOR
    assert items.loc[0, "keyword"] == "beauty"
    assert statuses.loc[0, "keyword"] == "beauty"


def test_fetch_competitor_intelligence_handles_all_empty_sources() -> None:
    items, statuses = fetch_competitor_intelligence(
        CompetitorQuery(competitors=("Acme",), keywords=("AI",), max_items_per_source=5),
        sources=("Meta Ad Library",),
    )

    assert items.empty
    assert statuses.loc[0, "status"] == "not configured"


def test_enrich_competitor_items_adds_decision_fields() -> None:
    items = pd.DataFrame(
        [
            {
                "source": "Meta Ad Library",
                "competitor": "Acme",
                "keyword": "Acme AI",
                "asset_type": "Ad",
                "title": "Start free trial for new AI workflow automation",
                "text": "Save time with AI automation and trusted workflow proof.",
                "url": "https://example.com/ad",
                "published_at": "2026-01-25",
                "author": "Acme",
                "platforms": "facebook, instagram",
                "engagement": 12.0,
            }
        ],
        columns=empty_competitor_frame().columns,
    )
    statuses = pd.DataFrame(
        [{"source": "Meta Ad Library", "keyword": "Acme AI", "status": "ok", "detail": "1 ad"}]
    )

    enriched = enrich_competitor_items(items, statuses, now=pd.Timestamp("2026-02-01T00:00:00Z"))

    assert enriched.loc[0, "source_confidence_label"] == "Direct source"
    assert enriched.loc[0, "creative_format"] == "Ad Format Unknown"
    assert enriched.loc[0, "campaign_type"] == "Lead Gen"
    assert enriched.loc[0, "creative_angle"] == "Problem / Solution"
    assert enriched.loc[0, "priority"] == "High"
    assert enriched.loc[0, "recommended_action"] == "Test next"
    assert enriched.loc[0, "freshness_days"] == 7
    assert enriched.loc[0, "signal_strength"] > 70


def test_enrich_competitor_items_marks_setup_gaps() -> None:
    items = pd.DataFrame(
        [
            {
                "source": "Meta Ad Library",
                "competitor": "Acme",
                "keyword": "Acme AI",
                "asset_type": "Ad",
                "title": "AI launch",
                "text": "Learn more about AI automation.",
                "url": "https://example.com/ad",
                "published_at": "2026-01-25",
                "author": "Acme",
                "platforms": "facebook",
                "engagement": 0.0,
            }
        ],
        columns=empty_competitor_frame().columns,
    )
    statuses = pd.DataFrame(
        [{"source": "Meta Ad Library", "keyword": "Acme AI", "status": "not configured", "detail": "missing token"}]
    )

    enriched = enrich_competitor_items(items, statuses, now=pd.Timestamp("2026-02-01T00:00:00Z"))

    assert enriched.loc[0, "priority"] == "Fix source"
    assert enriched.loc[0, "recommended_action"] == "Fix source"
    assert enriched.loc[0, "source_confidence_label"] == "Needs setup"


def test_signal_summaries_matrix_and_recommendations_are_stable() -> None:
    first, first_status = fetch_tiktok_creative_center_link("Acme AI", "Acme")
    second, _ = fetch_tiktok_creative_center_link("Beta automation", "Beta")
    items = pd.concat([first, second], ignore_index=True)
    statuses = pd.DataFrame(
        [
            first_status,
            {"source": "TikTok Creative Center", "keyword": "Beta automation", "status": "live link", "detail": "link"},
        ]
    )

    enriched = enrich_competitor_items(items, statuses, now=pd.Timestamp("2026-02-01T00:00:00Z"))
    summary = summarize_competitive_signals(enriched, statuses)
    matrix = build_theme_cta_matrix(enriched)
    recommendations = build_strategy_recommendations(enriched)

    assert summary["items"] == 2
    assert summary["active_sources"] == 1
    assert summary["source_gaps"] == 0
    assert not matrix.empty
    assert set(recommendations["recommended_action"]) == {"Open source"}
