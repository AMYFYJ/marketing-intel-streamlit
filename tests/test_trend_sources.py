from __future__ import annotations

import pandas as pd

from data_sources.trend_sources import (
    TrendQuery,
    compute_trend_summary,
    fetch_demand_pulse,
    fetch_gdelt,
    fetch_youtube,
    parse_keywords,
    recommend_campaign_angles,
    sentiment_score,
)


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.payload


def test_parse_keywords_deduplicates_and_strips() -> None:
    assert parse_keywords("AI marketing, retail media\nAI marketing") == ("AI marketing", "retail media")


def test_fetch_gdelt_parses_articles() -> None:
    def fake_get(*args, **kwargs):
        return FakeResponse({"articles": [{"title": "AI marketing growth", "url": "https://example.com", "seendate": "20260101120000", "domain": "example.com"}]})

    frame, status = fetch_gdelt("AI marketing", request_get=fake_get)

    assert status["status"] == "ok"
    assert len(frame) == 1
    assert frame.loc[0, "source"] == "GDELT"
    assert frame.loc[0, "keyword"] == "AI marketing"


def test_fetch_youtube_requires_key_and_parses_results() -> None:
    empty, status = fetch_youtube("ads", None)
    assert empty.empty
    assert status["status"] == "not configured"

    def fake_get(*args, **kwargs):
        return FakeResponse(
            {
                "items": [
                    {
                        "id": {"videoId": "abc123"},
                        "snippet": {
                            "title": "TikTok ads launch",
                            "publishedAt": "2026-01-01T12:00:00Z",
                            "description": "new popular launch",
                            "channelTitle": "Marketing Channel",
                        },
                    }
                ]
            }
        )

    frame, status = fetch_youtube("ads", "key", request_get=fake_get)

    assert status["status"] == "ok"
    assert frame.loc[0, "url"] == "https://www.youtube.com/watch?v=abc123"


def test_trend_summary_and_recommendations() -> None:
    now = pd.Timestamp.now(tz="UTC")
    items = pd.DataFrame(
        {
            "source": ["GDELT", "Reddit", "GDELT"],
            "keyword": ["retail media", "retail media", "TikTok ads"],
            "title": ["retail media growth", "retail media complaint", "TikTok ads viral"],
            "url": ["", "", ""],
            "published_at": [now, now - pd.Timedelta(days=1), now],
            "snippet": ["strong opportunity", "expensive problem", "popular new launch"],
            "author": ["a", "b", "c"],
            "engagement": [0.0, 2.0, 3.0],
            "sentiment": [1.0, -1.0, 1.0],
            "recency_hours": [1.0, 24.0, 2.0],
        }
    )

    summary = compute_trend_summary(items, lookback_days=7)
    angles = recommend_campaign_angles(summary, items)

    assert not summary.empty
    assert {"velocity", "momentum", "recent_mentions"}.issubset(summary.columns)
    assert set(summary["momentum"]).issubset({"Accelerating", "Steady", "Cooling"})
    assert set(angles["keyword"]) == {"retail media", "TikTok ads"}
    # Rationale reads in plain language rather than raw internals.
    assert "recent half" in angles["rationale"].iloc[0]


def test_trend_summary_momentum_compares_window_halves() -> None:
    now = pd.Timestamp.now(tz="UTC")
    base = {
        "source": "GDELT",
        "keyword": "retail media",
        "url": "",
        "published_at": now,
        "snippet": "",
        "author": "a",
        "engagement": 0.0,
        "sentiment": 0.0,
    }
    # Three mentions in the recent half (<= 84h for a 7-day lookback), one earlier.
    items = pd.DataFrame([{**base, "title": f"t{i}", "recency_hours": h} for i, h in enumerate([2.0, 30.0, 80.0, 150.0])])

    summary = compute_trend_summary(items, lookback_days=7)

    assert summary.loc[0, "mentions"] == 4
    assert summary.loc[0, "recent_mentions"] == 3
    assert summary.loc[0, "velocity"] == 200.0
    assert summary.loc[0, "momentum"] == "Accelerating"


def test_sentiment_score_handles_real_marketing_headlines() -> None:
    assert sentiment_score("amazing growth this quarter") > 0
    assert sentiment_score("lawsuit filed against ad platform") < 0
    assert sentiment_score("record losses and layoffs announced") < 0
    assert sentiment_score("the report was published on Tuesday") == 0


def test_fetch_demand_pulse_handles_export_only(tmp_path) -> None:
    export = tmp_path / "google_trends_export.csv"
    export.write_text("date,keyword,value\n2026-01-01,AI marketing,88\n", encoding="utf-8")

    items, statuses = fetch_demand_pulse(
        TrendQuery(keywords=("AI marketing",)),
        sources=("Google Trends export",),
        data_dir=tmp_path,
    )

    assert len(items) == 1
    assert statuses.loc[0, "status"] == "ok"


def test_fetch_gdelt_reports_rate_limit() -> None:
    def fake_get(*args, **kwargs):
        return FakeResponse({}, status_code=429)

    frame, status = fetch_gdelt("AI marketing", request_get=fake_get)

    assert frame.empty
    assert status["status"] == "rate limited"


def test_fetch_demand_pulse_handles_all_empty_sources(tmp_path) -> None:
    items, statuses = fetch_demand_pulse(
        TrendQuery(keywords=("AI marketing",)),
        sources=("Google Trends export",),
        data_dir=tmp_path,
    )

    assert items.empty
    assert statuses.loc[0, "status"] == "not configured"
