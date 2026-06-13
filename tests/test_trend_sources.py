from __future__ import annotations

import pandas as pd

from data_sources.trend_sources import (
    NO_CHANNEL_LABEL,
    TrendQuery,
    build_daily_series,
    compute_trend_summary,
    detect_channel,
    expand_search_query,
    fetch_demand_pulse,
    fetch_gdelt,
    fetch_gdelt_timeline,
    fetch_youtube,
    filter_to_lookback,
    parse_keywords,
    recommend_campaign_angles,
    sentiment_score,
    summarize_channels,
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


def test_fetch_gdelt_parses_articles_and_restricts_to_lookback() -> None:
    captured = {}

    def fake_get(url, params=None, **kwargs):
        captured.update(params or {})
        return FakeResponse({"articles": [{"title": "AI marketing growth", "url": "https://example.com", "seendate": "20260101120000", "domain": "example.com"}]})

    frame, status = fetch_gdelt("AI marketing", max_records=500, lookback_days=14, request_get=fake_get)

    assert status["status"] == "ok"
    assert len(frame) == 1
    assert frame.loc[0, "source"] == "GDELT"
    assert frame.loc[0, "keyword"] == "AI marketing"
    assert captured["timespan"] == "14d"
    assert captured["maxrecords"] == 250  # GDELT hard cap


def test_expand_search_query_broadens_verticals() -> None:
    assert "cosmetics" in expand_search_query("Beauty")
    assert "OR" in expand_search_query("retail")
    # Unknown keywords pass through unchanged.
    assert expand_search_query("pickleball gear") == "pickleball gear"


def test_filter_to_lookback_drops_stale_live_rows_only() -> None:
    frame = pd.DataFrame(
        {
            "source": ["GDELT", "Reddit", "Google Trends export", "Meta Ad Library"],
            "recency_hours": [300.0, 12.0, 999.0, 999.0],
        }
    )

    kept = filter_to_lookback(frame, lookback_days=7)

    # Stale GDELT row dropped; fresh Reddit kept; exports and Meta ads exempt.
    assert set(kept["source"]) == {"Reddit", "Google Trends export", "Meta Ad Library"}


def test_fetch_gdelt_applies_lookback_window_params() -> None:
    captured_params = {}

    def fake_get(*args, **kwargs):
        captured_params.update(kwargs["params"])
        return FakeResponse({"articles": []})

    fetch_gdelt(
        "AI marketing",
        request_get=fake_get,
        lookback_days=3,
        now=pd.Timestamp("2026-02-10T12:00:00Z"),
    )

    assert captured_params["startdatetime"] == "20260207120000"
    assert captured_params["enddatetime"] == "20260210120000"


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


def test_fetch_youtube_applies_published_after() -> None:
    captured_params = {}

    def fake_get(*args, **kwargs):
        captured_params.update(kwargs["params"])
        return FakeResponse({"items": []})

    fetch_youtube(
        "ads",
        "key",
        request_get=fake_get,
        lookback_days=2,
        now=pd.Timestamp("2026-02-10T12:00:00Z"),
    )

    assert captured_params["publishedAfter"] == "2026-02-08T12:00:00Z"


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


def test_detect_channel_classifies_marketing_channels() -> None:
    assert detect_channel("Beauty brands shift budgets into retail media networks") == "Retail Media"
    assert detect_channel("Why gaming studios bet on influencer partnerships") == "Influencer Marketing"
    assert detect_channel("Connected TV ad spending hits record highs") == "Connected TV"
    assert detect_channel("Quarterly results beat expectations") == NO_CHANNEL_LABEL
    # Word-boundary matching: 'ppc' must not match inside other words.
    assert detect_channel("the hippopotamus escaped") == NO_CHANNEL_LABEL


def test_summarize_channels_counts_classified_items_only() -> None:
    items = pd.DataFrame(
        {
            "keyword": ["Retail", "Retail", "Beauty"],
            "title": ["a", "b", "c"],
            "channel": ["Retail Media", NO_CHANNEL_LABEL, "Influencer Marketing"],
            "sentiment": [0.2, 0.0, -0.1],
        }
    )

    summary = summarize_channels(items)

    assert int(summary["mentions"].sum()) == 2
    assert set(summary["channel"]) == {"Retail Media", "Influencer Marketing"}


def test_fetch_demand_pulse_handles_export_only(tmp_path) -> None:
    export = tmp_path / "google_trends_export.csv"
    export.write_text("date,keyword,value\n2026-01-01,AI marketing,88\n", encoding="utf-8")

    items, statuses = fetch_demand_pulse(
        TrendQuery(keywords=("AI marketing",)),
        sources=("Google Trends export",),
        data_dir=tmp_path,
        now=pd.Timestamp("2026-01-05T00:00:00Z"),
    )

    assert len(items) == 1
    assert statuses.loc[0, "status"] == "ok"


def test_fetch_demand_pulse_filters_exports_by_keyword_and_lookback(tmp_path) -> None:
    export = tmp_path / "google_trends_export.csv"
    export.write_text(
        "date,keyword,value\n"
        "2026-02-09,AI marketing,88\n"
        "2026-01-01,AI marketing,44\n"
        "2026-02-09,retail media,77\n",
        encoding="utf-8",
    )

    items, statuses = fetch_demand_pulse(
        TrendQuery(keywords=("AI marketing",), lookback_days=7),
        sources=("Google Trends export",),
        data_dir=tmp_path,
        now=pd.Timestamp("2026-02-10T12:00:00Z"),
    )

    assert len(items) == 1
    assert items.loc[0, "keyword"] == "AI marketing"
    assert items.loc[0, "engagement"] == 88
    assert statuses.loc[0, "detail"] == "1 rows after filters"


def test_fetch_gdelt_reports_rate_limit() -> None:
    def fake_get(*args, **kwargs):
        return FakeResponse({}, status_code=429)

    frame, status = fetch_gdelt("AI marketing", request_get=fake_get)

    assert frame.empty
    assert status["status"] == "rate limited"


def test_fetch_gdelt_timeline_parses_series() -> None:
    def fake_get(*args, **kwargs):
        return FakeResponse(
            {
                "timeline": [
                    {
                        "series": "Volume Intensity",
                        "data": [
                            {"date": "20260101T120000Z", "value": 5, "norm": 100},
                            {"date": "20260102T120000Z", "value": 8, "norm": 100},
                        ],
                    }
                ]
            }
        )

    frame, status = fetch_gdelt_timeline("AI marketing", lookback_days=7, request_get=fake_get)

    assert status["status"] == "ok"
    assert len(frame) == 2
    assert list(frame.columns) == ["date", "keyword", "volume", "norm"]
    assert frame.loc[0, "volume"] == 5.0
    assert frame.loc[0, "keyword"] == "AI marketing"


def test_fetch_gdelt_timeline_reports_rate_limit() -> None:
    def fake_get(*args, **kwargs):
        return FakeResponse({}, status_code=429)

    frame, status = fetch_gdelt_timeline("AI marketing", request_get=fake_get)

    assert frame.empty
    assert status["status"] == "rate limited"


def test_build_daily_series_buckets_items_per_day() -> None:
    now = pd.Timestamp("2026-01-03 09:00", tz="UTC")
    items = pd.DataFrame(
        {
            "source": ["GDELT", "Reddit", "GDELT"],
            "keyword": ["ai marketing", "ai marketing", "ai marketing"],
            "title": ["a", "b", "c"],
            "published_at": [now, now, now - pd.Timedelta(days=1)],
        }
    )

    series = build_daily_series(items)

    assert list(series.columns) == ["date", "keyword", "volume", "norm"]
    assert series["volume"].sum() == 3
    # Two items share the most recent day, one falls on the prior day.
    assert set(series["volume"]) == {1, 2}


def test_fetch_demand_pulse_handles_all_empty_sources(tmp_path) -> None:
    items, statuses = fetch_demand_pulse(
        TrendQuery(keywords=("AI marketing",)),
        sources=("Google Trends export",),
        data_dir=tmp_path,
    )

    assert items.empty
    assert statuses.loc[0, "status"] == "not configured"


def test_fetch_youtube_error_uses_response_message_and_hides_key() -> None:
    error_payload = {"error": {"code": 403, "message": "The request is missing a valid API key."}}

    frame, status = fetch_youtube("ads", "SECRET-KEY", request_get=lambda *a, **k: FakeResponse(error_payload, status_code=403))

    assert frame.empty
    assert status["status"] == "failed"
    assert "missing a valid API key" in status["detail"]
    assert "SECRET-KEY" not in status["detail"]
