from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import quote_plus

import re

import feedparser
import numpy as np
import pandas as pd
import requests

from data_sources.api_errors import response_error_detail, strip_query_strings

# Momentum thresholds for the recent-vs-earlier mention comparison (percent).
ACCELERATING_THRESHOLD = 25.0
COOLING_THRESHOLD = -25.0

# Marketing channels detected in fetched items (first match wins, most specific
# first) so dashboards can break demand down by channel without user input.
CHANNEL_TERMS = {
    "Retail Media": ("retail media",),
    "Connected TV": ("connected tv", "ctv", "streaming ads", "streaming advertising", "addressable tv"),
    "Influencer Marketing": ("influencer", "influencers", "creator marketing", "creator economy"),
    "Paid Social": ("paid social", "social ads", "social advertising", "tiktok ads", "instagram ads", "facebook ads", "meta ads"),
    "Paid Search": ("paid search", "search ads", "search advertising", "google ads", "ppc"),
    "Email Marketing": ("email marketing", "email campaign", "email campaigns", "newsletter"),
    "Affiliate Marketing": ("affiliate", "affiliates"),
    "Out of Home": ("out of home", "billboard", "billboards", "ooh advertising"),
    "Podcast Advertising": ("podcast", "podcasts"),
}
NO_CHANNEL_LABEL = "No Channel Mention"

_CHANNEL_REGEXES = {
    label: re.compile(r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b")
    for label, terms in CHANNEL_TERMS.items()
}


def detect_channel(text: str) -> str:
    lowered = str(text).lower()
    for label, regex in _CHANNEL_REGEXES.items():
        if regex.search(lowered):
            return label
    return NO_CHANNEL_LABEL


# Broad single-word verticals are weak search terms; expand them into OR-queries
# (GDELT and Reddit both support quoted phrases and uppercase OR) for better recall.
VERTICAL_QUERY_EXPANSIONS = {
    "beauty": '(beauty OR cosmetics OR skincare OR makeup)',
    "clothing": '(clothing OR apparel OR fashion)',
    "consumer products": '("consumer products" OR "consumer goods" OR cpg)',
    "education": '(education OR edtech OR "online learning")',
    "finance": '(fintech OR banking OR "personal finance")',
    "fitness and wellness": '(fitness OR wellness OR gym)',
    "food and beverage": '("food and beverage" OR restaurants OR snacks)',
    "gaming": '(gaming OR "video games" OR esports)',
    "home and furniture": '(furniture OR "home decor" OR "home goods")',
    "live event tickets": '("concert tickets" OR "event tickets" OR ticketing)',
    "luxury": '(luxury OR "luxury brands" OR designer)',
    "music": '(music OR concerts OR musicians)',
    "pets": '(pets OR "pet food" OR "pet care")',
    "retail": '(retail OR retailers OR ecommerce)',
    "saas": '(saas OR "software as a service" OR "cloud software")',
    "service subscriptions": '(subscriptions OR "subscription service" OR "subscription business")',
    "sports": '(sports OR athletes OR sportswear)',
    "streaming services": '("streaming service" OR "streaming platform" OR "video streaming")',
    "toys": '(toys OR "toy industry" OR "toy brands")',
    "travel": '(travel OR tourism OR airlines)',
}


def expand_search_query(keyword: str) -> str:
    return VERTICAL_QUERY_EXPANSIONS.get(str(keyword).strip().lower(), keyword)

_VADER_ANALYZER = None


def _vader():
    global _VADER_ANALYZER
    if _VADER_ANALYZER is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        _VADER_ANALYZER = SentimentIntensityAnalyzer()
    return _VADER_ANALYZER


@dataclass(frozen=True)
class TrendQuery:
    keywords: tuple[str, ...]
    lookback_days: int = 7
    max_items_per_source: int = 25
    market: str = "US"


# Live feeds whose results must respect the lookback window; export files and
# Meta Ad Library (where long-running active ads are current demand) are exempt.
WINDOWED_SOURCES = ("GDELT", "Reddit", "YouTube")


def parse_keywords(raw: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(raw, str):
        pieces = raw.replace("\n", ",").split(",")
    else:
        pieces = list(raw)
    cleaned = []
    seen = set()
    for piece in pieces:
        keyword = str(piece).strip()
        if keyword and keyword.lower() not in seen:
            cleaned.append(keyword)
            seen.add(keyword.lower())
    return tuple(cleaned)


def fetch_demand_pulse(
    query: TrendQuery,
    sources: tuple[str, ...] = ("GDELT", "Reddit"),
    youtube_api_key: str | None = None,
    data_dir: str | Path = "data",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    statuses: list[dict[str, str]] = []

    for keyword in query.keywords:
        search_query = expand_search_query(keyword)
        if "GDELT" in sources:
            frame, status = fetch_gdelt(keyword, query.max_items_per_source, query.lookback_days, search_query=search_query)
            frames.append(frame)
            statuses.append(status)
        if "Reddit" in sources:
            frame, status = fetch_reddit(keyword, query.max_items_per_source, query.lookback_days, search_query=search_query)
            frames.append(frame)
            statuses.append(status)
        if "YouTube" in sources:
            frame, status = fetch_youtube(keyword, youtube_api_key, query.max_items_per_source, query.lookback_days)
            frames.append(frame)
            statuses.append(status)

    if "Google Trends export" in sources:
        frame, status = load_trends_export(Path(data_dir) / "google_trends_export.csv", "Google Trends export")
        frames.append(frame)
        statuses.append(status)
    if "Pinterest export" in sources:
        frame, status = load_trends_export(Path(data_dir) / "pinterest_trends_export.csv", "Pinterest export")
        frames.append(frame)
        statuses.append(status)

    non_empty_frames = [frame for frame in frames if not frame.empty]
    combined = pd.concat(non_empty_frames, ignore_index=True) if non_empty_frames else empty_trend_frame()
    combined = enrich_trend_items(combined, query.lookback_days)
    combined = filter_to_lookback(combined, query.lookback_days)
    return combined, pd.DataFrame(statuses)


def enrich_trend_items(frame: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Attach published_at/sentiment/channel/recency columns to raw trend rows."""
    if frame.empty:
        return frame
    enriched = frame.copy()
    enriched["published_at"] = pd.to_datetime(enriched["published_at"], errors="coerce", utc=True)
    enriched["sentiment"] = enriched.apply(lambda row: sentiment_score(f"{row['title']} {row['snippet']}"), axis=1)
    enriched["channel"] = enriched.apply(lambda row: detect_channel(f"{row['title']} {row['snippet']}"), axis=1)
    enriched["recency_hours"] = (pd.Timestamp.now(tz="UTC") - enriched["published_at"]).dt.total_seconds() / 3600
    enriched["recency_hours"] = enriched["recency_hours"].clip(lower=0).fillna(lookback_days * 24)
    return enriched


def filter_to_lookback(frame: pd.DataFrame, lookback_days: int) -> pd.DataFrame:
    """Drop live-feed rows older than the lookback window so charts honor it."""
    if frame.empty:
        return frame
    windowed = frame["source"].isin(WINDOWED_SOURCES)
    within = frame["recency_hours"] <= lookback_days * 24
    return frame[~windowed | within].reset_index(drop=True)


# GDELT allows roughly one request every five seconds; pace sequential calls so
# multi-keyword refreshes don't lose every keyword after the first to 429s.
GDELT_MIN_INTERVAL_SECONDS = 5.0
_last_gdelt_call = 0.0


def _respect_gdelt_rate_limit() -> None:
    global _last_gdelt_call
    wait = GDELT_MIN_INTERVAL_SECONDS - (time.monotonic() - _last_gdelt_call)
    if wait > 0:
        time.sleep(wait)
    _last_gdelt_call = time.monotonic()


def fetch_gdelt(
    keyword: str,
    max_records: int = 25,
    lookback_days: int = 7,
    search_query: str | None = None,
    request_get: Callable[..., object] = requests.get,
) -> tuple[pd.DataFrame, dict[str, str]]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": search_query or keyword,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": min(max_records, 250),
        # Restrict results to the lookback window so volume charts reflect it.
        "timespan": f"{max(int(lookback_days), 1)}d",
        "sort": "HybridRel",
    }
    live_request = request_get is requests.get
    try:
        if live_request:
            _respect_gdelt_rate_limit()
        response = request_get(url, params=params, timeout=12)
        status_code = getattr(response, "status_code", 200)
        if status_code == 429 and live_request:
            # One paced retry before giving up on this keyword.
            time.sleep(GDELT_MIN_INTERVAL_SECONDS)
            response = request_get(url, params=params, timeout=12)
            status_code = getattr(response, "status_code", 200)
        if status_code == 429:
            return empty_trend_frame(), _status("GDELT", keyword, "rate limited", "GDELT allows roughly one request every five seconds")
        response.raise_for_status()
        payload = response.json()
        articles = payload.get("articles", [])
    except ValueError as exc:
        return empty_trend_frame(), _status("GDELT", keyword, "failed", f"Non-JSON response: {exc}")
    except Exception as exc:  # pragma: no cover - exercised through status behavior in app
        return empty_trend_frame(), _status("GDELT", keyword, "failed", str(exc))

    rows = []
    for article in articles[:max_records]:
        rows.append(
            {
                "source": "GDELT",
                "keyword": keyword,
                "title": article.get("title") or "Untitled article",
                "url": article.get("url") or "",
                "published_at": _parse_datetime(article.get("seendate")),
                "snippet": article.get("domain") or article.get("sourcecountry") or "",
                "author": article.get("domain") or "",
                "engagement": 0.0,
            }
        )
    return pd.DataFrame(rows, columns=empty_trend_frame().columns), _status("GDELT", keyword, "ok", f"{len(rows)} articles")


def fetch_reddit(keyword: str, max_records: int = 25, lookback_days: int = 7, search_query: str | None = None) -> tuple[pd.DataFrame, dict[str, str]]:
    encoded = quote_plus(search_query or keyword)
    window = "day" if lookback_days <= 1 else ("week" if lookback_days <= 7 else "month")
    url = f"https://www.reddit.com/search.rss?q={encoded}&sort=new&t={window}&limit={min(max_records, 100)}"
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "marketing-intel-streamlit/1.0"})
        entries = feed.entries[:max_records]
        # Report blocked or malformed responses as failures instead of "ok, 0 posts".
        http_status = int(getattr(feed, "status", 200) or 200)
        if http_status >= 400:
            return empty_trend_frame(), _status("Reddit", keyword, "failed", f"HTTP {http_status} from Reddit RSS (often IP blocking on cloud hosts)")
        if not entries and getattr(feed, "bozo", False):
            return empty_trend_frame(), _status("Reddit", keyword, "failed", f"Unparseable feed: {getattr(feed, 'bozo_exception', 'unknown error')}")
    except Exception as exc:  # pragma: no cover
        return empty_trend_frame(), _status("Reddit", keyword, "failed", str(exc))

    rows = []
    for entry in entries:
        rows.append(
            {
                "source": "Reddit",
                "keyword": keyword,
                "title": entry.get("title", "Untitled Reddit post"),
                "url": entry.get("link", ""),
                "published_at": _parse_struct_time(entry.get("published_parsed")),
                "snippet": _strip_html(entry.get("summary", ""))[:350],
                "author": entry.get("author", ""),
                "engagement": float(entry.get("score", 0) or 0),
            }
        )
    return pd.DataFrame(rows, columns=empty_trend_frame().columns), _status("Reddit", keyword, "ok", f"{len(rows)} posts")


def fetch_youtube(keyword: str, api_key: str | None, max_records: int = 25, lookback_days: int = 7, request_get: Callable[..., object] = requests.get) -> tuple[pd.DataFrame, dict[str, str]]:
    if not api_key:
        return empty_trend_frame(), _status("YouTube", keyword, "not configured", "Set YOUTUBE_API_KEY in Streamlit secrets")
    url = "https://www.googleapis.com/youtube/v3/search"
    published_after = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max(int(lookback_days), 1))).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "part": "snippet",
        "q": keyword,
        "type": "video",
        "order": "date",
        "publishedAfter": published_after,
        "maxResults": min(max_records, 50),
        "key": api_key,
    }
    try:
        response = request_get(url, params=params, timeout=12)
        status_code = getattr(response, "status_code", 200)
        if status_code >= 400:
            # Surface the API's own explanation; never echo the request URL, which carries the API key.
            return empty_trend_frame(), _status("YouTube", keyword, "failed", response_error_detail(response, status_code))
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # pragma: no cover
        return empty_trend_frame(), _status("YouTube", keyword, "failed", strip_query_strings(str(exc)))

    rows = []
    for item in payload.get("items", [])[:max_records]:
        snippet = item.get("snippet", {})
        video_id = item.get("id", {}).get("videoId", "")
        rows.append(
            {
                "source": "YouTube",
                "keyword": keyword,
                "title": snippet.get("title", "Untitled video"),
                "url": f"https://www.youtube.com/watch?v={video_id}" if video_id else "",
                "published_at": _parse_datetime(snippet.get("publishedAt")),
                "snippet": snippet.get("description", ""),
                "author": snippet.get("channelTitle", ""),
                "engagement": 0.0,
            }
        )
    return pd.DataFrame(rows, columns=empty_trend_frame().columns), _status("YouTube", keyword, "ok", f"{len(rows)} videos")


def load_trends_export(path: Path, source_name: str) -> tuple[pd.DataFrame, dict[str, str]]:
    if not path.exists():
        return empty_trend_frame(), _status(source_name, "export", "not configured", f"Missing {path}")
    raw = pd.read_csv(path)
    columns = {column.lower().strip().replace(" ", "_"): column for column in raw.columns}
    keyword_col = columns.get("keyword") or columns.get("query") or columns.get("term")
    title_col = columns.get("title") or keyword_col
    value_col = columns.get("value") or columns.get("interest") or columns.get("score")
    date_col = columns.get("date") or columns.get("published_at")
    rows = []
    for _, row in raw.iterrows():
        keyword = str(row[keyword_col]) if keyword_col else "export"
        rows.append(
            {
                "source": source_name,
                "keyword": keyword,
                "title": str(row[title_col]) if title_col else keyword,
                "url": "",
                "published_at": _parse_datetime(row[date_col]) if date_col else pd.Timestamp.now(tz="UTC"),
                "snippet": "Imported trend export",
                "author": source_name,
                "engagement": float(row[value_col]) if value_col else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=empty_trend_frame().columns), _status(source_name, "export", "ok", f"{len(rows)} rows")


def compute_trend_summary(items: pd.DataFrame, lookback_days: int = 7) -> pd.DataFrame:
    """Per keyword/source: mention counts plus momentum.

    Velocity is the percent change in mentions between the most recent half of
    the lookback window and the earlier half, labeled Accelerating / Steady /
    Cooling so the number reads directly.
    """
    columns = ["keyword", "source", "mentions", "recent_mentions", "velocity", "momentum", "avg_sentiment", "engagement"]
    if items.empty:
        return pd.DataFrame(columns=columns)
    df = items.copy()
    half_window_hours = lookback_days * 24 / 2
    df["is_recent"] = df["recency_hours"].fillna(lookback_days * 24) <= half_window_hours
    summary = df.groupby(["keyword", "source"], as_index=False).agg(
        mentions=("title", "count"),
        recent_mentions=("is_recent", "sum"),
        avg_sentiment=("sentiment", "mean"),
        engagement=("engagement", "sum"),
    )
    earlier = summary["mentions"] - summary["recent_mentions"]
    summary["velocity"] = np.where(
        earlier > 0,
        (summary["recent_mentions"] - earlier) / earlier * 100,
        np.where(summary["recent_mentions"] > 0, 100.0, 0.0),
    )
    summary["momentum"] = np.select(
        [summary["velocity"] >= ACCELERATING_THRESHOLD, summary["velocity"] <= COOLING_THRESHOLD],
        ["Accelerating", "Cooling"],
        default="Steady",
    )
    return summary[columns].sort_values(["velocity", "mentions"], ascending=False)


def summarize_channels(items: pd.DataFrame) -> pd.DataFrame:
    """Mentions of specific marketing channels per keyword, for the channel breakdown chart."""
    if items.empty or "channel" not in items.columns:
        return pd.DataFrame(columns=["channel", "keyword", "mentions", "avg_sentiment"])
    classified = items[items["channel"] != NO_CHANNEL_LABEL]
    if classified.empty:
        return pd.DataFrame(columns=["channel", "keyword", "mentions", "avg_sentiment"])
    return (
        classified.groupby(["channel", "keyword"], as_index=False)
        .agg(mentions=("title", "count"), avg_sentiment=("sentiment", "mean"))
        .sort_values("mentions", ascending=False)
    )


def recommend_campaign_angles(summary: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=["keyword", "angle", "rationale"])
    rows = []
    for keyword, group in summary.groupby("keyword"):
        mentions = int(group["mentions"].sum())
        recent = int(group["recent_mentions"].sum())
        earlier = mentions - recent
        velocity = ((recent - earlier) / earlier * 100) if earlier > 0 else (100.0 if recent > 0 else 0.0)
        sentiment = float(group["avg_sentiment"].mean())
        top_source = group.sort_values("mentions", ascending=False).iloc[0]["source"]
        momentum = "accelerating" if velocity >= ACCELERATING_THRESHOLD else ("cooling" if velocity <= COOLING_THRESHOLD else "steady")
        tone = "positive" if sentiment > 0.15 else ("negative" if sentiment < -0.15 else "neutral")
        if sentiment > 0.15:
            angle = "Lean into proof and momentum"
        elif sentiment < -0.15:
            angle = "Address objections directly"
        elif velocity >= ACCELERATING_THRESHOLD:
            angle = "Launch timely educational creative"
        else:
            angle = "Monitor and test low-budget creative"
        rows.append(
            {
                "keyword": keyword,
                "angle": angle,
                "rationale": (
                    f"{mentions} mentions, {momentum} ({velocity:+.0f}% recent half vs earlier half) "
                    f"with {tone} sentiment ({sentiment:+.2f}); most coverage from {top_source}."
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("keyword")


def sentiment_score(text: str) -> float:
    """VADER compound score in [-1, 1]; built for short social/news text."""
    return float(_vader().polarity_scores(str(text))["compound"])


def empty_trend_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["source", "keyword", "title", "url", "published_at", "snippet", "author", "engagement"])


def _status(source: str, keyword: str, status: str, detail: str) -> dict[str, str]:
    return {"source": source, "keyword": keyword, "status": status, "detail": detail}


def _parse_datetime(value: object) -> pd.Timestamp:
    if value is None or value == "":
        return pd.Timestamp.now(tz="UTC")
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return pd.Timestamp.now(tz="UTC")
    return parsed


def _parse_struct_time(value: object) -> pd.Timestamp:
    if not value:
        return pd.Timestamp.now(tz="UTC")
    return pd.Timestamp(datetime(*value[:6], tzinfo=timezone.utc))


def _strip_html(value: str) -> str:
    text = str(value).replace("<br/>", " ").replace("<br>", " ")
    while "<" in text and ">" in text:
        start = text.find("<")
        end = text.find(">", start)
        if end == -1:
            break
        text = text[:start] + " " + text[end + 1 :]
    return " ".join(text.split())
