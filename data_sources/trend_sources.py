from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import quote_plus

import feedparser
import pandas as pd
import requests

POSITIVE_WORDS = {"growth", "launch", "win", "best", "viral", "increase", "strong", "new", "popular", "opportunity", "breakthrough"}
NEGATIVE_WORDS = {"drop", "decline", "bad", "risk", "problem", "complaint", "down", "weak", "expensive", "delay", "controversy"}


@dataclass(frozen=True)
class TrendQuery:
    keywords: tuple[str, ...]
    lookback_days: int = 7
    max_items_per_source: int = 25
    market: str = "US"


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
        if "GDELT" in sources:
            frame, status = fetch_gdelt(keyword, query.max_items_per_source)
            frames.append(frame)
            statuses.append(status)
        if "Reddit" in sources:
            frame, status = fetch_reddit(keyword, query.max_items_per_source, query.lookback_days)
            frames.append(frame)
            statuses.append(status)
        if "YouTube" in sources:
            frame, status = fetch_youtube(keyword, youtube_api_key, query.max_items_per_source)
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
    if not combined.empty:
        combined["published_at"] = pd.to_datetime(combined["published_at"], errors="coerce", utc=True)
        combined["sentiment"] = combined.apply(lambda row: sentiment_score(f"{row['title']} {row['snippet']}"), axis=1)
        combined["recency_hours"] = (pd.Timestamp.now(tz="UTC") - combined["published_at"]).dt.total_seconds() / 3600
        combined["recency_hours"] = combined["recency_hours"].clip(lower=0).fillna(query.lookback_days * 24)
    return combined, pd.DataFrame(statuses)


def fetch_gdelt(keyword: str, max_records: int = 25, request_get: Callable[..., object] = requests.get) -> tuple[pd.DataFrame, dict[str, str]]:
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": keyword,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "sort": "HybridRel",
    }
    try:
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


def fetch_reddit(keyword: str, max_records: int = 25, lookback_days: int = 7) -> tuple[pd.DataFrame, dict[str, str]]:
    encoded = quote_plus(keyword)
    url = f"https://www.reddit.com/search.rss?q={encoded}&sort=new&t=week"
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "marketing-intel-streamlit/1.0"})
        entries = feed.entries[:max_records]
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


def fetch_youtube(keyword: str, api_key: str | None, max_records: int = 25, request_get: Callable[..., object] = requests.get) -> tuple[pd.DataFrame, dict[str, str]]:
    if not api_key:
        return empty_trend_frame(), _status("YouTube", keyword, "not configured", "Set YOUTUBE_API_KEY in Streamlit secrets")
    url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": keyword,
        "type": "video",
        "order": "date",
        "maxResults": min(max_records, 50),
        "key": api_key,
    }
    try:
        response = request_get(url, params=params, timeout=12)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # pragma: no cover
        return empty_trend_frame(), _status("YouTube", keyword, "failed", str(exc))

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


def compute_trend_summary(items: pd.DataFrame) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame(columns=["keyword", "source", "mentions", "velocity", "avg_sentiment", "engagement"])
    df = items.copy()
    df["recency_weight"] = 1 / (1 + df["recency_hours"].fillna(168) / 24)
    summary = df.groupby(["keyword", "source"], as_index=False).agg(
        mentions=("title", "count"),
        velocity=("recency_weight", "sum"),
        avg_sentiment=("sentiment", "mean"),
        engagement=("engagement", "sum"),
    )
    return summary.sort_values(["velocity", "mentions"], ascending=False)


def recommend_campaign_angles(summary: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame(columns=["keyword", "angle", "rationale"])
    rows = []
    for keyword, group in summary.groupby("keyword"):
        velocity = float(group["velocity"].sum())
        sentiment = float(group["avg_sentiment"].mean())
        top_source = group.sort_values("mentions", ascending=False).iloc[0]["source"]
        if sentiment > 0.15:
            angle = "Lean into proof and momentum"
        elif sentiment < -0.15:
            angle = "Address objections directly"
        elif velocity >= summary["velocity"].quantile(0.70):
            angle = "Launch timely educational creative"
        else:
            angle = "Monitor and test low-budget creative"
        rows.append(
            {
                "keyword": keyword,
                "angle": angle,
                "rationale": f"{top_source} is the leading source with velocity {velocity:.1f} and sentiment {sentiment:.2f}.",
            }
        )
    return pd.DataFrame(rows).sort_values("keyword")


def sentiment_score(text: str) -> float:
    words = {word.strip(".,!?;:()[]{}\"'").lower() for word in str(text).split()}
    positive = len(words.intersection(POSITIVE_WORDS))
    negative = len(words.intersection(NEGATIVE_WORDS))
    total = positive + negative
    if total == 0:
        return 0.0
    return (positive - negative) / total


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
