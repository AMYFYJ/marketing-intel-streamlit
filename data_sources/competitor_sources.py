from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import quote_plus

import pandas as pd
import requests

from data_sources.trend_sources import fetch_gdelt, fetch_reddit, fetch_youtube, parse_keywords, sentiment_score

CTA_PATTERNS = {
    "Book demo": ("book demo", "request demo", "schedule demo"),
    "Shop now": ("shop now", "buy now", "limited time", "sale"),
    "Learn more": ("learn more", "read more", "see how"),
    "Free trial": ("free trial", "start free", "try free"),
    "Download": ("download", "get the guide", "get report"),
    "Sign up": ("sign up", "join now", "subscribe"),
}

THEME_TERMS = {
    "AI": ("ai", "automation", "machine learning", "agent"),
    "Efficiency": ("save time", "efficient", "productivity", "workflow"),
    "Discount": ("discount", "sale", "offer", "limited time"),
    "Trust": ("trusted", "secure", "proof", "case study"),
    "Launch": ("new", "launch", "introducing", "announcement"),
    "Education": ("guide", "webinar", "learn", "tips"),
}

LIVE_LINK_ASSET_TYPE = "Live search link"


def _compile_label_patterns(label_terms: dict[str, tuple[str, ...]]) -> dict[str, re.Pattern[str]]:
    # Word-boundary matching so "sale" does not match "Salesforce" and "ai" does not match "email".
    return {
        label: re.compile(r"\b(?:" + "|".join(re.escape(term) for term in terms) + r")\b")
        for label, terms in label_terms.items()
    }


_CTA_REGEXES = _compile_label_patterns(CTA_PATTERNS)
_THEME_REGEXES = _compile_label_patterns(THEME_TERMS)


@dataclass(frozen=True)
class CompetitorQuery:
    competitors: tuple[str, ...]
    keywords: tuple[str, ...] = ()
    country: str = "US"
    max_items_per_source: int = 20


def parse_competitors(raw: str | Iterable[str]) -> tuple[str, ...]:
    return parse_keywords(raw)


def fetch_competitor_intelligence(
    query: CompetitorQuery,
    sources: tuple[str, ...] = ("Meta Ad Library", "TikTok Creative Center", "Reddit", "GDELT"),
    meta_access_token: str | None = None,
    meta_api_version: str = "v21.0",
    youtube_api_key: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    statuses: list[dict[str, str]] = []
    search_terms = _search_terms(query)

    for competitor, term in search_terms:
        if "Meta Ad Library" in sources:
            frame, status = fetch_meta_ad_library(
                term,
                competitor=competitor,
                access_token=meta_access_token,
                country=query.country,
                api_version=meta_api_version,
                max_records=query.max_items_per_source,
            )
            frames.append(frame)
            statuses.append(status)
        if "TikTok Creative Center" in sources:
            frame, status = fetch_tiktok_creative_center_link(term, competitor)
            frames.append(frame)
            statuses.append(status)
        if "YouTube" in sources:
            trend_frame, status = fetch_youtube(term, youtube_api_key, query.max_items_per_source)
            frames.append(_trend_to_competitor_frame(trend_frame, competitor, "Video"))
            statuses.append(status)
        if "Reddit" in sources:
            trend_frame, status = fetch_reddit(term, query.max_items_per_source)
            frames.append(_trend_to_competitor_frame(trend_frame, competitor, "Social mention"))
            statuses.append(status)
        if "GDELT" in sources:
            trend_frame, status = fetch_gdelt(term, query.max_items_per_source)
            frames.append(_trend_to_competitor_frame(trend_frame, competitor, "News mention"))
            statuses.append(status)

    non_empty_frames = [frame for frame in frames if not frame.empty]
    combined = pd.concat(non_empty_frames, ignore_index=True) if non_empty_frames else empty_competitor_frame()
    if not combined.empty:
        combined["published_at"] = pd.to_datetime(combined["published_at"], errors="coerce", utc=True)
        combined["cta"] = combined.apply(lambda row: detect_cta(f"{row['title']} {row['text']}"), axis=1)
        combined["theme"] = combined.apply(lambda row: detect_theme(f"{row['title']} {row['text']}"), axis=1)
        combined["sentiment"] = combined.apply(lambda row: sentiment_score(f"{row['title']} {row['text']}"), axis=1)
    return combined, pd.DataFrame(statuses)


def fetch_meta_ad_library(
    search_term: str,
    competitor: str,
    access_token: str | None,
    country: str = "US",
    api_version: str = "v21.0",
    max_records: int = 20,
    request_get: Callable[..., object] = requests.get,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if not access_token:
        url = f"https://www.facebook.com/ads/library/?active_status=active&ad_type=all&country={country}&q={quote_plus(search_term)}&search_type=keyword_unordered"
        frame = pd.DataFrame(
            [
                {
                    "source": "Meta Ad Library",
                    "competitor": competitor,
                    "keyword": search_term,
                    "asset_type": LIVE_LINK_ASSET_TYPE,
                    "title": f"Open Meta Ad Library public search for {search_term}",
                    "text": "Set META_ACCESS_TOKEN for API results; this link opens the public Ad Library search.",
                    "url": url,
                    "published_at": pd.Timestamp.now(tz="UTC"),
                    "author": "Meta Ad Library",
                    "platforms": "Meta",
                    "engagement": 0.0,
                }
            ],
            columns=empty_competitor_frame().columns,
        )
        return frame, _status("Meta Ad Library", search_term, "not configured", f"Set META_ACCESS_TOKEN or open public search: {url}")

    endpoint = f"https://graph.facebook.com/{api_version}/ads_archive"
    params = {
        "access_token": access_token,
        "ad_reached_countries": f"['{country}']",
        "ad_type": "ALL",
        "search_terms": search_term,
        "limit": min(max_records, 50),
        "fields": "id,page_name,ad_snapshot_url,ad_delivery_start_time,ad_delivery_stop_time,publisher_platforms,ad_creative_bodies,ad_creative_link_titles",
    }
    try:
        response = request_get(endpoint, params=params, timeout=15)
        status_code = getattr(response, "status_code", 200)
        if status_code == 429:
            return empty_competitor_frame(), _status("Meta Ad Library", search_term, "rate limited", "Meta API returned 429")
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:  # pragma: no cover
        return empty_competitor_frame(), _status("Meta Ad Library", search_term, "failed", str(exc))

    rows = []
    for ad in payload.get("data", [])[:max_records]:
        bodies = ad.get("ad_creative_bodies") or []
        titles = ad.get("ad_creative_link_titles") or []
        rows.append(
            {
                "source": "Meta Ad Library",
                "competitor": competitor,
                "keyword": search_term,
                "asset_type": "Ad",
                "title": _first(titles) or ad.get("page_name", "Meta ad"),
                "text": _first(bodies),
                "url": ad.get("ad_snapshot_url", ""),
                "published_at": ad.get("ad_delivery_start_time") or pd.Timestamp.now(tz="UTC"),
                "author": ad.get("page_name", competitor),
                "platforms": ", ".join(ad.get("publisher_platforms") or []),
                "engagement": 0.0,
            }
        )
    return pd.DataFrame(rows, columns=empty_competitor_frame().columns), _status("Meta Ad Library", search_term, "ok", f"{len(rows)} ads")


def fetch_tiktok_creative_center_link(search_term: str, competitor: str) -> tuple[pd.DataFrame, dict[str, str]]:
    url = f"https://ads.tiktok.com/business/creativecenter/inspiration/topads/pc/en?period=7&keyword={quote_plus(search_term)}"
    frame = pd.DataFrame(
        [
            {
                "source": "TikTok Creative Center",
                "competitor": competitor,
                "keyword": search_term,
                "asset_type": LIVE_LINK_ASSET_TYPE,
                "title": f"Open TikTok Creative Center for {search_term}",
                "text": "TikTok Creative Center does not expose a stable public API; this link opens live creative results.",
                "url": url,
                "published_at": pd.Timestamp.now(tz="UTC"),
                "author": "TikTok Creative Center",
                "platforms": "TikTok",
                "engagement": 0.0,
            }
        ],
        columns=empty_competitor_frame().columns,
    )
    return frame, _status("TikTok Creative Center", search_term, "live link", url)


def exclude_live_link_rows(items: pd.DataFrame) -> pd.DataFrame:
    """Drop synthetic live-search-link placeholder rows so analytics only count real items."""
    if items.empty or "asset_type" not in items.columns:
        return items
    return items[items["asset_type"] != LIVE_LINK_ASSET_TYPE]


def compute_share_of_voice(items: pd.DataFrame) -> pd.DataFrame:
    items = exclude_live_link_rows(items)
    if items.empty:
        return pd.DataFrame(columns=["competitor", "source", "items", "share_of_voice"])
    counts = items.groupby(["competitor", "source"], as_index=False).agg(items=("title", "count"))
    totals = counts.groupby("source")["items"].transform("sum")
    counts["share_of_voice"] = counts["items"] / totals.replace(0, pd.NA)
    return counts.sort_values(["source", "share_of_voice"], ascending=[True, False])


def analyze_creative_patterns(items: pd.DataFrame) -> pd.DataFrame:
    items = exclude_live_link_rows(items)
    if items.empty:
        return pd.DataFrame(columns=["competitor", "theme", "cta", "items", "avg_sentiment"])
    return (
        items.groupby(["competitor", "theme", "cta"], as_index=False)
        .agg(items=("title", "count"), avg_sentiment=("sentiment", "mean"))
        .sort_values(["items", "avg_sentiment"], ascending=[False, False])
    )


def detect_cta(text: str) -> str:
    lowered = str(text).lower()
    for label, regex in _CTA_REGEXES.items():
        if regex.search(lowered):
            return label
    return "No explicit CTA"


def detect_theme(text: str) -> str:
    lowered = str(text).lower()
    for label, regex in _THEME_REGEXES.items():
        if regex.search(lowered):
            return label
    return "General"


def empty_competitor_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["source", "competitor", "keyword", "asset_type", "title", "text", "url", "published_at", "author", "platforms", "engagement"])


def _search_terms(query: CompetitorQuery) -> list[tuple[str, str]]:
    terms: list[tuple[str, str]] = []
    keywords = query.keywords or ("",)
    for competitor in query.competitors:
        for keyword in keywords:
            search_term = f"{competitor} {keyword}".strip()
            terms.append((competitor, search_term))
    return terms


def _trend_to_competitor_frame(frame: pd.DataFrame, competitor: str, asset_type: str) -> pd.DataFrame:
    if frame.empty:
        return empty_competitor_frame()
    return pd.DataFrame(
        {
            "source": frame["source"],
            "competitor": competitor,
            "keyword": frame["keyword"],
            "asset_type": asset_type,
            "title": frame["title"],
            "text": frame["snippet"],
            "url": frame["url"],
            "published_at": frame["published_at"],
            "author": frame["author"],
            "platforms": frame["source"],
            "engagement": frame["engagement"],
        },
        columns=empty_competitor_frame().columns,
    )


def _status(source: str, keyword: str, status: str, detail: str) -> dict[str, str]:
    return {"source": source, "keyword": keyword, "status": status, "detail": detail}


def _first(values: object) -> str:
    if isinstance(values, list) and values:
        return str(values[0])
    if isinstance(values, str):
        return values
    return ""
