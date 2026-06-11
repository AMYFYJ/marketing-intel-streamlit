from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
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

CREATIVE_ANGLE_PATTERNS = {
    "Before / After": ("before and after", "before/after", "transformation", "results after"),
    "Routine": ("routine", "regimen", "daily", "morning", "nightly", "step-by-step"),
    "Problem / Solution": ("problem", "solution", "save time", "reduce costs", "struggle", "pain point"),
    "Ingredient Claim": ("ingredient", "formula", "clinically", "spf", "retinol", "hyaluronic", "ceramide"),
    "Expert Proof": ("expert", "dermatologist", "doctor", "recommended", "certified", "study"),
    "UGC / Testimonial": ("testimonial", "review", "creator", "influencer", "ugc", "real customer", "customers love"),
    "Comparison": (" vs ", "versus", "compare", "alternative", "better than", "switch from"),
    "New Arrival": ("new", "launch", "introducing", "new arrival", "just dropped"),
    "Price / Promo": ("discount", "sale", "offer", "limited time", "bundle", "free shipping", "gift with purchase"),
    "Demo / How-To": ("demo", "how to", "tutorial", "walkthrough", "learn how", "guide"),
}

SOURCE_CONFIDENCE = {
    "ok": 1.0,
    "live link": 0.62,
    "rate limited": 0.35,
    "not configured": 0.25,
    "failed": 0.15,
}
MARKET_WIDE_COMPETITOR = "Market-wide"


@dataclass(frozen=True)
class CompetitorQuery:
    competitors: tuple[str, ...]
    keywords: tuple[str, ...] = ()
    country: str = "US"
    max_items_per_source: int = 20


def parse_competitors(raw: str) -> tuple[str, ...]:
    return parse_keywords(raw)


def fetch_competitor_intelligence(
    query: CompetitorQuery,
    sources: tuple[str, ...] = ("Meta Ad Library", "TikTok Creative Center", "LinkedIn Ad Library", "Reddit", "GDELT"),
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
        if "LinkedIn Ad Library" in sources:
            frame, status = fetch_linkedin_ad_library_link(term, competitor)
            frames.append(frame)
            statuses.append(status)
        if "X Ads Repository (EU Only)" in sources:
            frame, status = fetch_x_ads_repository_link(term, competitor)
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
        return empty_competitor_frame(), _status("Meta Ad Library", search_term, "not configured", f"Set META_ACCESS_TOKEN or open public search: {url}")

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
                "asset_type": "Live search link",
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


def fetch_linkedin_ad_library_link(search_term: str, competitor: str) -> tuple[pd.DataFrame, dict[str, str]]:
    url = f"https://www.linkedin.com/ads/library/?keyword={quote_plus(search_term)}"
    frame = pd.DataFrame(
        [
            {
                "source": "LinkedIn Ad Library",
                "competitor": competitor,
                "keyword": search_term,
                "asset_type": "Live search link",
                "title": f"Open LinkedIn Ad Library for {search_term}",
                "text": (
                    "LinkedIn Ad Library is a public ad transparency database. "
                    "This link opens the library so you can search by advertiser, keyword, country, or date range."
                ),
                "url": url,
                "published_at": pd.Timestamp.now(tz="UTC"),
                "author": "LinkedIn Ad Library",
                "platforms": "LinkedIn",
                "engagement": 0.0,
            }
        ],
        columns=empty_competitor_frame().columns,
    )
    return frame, _status("LinkedIn Ad Library", search_term, "live link", url)


def fetch_x_ads_repository_link(search_term: str, competitor: str) -> tuple[pd.DataFrame, dict[str, str]]:
    url = "https://ads.twitter.com/ads-repository"
    frame = pd.DataFrame(
        [
            {
                "source": "X Ads Repository (EU Only)",
                "competitor": competitor,
                "keyword": search_term,
                "asset_type": "Live search link",
                "title": f"Open X Ads Repository for {search_term}",
                "text": (
                    "X Ads Repository is the EU Digital Services Act transparency repository. "
                    "Use it for EU-served ads by account, country, and date range; it is not a broad global competitor ad library."
                ),
                "url": url,
                "published_at": pd.Timestamp.now(tz="UTC"),
                "author": "X Ads Repository",
                "platforms": "X",
                "engagement": 0.0,
            }
        ],
        columns=empty_competitor_frame().columns,
    )
    return frame, _status("X Ads Repository (EU Only)", search_term, "live link", url)


def compute_share_of_voice(items: pd.DataFrame) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame(columns=["competitor", "source", "items", "share_of_voice"])
    counts = items.groupby(["competitor", "source"], as_index=False).agg(items=("title", "count"))
    totals = counts.groupby("source")["items"].transform("sum")
    counts["share_of_voice"] = counts["items"] / totals.replace(0, pd.NA)
    return counts.sort_values(["source", "share_of_voice"], ascending=[True, False])


def analyze_creative_patterns(items: pd.DataFrame) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame(columns=["competitor", "theme", "cta", "items", "avg_sentiment"])
    return (
        items.groupby(["competitor", "theme", "cta"], as_index=False)
        .agg(items=("title", "count"), avg_sentiment=("sentiment", "mean"))
        .sort_values(["items", "avg_sentiment"], ascending=[False, False])
    )


def enrich_competitor_items(items: pd.DataFrame, statuses: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    """Add proxy decision fields for creative monitoring.

    Public competitive sources show observed creative and mention signals, not paid-media
    performance. These fields intentionally score recency, source access, and copy patterns.
    """
    expected = empty_competitor_frame().columns.tolist() + [
        "cta",
        "theme",
        "sentiment",
        "freshness_days",
        "source_confidence",
        "source_confidence_label",
        "signal_strength",
        "creative_format",
        "campaign_type",
        "creative_angle",
        "priority",
        "recommended_action",
    ]
    if items.empty:
        return pd.DataFrame(columns=expected)

    current_time = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("UTC")

    df = items.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True).fillna(current_time)
    if "cta" not in df:
        df["cta"] = df.apply(lambda row: detect_cta(f"{row['title']} {row['text']}"), axis=1)
    if "theme" not in df:
        df["theme"] = df.apply(lambda row: detect_theme(f"{row['title']} {row['text']}"), axis=1)
    if "sentiment" not in df:
        df["sentiment"] = df.apply(lambda row: sentiment_score(f"{row['title']} {row['text']}"), axis=1)

    df["freshness_days"] = ((current_time - df["published_at"]).dt.total_seconds() / 86_400).clip(lower=0).fillna(0)
    df["source_confidence"] = _source_confidence(df, statuses)
    df["source_confidence_label"] = df["source_confidence"].map(_confidence_label)
    df["signal_strength"] = _signal_strength(df)
    df["creative_format"] = df.apply(_creative_format, axis=1)
    df["campaign_type"] = df.apply(_campaign_type, axis=1)
    df["creative_angle"] = df.apply(_creative_angle, axis=1)
    df["priority"] = df.apply(_priority, axis=1)
    df["recommended_action"] = df.apply(_recommended_action, axis=1)
    return df


def summarize_competitive_signals(items: pd.DataFrame, statuses: pd.DataFrame) -> dict[str, object]:
    enriched = enrich_competitor_items(items, statuses) if "signal_strength" not in items.columns else items
    active_statuses = statuses[statuses["status"].isin(["ok", "live link"])] if not statuses.empty else pd.DataFrame()
    gap_statuses = statuses[~statuses["status"].isin(["ok", "live link"])] if not statuses.empty else pd.DataFrame()
    if enriched.empty:
        return {
            "items": 0,
            "active_sources": int(active_statuses["source"].nunique()) if not active_statuses.empty else 0,
            "source_gaps": int(len(gap_statuses)),
            "sov_leader": "No data",
            "top_theme": "No data",
            "top_cta": "No data",
            "newest_signal": "No data",
            "test_next": 0,
            "avg_signal_strength": 0.0,
        }

    sov = compute_share_of_voice(enriched)
    sov_leader = sov.sort_values("share_of_voice", ascending=False).iloc[0]["competitor"] if not sov.empty else "No data"
    top_theme = str(enriched["theme"].value_counts().idxmax()) if "theme" in enriched else "No data"
    top_cta = str(enriched["cta"].value_counts().idxmax()) if "cta" in enriched else "No data"
    newest = enriched.sort_values("published_at", ascending=False).iloc[0]
    return {
        "items": int(len(enriched)),
        "active_sources": int(active_statuses["source"].nunique()) if not active_statuses.empty else int(enriched["source"].nunique()),
        "source_gaps": int(len(gap_statuses)),
        "sov_leader": str(sov_leader),
        "top_theme": top_theme,
        "top_cta": top_cta,
        "newest_signal": f"{newest['competitor']} · {newest['source']}",
        "test_next": int(enriched["recommended_action"].eq("Test next").sum()),
        "avg_signal_strength": float(enriched["signal_strength"].mean()),
    }


def build_theme_cta_matrix(items: pd.DataFrame) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame()
    matrix = (
        items.groupby(["theme", "cta"], as_index=False)
        .agg(signals=("title", "count"), avg_strength=("signal_strength", "mean"))
        .sort_values(["signals", "avg_strength"], ascending=[False, False])
    )
    return matrix.pivot(index="theme", columns="cta", values="signals").fillna(0)


def build_strategy_recommendations(items: pd.DataFrame) -> pd.DataFrame:
    columns = ["priority", "recommended_action", "competitor", "creative_angle", "rationale", "representative_signal", "url"]
    if items.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    grouped = (
        items.groupby(["competitor", "theme", "cta", "recommended_action"], as_index=False)
        .agg(
            signals=("title", "count"),
            avg_strength=("signal_strength", "mean"),
            newest=("published_at", "max"),
            avg_sentiment=("sentiment", "mean"),
        )
        .sort_values(["recommended_action", "avg_strength", "signals"], ascending=[False, False, False])
    )
    for _, row in grouped.iterrows():
        sample = items[
            (items["competitor"].eq(row["competitor"]))
            & (items["theme"].eq(row["theme"]))
            & (items["cta"].eq(row["cta"]))
            & (items["recommended_action"].eq(row["recommended_action"]))
        ].sort_values("signal_strength", ascending=False).iloc[0]
        rows.append(
            {
                "priority": _priority(sample),
                "recommended_action": row["recommended_action"],
                "competitor": row["competitor"],
                "creative_angle": _creative_angle(sample),
                "rationale": (
                    f"{int(row['signals'])} signals, average strength {row['avg_strength']:.0f}, "
                    f"sentiment {row['avg_sentiment']:.2f}."
                ),
                "representative_signal": sample["title"],
                "url": sample["url"],
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["priority", "recommended_action"],
        key=lambda series: series.map({"High": 0, "Medium": 1, "Low": 2, "Fix source": 3}).fillna(4),
    )


def detect_cta(text: str) -> str:
    lowered = str(text).lower()
    for label, patterns in CTA_PATTERNS.items():
        if any(pattern in lowered for pattern in patterns):
            return label
    return "No explicit CTA"


def detect_theme(text: str) -> str:
    lowered = str(text).lower()
    for label, patterns in THEME_TERMS.items():
        if any(pattern in lowered for pattern in patterns):
            return label
    return "General"


def empty_competitor_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["source", "competitor", "keyword", "asset_type", "title", "text", "url", "published_at", "author", "platforms", "engagement"])


def _search_terms(query: CompetitorQuery) -> list[tuple[str, str]]:
    terms: list[tuple[str, str]] = []
    keywords = query.keywords or ("",)
    if not query.competitors:
        return [(MARKET_WIDE_COMPETITOR, keyword) for keyword in keywords if keyword]
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


def _source_confidence(items: pd.DataFrame, statuses: pd.DataFrame) -> pd.Series:
    if statuses.empty:
        return pd.Series(0.5, index=items.index)
    lookup = statuses.copy()
    lookup["source"] = lookup["source"].astype(str)
    lookup["keyword"] = lookup["keyword"].astype(str)
    lookup["confidence"] = lookup["status"].map(SOURCE_CONFIDENCE).fillna(0.4)
    exact = lookup.drop_duplicates(["source", "keyword"]).set_index(["source", "keyword"])["confidence"].to_dict()
    by_source = lookup.groupby("source")["confidence"].max().to_dict()
    values = []
    for _, row in items.iterrows():
        key = (str(row["source"]), str(row["keyword"]))
        values.append(float(exact.get(key, by_source.get(str(row["source"]), 0.5))))
    return pd.Series(values, index=items.index)


def _confidence_label(value: float) -> str:
    if value >= 0.8:
        return "Direct source"
    if value >= 0.55:
        return "Live reference"
    if value >= 0.3:
        return "Partial"
    return "Needs setup"


def _signal_strength(items: pd.DataFrame) -> pd.Series:
    recency_score = 1 / (1 + items["freshness_days"] / 14)
    cta_score = items["cta"].ne("No explicit CTA").astype(float)
    theme_score = items["theme"].ne("General").astype(float)
    sentiment_score_abs = items["sentiment"].abs().clip(upper=1)
    engagement_score = _percentile(items["engagement"].fillna(0))
    strength = (
        items["source_confidence"] * 30
        + recency_score * 25
        + cta_score * 15
        + theme_score * 15
        + sentiment_score_abs * 5
        + engagement_score * 10
    )
    return strength.clip(0, 100).round(1)


def _percentile(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True).fillna(0.5)


def _creative_format(row: pd.Series) -> str:
    source = str(row.get("source", "")).lower()
    asset_type = str(row.get("asset_type", "")).lower()
    platforms = str(row.get("platforms", "")).lower()
    text = f"{row.get('title', '')} {row.get('text', '')}".lower()
    if "youtube" in source or "youtube" in platforms:
        return "Video"
    if "tiktok" in source or "tiktok" in platforms:
        return "Video"
    if "linkedin" in source or "linkedin" in platforms:
        return "Ad Library Link"
    if "x ads repository" in source or platforms == "x":
        return "Ad Repository Link"
    if "reddit" in source:
        return "Social Post"
    if "gdelt" in source:
        return "News Article"
    if "carousel" in text:
        return "Carousel"
    if "video" in text or "watch" in text:
        return "Video"
    if any(term in text for term in ("static", "image", "photo", "graphic")):
        return "Static Image"
    if "ad" in asset_type:
        return "Ad Format Unknown"
    if "live search link" in asset_type:
        return "Live Search Link"
    return str(row.get("asset_type", "") or "Unknown")


def _campaign_type(row: pd.Series) -> str:
    source = str(row.get("source", "")).lower()
    cta = str(row.get("cta", "No explicit CTA"))
    theme = str(row.get("theme", "General"))
    text = f"{row.get('title', '')} {row.get('text', '')}".lower()
    if any(term in text for term in ("holiday", "seasonal", "back to school", "black friday", "cyber monday")):
        return "Seasonal"
    if any(term in text for term in ("creator", "influencer", "ugc", "testimonial", "review")):
        return "Influencer / Creator"
    if cta == "Shop now" or theme == "Discount" or any(term in text for term in ("discount", "sale", "bundle", "free shipping", "limited time")):
        return "Discount / Promo"
    if theme == "Launch" or any(term in text for term in ("new arrival", "just dropped", "introducing")):
        return "Product Launch"
    if cta in {"Book demo", "Free trial", "Sign up"}:
        return "Lead Gen"
    if cta in {"Download", "Learn more"} or theme == "Education":
        return "Educational"
    if theme == "Trust" or any(term in text for term in ("proof", "trusted", "secure", "case study")):
        return "Social Proof / Trust"
    if theme in {"AI", "Efficiency"}:
        return "Product Feature"
    if "reddit" in source or "gdelt" in source:
        return "Organic / PR"
    return "Brand Awareness"


def _creative_angle(row: pd.Series) -> str:
    text = f"{row.get('title', '')} {row.get('text', '')}".lower()
    for label, patterns in CREATIVE_ANGLE_PATTERNS.items():
        if any(pattern in text for pattern in patterns):
            return label
    theme = str(row.get("theme", "General"))
    cta = str(row.get("cta", "No explicit CTA"))
    if cta == "No explicit CTA":
        return theme
    if theme == "General":
        return cta
    return f"{theme} + {cta}"


def _priority(row: pd.Series) -> str:
    if float(row.get("source_confidence", 0)) < 0.3:
        return "Fix source"
    if float(row.get("signal_strength", 0)) >= 72:
        return "High"
    if float(row.get("signal_strength", 0)) >= 48:
        return "Medium"
    return "Low"


def _recommended_action(row: pd.Series) -> str:
    if float(row.get("source_confidence", 0)) < 0.3:
        return "Fix source"
    if str(row.get("asset_type", "")).lower() == "live search link":
        return "Open source"
    if float(row.get("signal_strength", 0)) >= 72:
        return "Test next"
    if float(row.get("freshness_days", 99)) <= 21 or float(row.get("signal_strength", 0)) >= 48:
        return "Watch"
    return "Archive"
