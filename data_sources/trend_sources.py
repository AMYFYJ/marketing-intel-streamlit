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

SOURCE_CONFIDENCE = {
    "ok": 1.0,
    "rate limited": 0.35,
    "not configured": 0.25,
    "failed": 0.15,
}

INTENT_TERMS = {
    "Pain": ("problem", "complaint", "risk", "struggle", "hard", "expensive", "delay", "issue", "frustrated", "weak"),
    "Question": ("how", "what", "why", "should", "guide", "tips", "tutorial", "?"),
    "Comparison": ("vs", "versus", "alternative", "compare", "best", "top", "better"),
    "Purchase research": ("pricing", "cost", "trial", "demo", "buy", "tool", "platform", "vendor", "solution"),
}

STOP_WORDS = {
    "about",
    "after",
    "again",
    "against",
    "with",
    "from",
    "that",
    "this",
    "there",
    "their",
    "they",
    "have",
    "into",
    "your",
    "will",
    "were",
    "what",
    "when",
    "where",
    "which",
    "marketing",
}


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


def enrich_demand_signals(items: pd.DataFrame, statuses: pd.DataFrame, now: pd.Timestamp | None = None) -> pd.DataFrame:
    expected = empty_trend_frame().columns.tolist() + [
        "sentiment",
        "recency_hours",
        "freshness_hours",
        "source_confidence",
        "source_confidence_label",
        "urgency_score",
        "noise_risk",
        "intent",
        "audience_language",
        "priority",
        "recommended_action",
        "campaign_hook",
    ]
    if items.empty:
        return pd.DataFrame(columns=expected)

    current_time = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    if current_time.tzinfo is None:
        current_time = current_time.tz_localize("UTC")

    df = items.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=True).fillna(current_time)
    if "sentiment" not in df:
        df["sentiment"] = df.apply(lambda row: sentiment_score(f"{row['title']} {row['snippet']}"), axis=1)
    df["freshness_hours"] = ((current_time - df["published_at"]).dt.total_seconds() / 3600).clip(lower=0).fillna(168)
    if "recency_hours" not in df:
        df["recency_hours"] = df["freshness_hours"]
    df["source_confidence"] = _source_confidence(df, statuses)
    df["source_confidence_label"] = df["source_confidence"].map(_confidence_label)
    df["intent"] = df.apply(lambda row: classify_signal_intent(f"{row['title']} {row['snippet']}"), axis=1)
    df["audience_language"] = df.apply(lambda row: _audience_language(f"{row['title']} {row['snippet']}"), axis=1)
    df["noise_risk"] = df.apply(_noise_risk, axis=1)
    df["keyword_velocity"] = _keyword_velocity(df)
    df["urgency_score"] = _urgency_score(df)
    df["priority"] = df.apply(_priority, axis=1)
    df["recommended_action"] = df.apply(_recommended_action, axis=1)
    df["campaign_hook"] = df.apply(_campaign_hook, axis=1)
    return df


def summarize_demand_brief(items: pd.DataFrame, statuses: pd.DataFrame) -> dict[str, object]:
    enriched = enrich_demand_signals(items, statuses) if "urgency_score" not in items.columns else items
    usable_statuses = statuses[statuses["status"].eq("ok")] if not statuses.empty else pd.DataFrame()
    gap_statuses = statuses[~statuses["status"].eq("ok")] if not statuses.empty else pd.DataFrame()
    if enriched.empty:
        return {
            "active_keywords": 0,
            "source_coverage": "0/0",
            "source_gaps": int(len(gap_statuses)),
            "rising_topic": "No data",
            "urgency_score": 0.0,
            "sentiment_shift": "0.00",
            "audience_language": "No signal yet",
            "next_move": "Broaden query",
            "test_now": 0,
            "noise_risk": 0,
        }

    source_total = int(statuses["source"].nunique()) if not statuses.empty else int(enriched["source"].nunique())
    source_ok = int(usable_statuses["source"].nunique()) if not usable_statuses.empty else int(enriched["source"].nunique())
    keyword_strength = (
        enriched.groupby("keyword", as_index=False)
        .agg(avg_urgency=("urgency_score", "mean"), signals=("title", "count"))
        .sort_values(["avg_urgency", "signals"], ascending=False)
    )
    top_signal = enriched.sort_values(["urgency_score", "freshness_hours"], ascending=[False, True]).iloc[0]
    action_order = {"Test now": 0, "Content idea": 1, "Monitor": 2, "Ignore/noisy": 3, "Fix source": 4}
    next_move = (
        enriched["recommended_action"]
        .value_counts()
        .rename_axis("action")
        .reset_index(name="count")
        .assign(order=lambda frame: frame["action"].map(action_order).fillna(5))
        .sort_values(["order", "count"], ascending=[True, False])
        .iloc[0]["action"]
    )
    return {
        "active_keywords": int(enriched["keyword"].nunique()),
        "source_coverage": f"{source_ok}/{source_total}",
        "source_gaps": int(len(gap_statuses)),
        "rising_topic": str(keyword_strength.iloc[0]["keyword"]) if not keyword_strength.empty else "No data",
        "urgency_score": float(enriched["urgency_score"].mean()),
        "sentiment_shift": f"{float(enriched['sentiment'].mean()):+.2f}",
        "audience_language": str(top_signal["audience_language"]),
        "next_move": str(next_move),
        "test_now": int(enriched["recommended_action"].eq("Test now").sum()),
        "noise_risk": int(enriched["noise_risk"].eq("High").sum()),
    }


def classify_signal_intent(text: str) -> str:
    lowered = str(text).lower()
    for label, terms in INTENT_TERMS.items():
        if any(term in lowered for term in terms):
            return label
    return "General mention"


def build_signal_opportunities(items: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    columns = ["priority", "recommended_action", "keyword", "intent", "campaign_hook", "rationale", "representative_signal", "url"]
    if items.empty:
        return pd.DataFrame(columns=columns)
    enriched = enrich_demand_signals(items, pd.DataFrame()) if "urgency_score" not in items.columns else items.copy()
    velocity_lookup = summary.groupby("keyword")["velocity"].sum().to_dict() if not summary.empty and "velocity" in summary else {}
    rows = []
    grouped = (
        enriched.groupby(["keyword", "intent", "recommended_action"], as_index=False)
        .agg(
            signals=("title", "count"),
            avg_urgency=("urgency_score", "mean"),
            avg_sentiment=("sentiment", "mean"),
            high_noise=("noise_risk", lambda values: int((values == "High").sum())),
        )
        .sort_values(["avg_urgency", "signals"], ascending=[False, False])
    )
    for _, row in grouped.iterrows():
        sample = enriched[
            (enriched["keyword"].eq(row["keyword"]))
            & (enriched["intent"].eq(row["intent"]))
            & (enriched["recommended_action"].eq(row["recommended_action"]))
        ].sort_values("urgency_score", ascending=False).iloc[0]
        rows.append(
            {
                "priority": sample["priority"],
                "recommended_action": row["recommended_action"],
                "keyword": row["keyword"],
                "intent": row["intent"],
                "campaign_hook": sample["campaign_hook"],
                "rationale": (
                    f"{int(row['signals'])} signals, urgency {row['avg_urgency']:.0f}, "
                    f"sentiment {row['avg_sentiment']:+.2f}, velocity {velocity_lookup.get(row['keyword'], 0):.1f}."
                ),
                "representative_signal": sample["title"],
                "url": sample["url"],
            }
        )
    priority_order = {"High": 0, "Medium": 1, "Low": 2, "Fix source": 3}
    action_order = {"Test now": 0, "Content idea": 1, "Monitor": 2, "Ignore/noisy": 3, "Fix source": 4}
    return (
        pd.DataFrame(rows, columns=columns)
        .assign(
            priority_rank=lambda frame: frame["priority"].map(priority_order).fillna(4),
            action_rank=lambda frame: frame["recommended_action"].map(action_order).fillna(5),
        )
        .sort_values(["priority_rank", "action_rank"])
        .drop(columns=["priority_rank", "action_rank"])
    )


def build_keyword_source_matrix(items: pd.DataFrame) -> pd.DataFrame:
    if items.empty:
        return pd.DataFrame()
    metric = "urgency_score" if "urgency_score" in items.columns else "title"
    if metric == "title":
        grid = items.groupby(["keyword", "source"], as_index=False).agg(value=("title", "count"))
    else:
        grid = items.groupby(["keyword", "source"], as_index=False).agg(value=("urgency_score", "mean"))
    return grid.pivot(index="keyword", columns="source", values="value").fillna(0)


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
        return "Direct"
    if value >= 0.5:
        return "Partial"
    if value >= 0.3:
        return "Limited"
    return "Needs setup"


def _audience_language(text: str) -> str:
    cleaned = " ".join(str(text).split())
    if not cleaned:
        return "No phrase available"
    sentences = [piece.strip() for piece in cleaned.replace("?", "?.").replace("!", "!.").split(".") if piece.strip()]
    questions = [sentence for sentence in sentences if "?" in sentence]
    phrase = questions[0] if questions else sentences[0] if sentences else cleaned
    return phrase[:110].rstrip()


def _noise_risk(row: pd.Series) -> str:
    keyword_terms = [term for term in str(row.get("keyword", "")).lower().split() if term not in STOP_WORDS]
    text = f"{row.get('title', '')} {row.get('snippet', '')}".lower()
    keyword_hits = sum(1 for term in keyword_terms if term and term in text)
    broad_keyword = len(keyword_terms) <= 1
    general_intent = row.get("intent") == "General mention"
    neutral = abs(float(row.get("sentiment", 0))) < 0.15
    if broad_keyword and general_intent and neutral:
        return "High"
    if general_intent or keyword_hits == 0:
        return "Medium"
    return "Low"


def _keyword_velocity(items: pd.DataFrame) -> pd.Series:
    recency_weight = 1 / (1 + items["recency_hours"].fillna(168) / 24)
    velocity = recency_weight.groupby(items["keyword"]).transform("sum")
    return velocity.fillna(0)


def _urgency_score(items: pd.DataFrame) -> pd.Series:
    recency_score = 1 / (1 + items["freshness_hours"].fillna(168) / 48)
    engagement_score = _percentile(items["engagement"].fillna(0))
    velocity_score = _percentile(items["keyword_velocity"].fillna(0))
    intent_score = items["intent"].ne("General mention").astype(float)
    noise_penalty = items["noise_risk"].map({"High": 18, "Medium": 7, "Low": 0}).fillna(7)
    urgency = (
        recency_score * 30
        + items["source_confidence"] * 20
        + engagement_score * 15
        + velocity_score * 12
        + items["sentiment"].abs().clip(upper=1) * 10
        + intent_score * 13
        - noise_penalty
    )
    return urgency.clip(0, 100).round(1)


def _percentile(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True).fillna(0.5)


def _priority(row: pd.Series) -> str:
    if float(row.get("source_confidence", 0)) < 0.3:
        return "Fix source"
    if float(row.get("urgency_score", 0)) >= 72 and row.get("noise_risk") != "High":
        return "High"
    if float(row.get("urgency_score", 0)) >= 48:
        return "Medium"
    return "Low"


def _recommended_action(row: pd.Series) -> str:
    if float(row.get("source_confidence", 0)) < 0.3:
        return "Fix source"
    if row.get("noise_risk") == "High" and float(row.get("urgency_score", 0)) < 60:
        return "Ignore/noisy"
    if float(row.get("urgency_score", 0)) >= 72 and row.get("noise_risk") != "High":
        return "Test now"
    if row.get("intent") in {"Question", "Comparison", "Purchase research", "Pain"} and float(row.get("urgency_score", 0)) >= 45:
        return "Content idea"
    if float(row.get("urgency_score", 0)) >= 45:
        return "Monitor"
    return "Ignore/noisy" if row.get("noise_risk") == "High" else "Monitor"


def _campaign_hook(row: pd.Series) -> str:
    phrase = str(row.get("audience_language", "this signal"))
    intent = row.get("intent")
    if intent == "Pain":
        return f"Answer the objection: {phrase}"
    if intent == "Question":
        return f"Turn the question into how-to content: {phrase}"
    if intent == "Comparison":
        return f"Create comparison messaging around: {phrase}"
    if intent == "Purchase research":
        return f"Build a conversion offer around: {phrase}"
    if row.get("recommended_action") == "Ignore/noisy":
        return f"Hold until the signal gets more specific: {phrase}"
    return f"Use a timely trend hook: {phrase}"
