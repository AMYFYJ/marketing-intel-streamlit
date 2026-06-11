from __future__ import annotations

import numpy as np
import pandas as pd

# Lifecycle stages, ordered from earliest to latest in a typical demand curve.
LIFECYCLE_STAGES = ("Dormant", "Emerging", "Accelerating", "Peaking", "Cooling")

# A keyword needs at least this many total mentions before we trust its sentiment / signal.
MIN_CONFIDENT_VOLUME = 12


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def compute_momentum(timeline: pd.DataFrame) -> pd.DataFrame:
    """Per-keyword momentum and acceleration from a daily volume time series.

    ``momentum`` is the percentage change of the most recent half-window's average daily volume
    versus the prior half-window. ``acceleration`` compares the latest quarter against the one
    before it, so a positive value means the trend is still speeding up.
    """
    columns = ["keyword", "recent_avg", "prior_avg", "momentum", "acceleration", "latest_volume", "days"]
    if timeline.empty:
        return _empty(columns)

    rows = []
    for keyword, group in timeline.sort_values("date").groupby("keyword"):
        volumes = group["volume"].to_numpy(dtype=float)
        days = len(volumes)
        if days == 0:
            continue
        half = max(days // 2, 1)
        recent_avg = float(volumes[-half:].mean())
        prior_avg = float(volumes[:-half].mean()) if days > half else 0.0
        momentum = _pct_change(recent_avg, prior_avg)

        quarter = max(days // 4, 1)
        recent_q = float(volumes[-quarter:].mean())
        prior_q = float(volumes[-2 * quarter : -quarter].mean()) if days >= 2 * quarter else 0.0
        acceleration = _pct_change(recent_q, prior_q)

        rows.append(
            {
                "keyword": keyword,
                "recent_avg": recent_avg,
                "prior_avg": prior_avg,
                "momentum": momentum,
                "acceleration": acceleration,
                "latest_volume": float(volumes[-1]),
                "days": days,
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    frame["lifecycle"] = frame.apply(classify_lifecycle, axis=1)
    return frame


def classify_lifecycle(row: pd.Series) -> str:
    """Map a momentum row to a lifecycle stage using slope and acceleration thresholds."""
    momentum = float(row.get("momentum", 0.0))
    acceleration = float(row.get("acceleration", 0.0))
    recent_avg = float(row.get("recent_avg", 0.0))

    if recent_avg < 1.0:
        return "Dormant"
    if momentum >= 0.15 and acceleration >= 0.0:
        return "Accelerating"
    if momentum >= 0.15 and acceleration < 0.0:
        # Still up overall but decelerating — the peak is near or just passed.
        return "Peaking"
    if momentum <= -0.15:
        return "Cooling"
    if momentum > 0.0:
        return "Emerging"
    return "Cooling" if acceleration < -0.15 else "Emerging"


def compute_baseline_stats(timeline: pd.DataFrame, lookback_days: int = 7) -> pd.DataFrame:
    """Compare the recent ``lookback_days`` against each keyword's trailing baseline.

    Uses a robust median / MAD z-score (the same idiom as
    ``campaign_data.detect_anomalies``) so a spike is judged against the keyword's own history
    rather than an absolute threshold — which is how you separate a real demand shift from
    ordinary seasonal noise.
    """
    columns = [
        "keyword",
        "current_avg",
        "prior_avg",
        "baseline_median",
        "baseline_mad",
        "robust_z",
        "vs_baseline_pct",
        "classification",
    ]
    if timeline.empty:
        return _empty(columns)

    rows = []
    for keyword, group in timeline.sort_values("date").groupby("keyword"):
        volumes = group["volume"].to_numpy(dtype=float)
        days = len(volumes)
        if days == 0:
            continue
        window = min(int(lookback_days), days)
        current = volumes[-window:]
        baseline = volumes[:-window] if days > window else volumes
        prior = volumes[-2 * window : -window] if days >= 2 * window else np.array([])

        current_avg = float(current.mean())
        prior_avg = float(prior.mean()) if prior.size else 0.0
        median = float(np.median(baseline))
        mad = float(np.median(np.abs(baseline - median)))
        if mad > 0:
            # 0.6745 scales MAD to be comparable to a standard deviation for normal data.
            robust_z = 0.6745 * (current_avg - median) / mad
        else:
            # MAD collapses to 0 when most baseline days share the same value; fall back to std
            # so a genuine spike above a flat-but-noisy baseline is still flagged.
            std = float(np.std(baseline))
            robust_z = (current_avg - median) / std if std > 0 else 0.0

        rows.append(
            {
                "keyword": keyword,
                "current_avg": current_avg,
                "prior_avg": prior_avg,
                "baseline_median": median,
                "baseline_mad": mad,
                "robust_z": robust_z,
                "vs_baseline_pct": _pct_change(current_avg, median),
                "classification": _baseline_class(robust_z),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def compute_signal_confidence(items: pd.DataFrame) -> pd.DataFrame:
    """Score how trustworthy each keyword's signal is from corroboration and sample size.

    Confidence rewards (a) multiple independent sources saying the same thing and (b) enough total
    mentions to rise above statistical noise. Sentiment is reported with its sample size and a
    positive/negative split so a thin, one-source signal is never read as a confident trend.
    """
    columns = [
        "keyword",
        "mentions",
        "sources",
        "source_list",
        "positive",
        "negative",
        "avg_sentiment",
        "confidence",
        "low_sample",
    ]
    if items.empty:
        return _empty(columns)

    df = items.copy()
    df["sentiment"] = pd.to_numeric(df.get("sentiment", 0.0), errors="coerce").fillna(0.0)
    rows = []
    for keyword, group in df.groupby("keyword"):
        mentions = int(len(group))
        sources = sorted(str(s) for s in group["source"].dropna().unique())
        n_sources = len(sources)
        positive = int((group["sentiment"] > 0).sum())
        negative = int((group["sentiment"] < 0).sum())

        # Corroboration: 1 source -> 0.34, 2 -> 0.67, 3+ -> 1.0.
        corroboration = min(n_sources / 3.0, 1.0)
        # Volume sufficiency saturates once a keyword clears the noise threshold.
        volume_sufficiency = min(mentions / MIN_CONFIDENT_VOLUME, 1.0)
        confidence = round(0.6 * corroboration + 0.4 * volume_sufficiency, 3)

        rows.append(
            {
                "keyword": keyword,
                "mentions": mentions,
                "sources": n_sources,
                "source_list": ", ".join(sources),
                "positive": positive,
                "negative": negative,
                "avg_sentiment": float(group["sentiment"].mean()),
                "confidence": confidence,
                "low_sample": mentions < MIN_CONFIDENT_VOLUME,
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("confidence", ascending=False).reset_index(drop=True)


def build_action_queue(
    momentum: pd.DataFrame,
    confidence: pd.DataFrame,
    baseline: pd.DataFrame,
    max_actions: int = 5,
) -> pd.DataFrame:
    """Rank keywords into a short, explained action list to avoid alert fatigue.

    Priority combines momentum, signal confidence, and how far the keyword sits above its own
    baseline. The list is hard-capped at ``max_actions`` so the queue stays decision-grade rather
    than another wall of undifferentiated alerts.
    """
    columns = ["keyword", "lifecycle", "confidence", "priority", "action", "why"]
    if confidence.empty:
        return _empty(columns)

    merged = confidence.merge(
        momentum[["keyword", "momentum", "lifecycle"]], on="keyword", how="left"
    ).merge(
        baseline[["keyword", "robust_z", "vs_baseline_pct", "classification"]], on="keyword", how="left"
    )
    merged["momentum"] = merged["momentum"].fillna(0.0)
    merged["lifecycle"] = merged["lifecycle"].fillna("Dormant")
    merged["robust_z"] = merged["robust_z"].fillna(0.0)
    merged["vs_baseline_pct"] = merged["vs_baseline_pct"].fillna(0.0)
    merged["classification"] = merged["classification"].fillna("Within normal range")

    # Momentum/anomaly factors are shifted to stay positive so confidence always matters.
    momentum_factor = (1 + merged["momentum"]).clip(lower=0.1)
    anomaly_factor = (1 + merged["robust_z"].clip(lower=0) / 3.0).clip(lower=1.0)
    merged["priority"] = (merged["confidence"] * momentum_factor * anomaly_factor).round(3)

    merged = merged.sort_values("priority", ascending=False).head(max_actions)
    merged["action"] = merged.apply(_recommend_action, axis=1)
    merged["why"] = merged.apply(_explain_action, axis=1)
    return merged[columns].reset_index(drop=True)


def _recommend_action(row: pd.Series) -> str:
    lifecycle = row["lifecycle"]
    confident = not bool(row.get("low_sample", False)) and float(row["confidence"]) >= 0.5
    above_baseline = float(row.get("robust_z", 0.0)) >= 3.5

    if lifecycle == "Accelerating" and confident:
        return "Increase budget"
    if lifecycle in ("Accelerating", "Emerging") and confident:
        return "Launch test creative on this angle"
    if lifecycle == "Peaking":
        return "Capture now, prepare to taper"
    if lifecycle == "Cooling" or float(row.get("vs_baseline_pct", 0.0)) < -0.15:
        return "Deprioritize"
    if above_baseline or not confident:
        return "Add to watchlist"
    return "Add to watchlist"


def _explain_action(row: pd.Series) -> str:
    parts = [
        f"{row['lifecycle']} stage",
        f"momentum {float(row.get('momentum', 0.0)) * 100:+.0f}%",
        f"{int(row['sources'])} source(s), {int(row['mentions'])} mentions",
        f"confidence {float(row['confidence']):.2f}",
    ]
    z = float(row.get("robust_z", 0.0))
    if abs(z) >= 1.0:
        parts.append(f"{row.get('classification', '')} (z={z:+.1f})")
    if bool(row.get("low_sample", False)):
        parts.append("low sample — verify before scaling")
    return "; ".join(p for p in parts if p)


def lifecycle_verdict(row: pd.Series) -> str:
    """One-sentence answer to: is demand still building, and is the window to act open?"""
    momentum = float(row.get("momentum", 0.0)) * 100
    stage = str(row.get("lifecycle", "Dormant"))
    if stage == "Accelerating":
        return f"Demand is still building ({momentum:+.0f}% and speeding up) — the window to act is open."
    if stage == "Emerging":
        return f"Interest is starting to build ({momentum:+.0f}%) — early-mover window before it gets crowded."
    if stage == "Peaking":
        return f"Still up ({momentum:+.0f}%) but decelerating — at or near the peak; capture demand now and prepare to taper."
    if stage == "Cooling":
        return f"Demand is fading ({momentum:+.0f}%) — the peak has likely passed and this wave is missed."
    return "No meaningful demand volume right now — nothing to act on."


def confidence_verdict(row: pd.Series) -> str:
    """One-sentence answer to: can I trust this signal, or is it noise / a single-source echo?"""
    mentions = int(row.get("mentions", 0))
    sources = int(row.get("sources", 0))
    source_list = str(row.get("source_list", ""))
    if bool(row.get("low_sample", False)):
        return (
            f"Not yet — only {mentions} mention(s), below the {MIN_CONFIDENT_VOLUME}-mention noise "
            "threshold; this could be statistical noise."
        )
    if sources <= 1:
        return f"Partly — a single-source echo (only {source_list}); corroborate elsewhere before acting."
    return f"Yes — corroborated by {sources} independent sources ({source_list}) across {mentions} mentions."


def baseline_verdict(row: pd.Series) -> str:
    """One-sentence answer to: is this spike real for this keyword's own history, and does it justify a move?"""
    z = float(row.get("robust_z", 0.0))
    pct = float(row.get("vs_baseline_pct", 0.0)) * 100
    if z >= 3.5:
        return (
            f"Genuinely unusual — demand is {pct:+.0f}% vs its own baseline (z={z:+.1f}); "
            "a real shift that justifies a budget/bid move."
        )
    if z <= -3.5:
        return f"Well below its own baseline ({pct:+.0f}%, z={z:+.1f}) — demand has dropped; consider pulling back."
    return (
        f"Within its normal range ({pct:+.0f}% vs baseline, z={z:+.1f}) — likely ordinary fluctuation; "
        "no budget change warranted."
    )


def _baseline_class(robust_z: float) -> str:
    if robust_z >= 3.5:
        return "Above baseline (unusual)"
    if robust_z <= -3.5:
        return "Below baseline"
    return "Within normal range"


def _pct_change(current: float, prior: float) -> float:
    if prior <= 0:
        return 1.0 if current > 0 else 0.0
    return (current - prior) / prior
