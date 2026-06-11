from __future__ import annotations

import numpy as np
import pandas as pd

from data_sources.demand_metrics import (
    baseline_verdict,
    build_action_queue,
    classify_lifecycle,
    compute_baseline_stats,
    compute_momentum,
    compute_signal_confidence,
    confidence_verdict,
    lifecycle_verdict,
)


def _timeline(keyword: str, volumes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=len(volumes), freq="D")
    return pd.DataFrame({"date": dates, "keyword": keyword, "volume": volumes, "norm": float(sum(volumes))})


def test_compute_momentum_detects_rising_trend() -> None:
    rising = _timeline("ai marketing", [1, 1, 2, 2, 5, 6, 8, 10])
    momentum = compute_momentum(rising)

    assert len(momentum) == 1
    row = momentum.iloc[0]
    assert row["momentum"] > 0
    assert row["lifecycle"] in {"Accelerating", "Emerging", "Peaking"}


def test_compute_momentum_handles_empty() -> None:
    assert compute_momentum(pd.DataFrame(columns=["date", "keyword", "volume", "norm"])).empty


def test_classify_lifecycle_rules() -> None:
    assert classify_lifecycle(pd.Series({"momentum": 0.5, "acceleration": 0.2, "recent_avg": 10})) == "Accelerating"
    assert classify_lifecycle(pd.Series({"momentum": 0.5, "acceleration": -0.3, "recent_avg": 10})) == "Peaking"
    assert classify_lifecycle(pd.Series({"momentum": -0.4, "acceleration": -0.1, "recent_avg": 10})) == "Cooling"
    assert classify_lifecycle(pd.Series({"momentum": 0.0, "acceleration": 0.0, "recent_avg": 0.2})) == "Dormant"


def test_compute_baseline_stats_flags_spike() -> None:
    # A flat baseline followed by a sharp recent spike should read as unusual.
    timeline = _timeline("retail media", [2, 2, 3, 2, 2, 3, 2, 2, 20, 22])
    baseline = compute_baseline_stats(timeline, lookback_days=2)

    row = baseline.iloc[0]
    assert row["robust_z"] > 3.5
    assert row["classification"] == "Above baseline (unusual)"


def test_compute_baseline_stats_normal_range() -> None:
    timeline = _timeline("retail media", [5, 6, 5, 4, 5, 6, 5, 5, 6, 5])
    baseline = compute_baseline_stats(timeline, lookback_days=2)
    assert baseline.iloc[0]["classification"] == "Within normal range"


def test_compute_signal_confidence_rewards_corroboration() -> None:
    items = pd.DataFrame(
        {
            "keyword": ["ai marketing"] * 4 + ["niche topic"],
            "source": ["GDELT", "Reddit", "YouTube", "GDELT", "Reddit"],
            "title": ["a", "b", "c", "d", "e"],
            "sentiment": [1.0, -1.0, 1.0, 0.0, 1.0],
        }
    )
    confidence = compute_signal_confidence(items)

    multi = confidence[confidence["keyword"] == "ai marketing"].iloc[0]
    thin = confidence[confidence["keyword"] == "niche topic"].iloc[0]
    assert multi["sources"] == 3
    assert multi["confidence"] > thin["confidence"]
    assert thin["low_sample"]


def test_build_action_queue_caps_and_explains() -> None:
    keywords = [f"kw{i}" for i in range(8)]
    momentum = pd.DataFrame(
        {
            "keyword": keywords,
            "momentum": np.linspace(0.5, -0.2, 8),
            "lifecycle": ["Accelerating"] * 4 + ["Cooling"] * 4,
        }
    )
    confidence = pd.DataFrame(
        {
            "keyword": keywords,
            "confidence": np.linspace(0.9, 0.2, 8),
            "mentions": [20] * 8,
            "sources": [3] * 8,
            "low_sample": [False] * 8,
        }
    )
    baseline = pd.DataFrame(
        {
            "keyword": keywords,
            "robust_z": [4.0] + [0.0] * 7,
            "vs_baseline_pct": [0.5] + [0.0] * 7,
            "classification": ["Above baseline (unusual)"] + ["Within normal range"] * 7,
        }
    )

    actions = build_action_queue(momentum, confidence, baseline, max_actions=5)
    assert len(actions) == 5
    assert actions.iloc[0]["keyword"] == "kw0"
    assert actions.iloc[0]["action"]
    assert "stage" in actions.iloc[0]["why"]


def test_build_action_queue_handles_empty() -> None:
    empty = pd.DataFrame(columns=["keyword", "confidence", "mentions", "sources", "low_sample"])
    assert build_action_queue(pd.DataFrame(), empty, pd.DataFrame()).empty


def test_lifecycle_verdict_answers_window_to_act() -> None:
    accelerating = pd.Series({"lifecycle": "Accelerating", "momentum": 0.4})
    cooling = pd.Series({"lifecycle": "Cooling", "momentum": -0.3})

    assert "window to act is open" in lifecycle_verdict(accelerating)
    assert "peak has likely passed" in lifecycle_verdict(cooling)


def test_confidence_verdict_flags_noise_and_echo() -> None:
    thin = pd.Series({"mentions": 3, "sources": 1, "source_list": "Reddit", "low_sample": True})
    echo = pd.Series({"mentions": 20, "sources": 1, "source_list": "GDELT", "low_sample": False})
    solid = pd.Series({"mentions": 30, "sources": 3, "source_list": "GDELT, Reddit, YouTube", "low_sample": False})

    assert "statistical noise" in confidence_verdict(thin)
    assert "single-source echo" in confidence_verdict(echo)
    assert confidence_verdict(solid).startswith("Yes")


def test_baseline_verdict_says_whether_to_move_budget() -> None:
    unusual = pd.Series({"robust_z": 4.2, "vs_baseline_pct": 1.4})
    normal = pd.Series({"robust_z": 0.5, "vs_baseline_pct": 0.05})

    assert "justifies a budget/bid move" in baseline_verdict(unusual)
    assert "no budget change warranted" in baseline_verdict(normal)
