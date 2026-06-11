from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

# Penalty strength applied to volatile channels when risk tolerance is low.
RISK_PENALTY_STRENGTH = 0.35
# Multiples of the anchor spend (scaled by the diminishing-return index) at
# which a channel's marginal ROAS has decayed by 1/e.
SATURATION_DEPTH_FACTOR = 3.0
ALLOCATION_STEPS = 200
# Planning horizon expressed in weeks (a monthly budget by default).
PLANNING_WEEKS = 4.33
# Caps the per-period reference spend so market-scale demo panels behave like a
# single advertiser account; real uploads below the cap use their actual scale.
REFERENCE_SPEND_CAP = 750_000.0


@dataclass(frozen=True)
class OptimizerConfig:
    total_budget: float
    target_roas: float = 2.0
    target_cpa: float = 80.0
    risk_tolerance: float = 0.5
    objective: str | None = None
    excluded_platforms: tuple[str, ...] = ()


def prepare_channel_summary(
    frame: pd.DataFrame,
    objective: str | None = None,
    excluded_platforms: Iterable[str] = (),
) -> pd.DataFrame:
    df = frame.copy()
    if objective and objective != "All":
        df = df[df["objective"] == objective]
    excluded = set(excluded_platforms)
    if excluded:
        df = df[~df["platform"].isin(excluded)]
    if df.empty:
        return pd.DataFrame(
            columns=[
                "platform",
                "historical_spend",
                "historical_revenue",
                "historical_conversions",
                "historical_roas",
                "historical_cpa",
                "baseline_spend",
                "marginal_roas",
                "risk_index",
                "saturation_index",
                "aov",
                "rows",
            ]
        )

    # Normalize history to a per-period baseline so results do not depend on how
    # many weeks of data were provided (a one-week upload vs two years of history).
    n_weeks = max(pd.to_datetime(df["date"]).dt.to_period("W").nunique(), 1)

    grouped = df.groupby("platform").agg(
        historical_spend=("spend", "sum"),
        historical_revenue=("revenue", "sum"),
        historical_conversions=("conversions", "sum"),
        marginal_roas=("marginal_roas", "mean"),
        saturation_index=("diminishing_return_index", "mean"),
        roas_std=("roas", "std"),
        rows=("campaign_id", "count"),
    ).reset_index()
    grouped["historical_roas"] = grouped["historical_revenue"] / grouped["historical_spend"].replace(0, np.nan)
    grouped["historical_cpa"] = grouped["historical_spend"] / grouped["historical_conversions"].replace(0, np.nan)
    grouped["baseline_spend"] = grouped["historical_spend"] / n_weeks * PLANNING_WEEKS
    grouped["aov"] = grouped["historical_revenue"] / grouped["historical_conversions"].replace(0, np.nan)
    grouped["risk_index"] = (grouped["roas_std"].fillna(0) / grouped["historical_roas"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    grouped["risk_index"] = grouped["risk_index"].clip(0, 2)
    grouped = grouped.fillna({"historical_roas": 0, "historical_cpa": 0, "aov": 100, "marginal_roas": 0, "saturation_index": 0.5})
    return grouped.drop(columns=["roas_std"])


def _response_curve_params(summary: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-platform exponential-saturation response curves.

    Revenue at spend s follows R(s) = roas0 * S * (1 - exp(-s/S)), so the
    marginal ROAS is R'(s) = roas0 * exp(-s/S). Each channel is anchored at its
    historical share of a per-period reference spend (capped so market-scale
    demo data behaves like one advertiser account); at the anchor, the curve's
    marginal ROAS equals the observed marginal ROAS. The saturation scale S
    grows with the channel's diminishing-return index, so deeper channels hold
    their returns longer as budgets rise. Anchoring to shares of a per-period
    reference keeps results independent of how much history was uploaded.
    """
    saturation = summary["saturation_index"].clip(0.18, 1.05).to_numpy(dtype=float)
    baseline_total = max(float(summary["baseline_spend"].sum()), 1.0)
    reference_total = min(baseline_total, REFERENCE_SPEND_CAP)
    historical_share = (summary["historical_spend"] / max(float(summary["historical_spend"].sum()), 1.0)).to_numpy(dtype=float)
    anchor_spend = np.maximum(historical_share * reference_total, reference_total * 0.01)
    scale = anchor_spend * saturation * SATURATION_DEPTH_FACTOR
    roas0 = summary["marginal_roas"].clip(lower=0.05).to_numpy(dtype=float) * np.exp(anchor_spend / scale)
    return roas0, scale


def allocate_budget(frame: pd.DataFrame, config: OptimizerConfig) -> pd.DataFrame:
    if config.total_budget <= 0:
        raise ValueError("total_budget must be positive")
    if not 0 <= config.risk_tolerance <= 1:
        raise ValueError("risk_tolerance must be between 0 and 1")

    summary = prepare_channel_summary(frame, config.objective, config.excluded_platforms)
    if summary.empty:
        return summary.assign(
            allocation=[],
            allocation_share=[],
            expected_revenue=[],
            expected_profit=[],
            expected_roas=[],
            expected_conversions=[],
            expected_cpa=[],
            action=[],
        )

    roas0, scale = _response_curve_params(summary)
    aov = summary["aov"].clip(lower=1.0).to_numpy(dtype=float)
    risk_index = summary["risk_index"].to_numpy(dtype=float)
    risk_factor = 1 - (1 - config.risk_tolerance) * RISK_PENALTY_STRENGTH * risk_index
    risk_factor = np.clip(risk_factor, 0.05, 1.0)

    # Greedy water-filling on certainty-equivalent returns: each increment goes
    # to the channel with the highest risk-adjusted marginal ROAS, and targets
    # are checked against the risk-adjusted value, so low risk tolerance both
    # reorders the mix and funds volatile channels less deeply.
    step = config.total_budget / ALLOCATION_STEPS
    allocation = np.zeros(len(summary))
    constrained = np.zeros(len(summary), dtype=bool)
    spent = 0.0
    for _ in range(ALLOCATION_STEPS):
        adjusted_marginal = roas0 * np.exp(-allocation / scale) * risk_factor
        marginal_cpa = aov / np.maximum(adjusted_marginal, 1e-9)
        eligible = adjusted_marginal >= config.target_roas
        if config.target_cpa > 0:
            eligible &= marginal_cpa <= config.target_cpa
        constrained = constrained | (~eligible)
        if not eligible.any():
            break
        best = int(np.argmax(np.where(eligible, adjusted_marginal, -np.inf)))
        allocation[best] += step
        spent += step

    result = summary.copy()
    result["allocation"] = allocation
    result["allocation_share"] = allocation / config.total_budget
    result["expected_revenue"] = roas0 * scale * (1 - np.exp(-allocation / scale))
    result["expected_profit"] = result["expected_revenue"] - result["allocation"]
    result["expected_roas"] = result["expected_revenue"] / result["allocation"].replace(0, np.nan)
    result["expected_conversions"] = result["expected_revenue"] / aov
    result["expected_cpa"] = result["allocation"] / result["expected_conversions"].replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    result["unallocated"] = config.total_budget - spent

    historical_share = summary["historical_spend"] / max(float(summary["historical_spend"].sum()), 1.0)
    result["action"] = _actions(result["allocation_share"].to_numpy(), historical_share.to_numpy(), constrained)
    return result.sort_values("allocation", ascending=False).reset_index(drop=True)


def _actions(allocation_share: np.ndarray, historical_share: np.ndarray, constrained: np.ndarray) -> list[str]:
    actions = []
    for share, hist, limited in zip(allocation_share, historical_share, constrained):
        if limited and (share == 0 or share < hist * 0.85):
            actions.append("Limit (target)")
        elif share > hist * 1.15:
            actions.append("Increase")
        elif share < hist * 0.85:
            actions.append("Decrease")
        else:
            actions.append("Maintain")
    return actions


def summarize_allocation(allocation: pd.DataFrame) -> dict[str, float]:
    budget = float(allocation["allocation"].sum()) if not allocation.empty else 0.0
    revenue = float(allocation["expected_revenue"].sum()) if not allocation.empty else 0.0
    conversions = float(allocation["expected_conversions"].sum()) if not allocation.empty else 0.0
    unallocated = float(allocation["unallocated"].iloc[0]) if "unallocated" in allocation.columns and not allocation.empty else 0.0
    return {
        "budget": budget,
        "expected_revenue": revenue,
        "expected_profit": revenue - budget,
        "expected_roas": revenue / budget if budget else 0.0,
        "expected_conversions": conversions,
        "expected_cpa": budget / conversions if conversions else 0.0,
        "unallocated": unallocated,
    }
