from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


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
                "marginal_roas",
                "risk_index",
                "saturation_index",
                "aov",
                "rows",
            ]
        )

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
    grouped["aov"] = grouped["historical_revenue"] / grouped["historical_conversions"].replace(0, np.nan)
    grouped["risk_index"] = (grouped["roas_std"].fillna(0) / grouped["historical_roas"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0)
    grouped["risk_index"] = grouped["risk_index"].clip(0, 2)
    grouped = grouped.fillna({"historical_roas": 0, "historical_cpa": 0, "aov": 100, "marginal_roas": 0, "saturation_index": 0.5})
    return grouped.drop(columns=["roas_std"])


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

    risk_penalty = (1 - config.risk_tolerance) * summary["risk_index"] * 0.22
    saturation_bonus = summary["saturation_index"].clip(0.15, 1.05)
    score = summary["marginal_roas"].clip(lower=0.05) * saturation_bonus * (1 - risk_penalty).clip(0.35, 1.1)
    score = score.replace([np.inf, -np.inf], np.nan).fillna(0)
    if float(score.sum()) <= 0:
        allocation_share = pd.Series(1 / len(summary), index=summary.index)
    else:
        allocation_share = score / score.sum()

    result = summary.copy()
    result["allocation_share"] = allocation_share
    result["allocation"] = result["allocation_share"] * config.total_budget

    historical_spend = result["historical_spend"].replace(0, np.nan)
    relative_budget = (result["allocation"] / historical_spend).replace([np.inf, -np.inf], np.nan).fillna(0.05)
    decay = 1 / (1 + np.power(relative_budget * 8.0, 0.62) * (1 - result["saturation_index"].clip(0.15, 0.98)))
    response_roas = result["marginal_roas"].clip(lower=0.05) * decay.clip(0.20, 1.05)

    result["expected_revenue"] = result["allocation"] * response_roas
    result["expected_profit"] = result["expected_revenue"] - result["allocation"]
    result["expected_roas"] = result["expected_revenue"] / result["allocation"].replace(0, np.nan)
    result["expected_conversions"] = result["expected_revenue"] / result["aov"].replace(0, np.nan)
    result["expected_cpa"] = result["allocation"] / result["expected_conversions"].replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], np.nan).fillna(0)
    result["action"] = result.apply(lambda row: _action(row, config.target_roas, config.target_cpa), axis=1)
    return result.sort_values("allocation", ascending=False).reset_index(drop=True)


def summarize_allocation(allocation: pd.DataFrame) -> dict[str, float]:
    budget = float(allocation["allocation"].sum()) if not allocation.empty else 0.0
    revenue = float(allocation["expected_revenue"].sum()) if not allocation.empty else 0.0
    conversions = float(allocation["expected_conversions"].sum()) if not allocation.empty else 0.0
    return {
        "budget": budget,
        "expected_revenue": revenue,
        "expected_profit": revenue - budget,
        "expected_roas": revenue / budget if budget else 0.0,
        "expected_conversions": conversions,
        "expected_cpa": budget / conversions if conversions else 0.0,
    }


def _action(row: pd.Series, target_roas: float, target_cpa: float) -> str:
    roas_ok = row["expected_roas"] >= target_roas
    cpa_ok = row["expected_cpa"] <= target_cpa if target_cpa > 0 else True
    if roas_ok and cpa_ok:
        return "Increase"
    if row["expected_roas"] >= target_roas * 0.75 or cpa_ok:
        return "Maintain"
    return "Limit"
