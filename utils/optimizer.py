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
    business_goal: str = "Profitable growth"
    planning_horizon_weeks: int = 4


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


def summarize_goal_fit(allocation: pd.DataFrame, config: OptimizerConfig) -> dict[str, object]:
    summary = summarize_allocation(allocation)
    budget = summary["budget"]
    if allocation.empty or budget <= 0:
        return {
            "business_goal": config.business_goal,
            "status": "No allocation",
            "summary": "No eligible channels remain for this scenario.",
            "target_roas_gap": 0.0,
            "target_cpa_gap": 0.0,
            "budget_increase": 0.0,
            "budget_maintain": 0.0,
            "budget_limit": 0.0,
            "increase_share": 0.0,
            "limit_share": 0.0,
            "top_platform": "None",
            "top_platform_share": 0.0,
            "primary_constraint": "No eligible channels remain after filters.",
            "planning_horizon_weeks": config.planning_horizon_weeks,
        }

    by_action = allocation.groupby("action")["allocation"].sum()
    budget_increase = float(by_action.get("Increase", 0.0))
    budget_maintain = float(by_action.get("Maintain", 0.0))
    budget_limit = float(by_action.get("Limit", 0.0))
    increase_share = budget_increase / budget
    limit_share = budget_limit / budget

    top_row = allocation.sort_values("allocation", ascending=False).iloc[0]
    status = _goal_status(summary, config, increase_share, limit_share)
    primary_constraint = _primary_constraint(summary, config, limit_share)
    target_roas_gap = summary["expected_roas"] - config.target_roas
    target_cpa_gap = config.target_cpa - summary["expected_cpa"]
    horizon = max(int(config.planning_horizon_weeks), 1)

    return {
        "business_goal": config.business_goal,
        "status": status,
        "summary": (
            f"{status}: {config.business_goal} plan puts {float(top_row['allocation_share']):.0%} "
            f"of budget into {top_row['platform']} and yields "
            f"{summary['expected_roas']:.2f}x ROAS / ${summary['expected_cpa']:,.0f} CPA "
            f"for the {horizon}-week scenario."
        ),
        "target_roas_gap": target_roas_gap,
        "target_cpa_gap": target_cpa_gap,
        "budget_increase": budget_increase,
        "budget_maintain": budget_maintain,
        "budget_limit": budget_limit,
        "increase_share": increase_share,
        "limit_share": limit_share,
        "top_platform": str(top_row["platform"]),
        "top_platform_share": float(top_row["allocation_share"]),
        "primary_constraint": primary_constraint,
        "planning_horizon_weeks": horizon,
    }


def build_allocation_action_plan(allocation: pd.DataFrame, config: OptimizerConfig) -> pd.DataFrame:
    columns = [
        "priority",
        "platform",
        "decision",
        "action",
        "allocation",
        "allocation_share",
        "historical_spend_share",
        "recommended_shift",
        "shift_direction",
        "expected_revenue",
        "expected_roas",
        "expected_cpa",
        "expected_profit",
        "marginal_roas",
        "risk_index",
        "saturation_index",
        "business_reason",
        "next_step",
    ]
    if allocation.empty:
        return pd.DataFrame(columns=columns)

    plan = allocation.copy()
    total_historical_spend = float(plan["historical_spend"].sum())
    if total_historical_spend > 0:
        plan["historical_spend_share"] = plan["historical_spend"] / total_historical_spend
    else:
        plan["historical_spend_share"] = 1 / len(plan)
    plan["recommended_shift"] = plan["allocation_share"] - plan["historical_spend_share"]
    plan["shift_direction"] = np.select(
        [
            plan["recommended_shift"] >= 0.04,
            plan["recommended_shift"] <= -0.04,
        ],
        ["Shift budget in", "Pull budget back"],
        default="Hold near current mix",
    )
    plan["decision"] = plan["action"].map(
        {
            "Increase": "Fund incrementally",
            "Maintain": "Protect and optimize",
            "Limit": "Cap or rework",
        }
    ).fillna("Review")
    plan["business_reason"] = plan.apply(lambda row: _business_reason(row, config), axis=1)
    plan["next_step"] = plan.apply(lambda row: _next_step(row), axis=1)

    action_weight = plan["action"].map({"Increase": 3.0, "Maintain": 2.0, "Limit": 1.0}).fillna(0.0)
    plan["priority_score"] = (
        action_weight * 100
        + plan["allocation_share"] * 100
        + plan["marginal_roas"].clip(0, 10) * 4
        - plan["risk_index"].clip(0, 2) * 8
    )
    plan = plan.sort_values(["priority_score", "allocation"], ascending=[False, False]).reset_index(drop=True)
    plan["priority"] = plan.index + 1
    return plan[columns]


def _action(row: pd.Series, target_roas: float, target_cpa: float) -> str:
    roas_ok = row["expected_roas"] >= target_roas
    cpa_ok = row["expected_cpa"] <= target_cpa if target_cpa > 0 else True
    if roas_ok and cpa_ok:
        return "Increase"
    if row["expected_roas"] >= target_roas * 0.75 or cpa_ok:
        return "Maintain"
    return "Limit"


def _goal_status(
    summary: dict[str, float],
    config: OptimizerConfig,
    increase_share: float,
    limit_share: float,
) -> str:
    roas_ok = config.target_roas <= 0 or summary["expected_roas"] >= config.target_roas
    cpa_ok = config.target_cpa <= 0 or summary["expected_cpa"] <= config.target_cpa
    profit_ok = summary["expected_profit"] >= 0
    goal = config.business_goal

    if goal == "Acquire customers efficiently":
        on_track = cpa_ok and profit_ok
        guarded = summary["expected_cpa"] <= config.target_cpa * 1.15 and summary["expected_roas"] >= config.target_roas * 0.75
    elif goal == "Scale winners":
        on_track = increase_share >= 0.50 and summary["expected_roas"] >= config.target_roas * 0.85 and profit_ok
        guarded = increase_share >= 0.30 and profit_ok
    elif goal == "Reduce waste":
        on_track = limit_share <= 0.15 and roas_ok and cpa_ok
        guarded = limit_share <= 0.30 and (roas_ok or cpa_ok)
    else:
        on_track = roas_ok and cpa_ok and profit_ok
        guarded = (roas_ok or cpa_ok) and profit_ok

    if on_track:
        return "On track"
    if guarded:
        return "Needs guardrails"
    return "Off target"


def _primary_constraint(summary: dict[str, float], config: OptimizerConfig, limit_share: float) -> str:
    if summary["expected_profit"] < 0:
        return "Expected profit is negative after the recommended allocation."
    if config.target_roas > 0 and summary["expected_roas"] < config.target_roas:
        gap = config.target_roas - summary["expected_roas"]
        return f"Expected ROAS is {gap:.2f}x below target."
    if config.target_cpa > 0 and summary["expected_cpa"] > config.target_cpa:
        gap = summary["expected_cpa"] - config.target_cpa
        return f"Expected CPA is ${gap:,.0f} above target."
    if limit_share > 0.25:
        return f"{limit_share:.0%} of budget is assigned to channels marked Limit."
    return "No major constraint in the configured scenario."


def _business_reason(row: pd.Series, config: OptimizerConfig) -> str:
    roas_ok = row["expected_roas"] >= config.target_roas
    cpa_ok = row["expected_cpa"] <= config.target_cpa if config.target_cpa > 0 else True
    if row["action"] == "Increase":
        if roas_ok and cpa_ok:
            return "Expected ROAS and CPA both meet the configured guardrails."
        return "Expected return is strong, but one guardrail still needs monitoring."
    if row["action"] == "Maintain":
        return "Close enough to keep funded, but not strong enough for a larger budget shift."
    return "Expected return misses the configured ROAS or CPA guardrail."


def _next_step(row: pd.Series) -> str:
    if row["action"] == "Increase":
        return "Increase in controlled increments and monitor CPA, ROAS, and saturation weekly."
    if row["action"] == "Maintain":
        return "Hold budget while testing creative, audience, or landing-page improvements."
    return "Cap spend, diagnose the weak KPI, and move flexible budget to higher-priority channels."
