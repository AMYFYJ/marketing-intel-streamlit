from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_sources.campaign_data import AUDIENCES, PLATFORM_CREATIVE_FORMATS


@dataclass(frozen=True)
class GoalSpec:
    """Describes one end-goal a marketer can plan toward.

    ``metric_column`` is the dataset column the goal maximizes. ``affinity_objectives`` are the
    campaign objectives most aligned with the goal. When ``strict_affinity`` is set the goal's
    metric only really exists inside those objectives (leads live in Lead-Gen campaigns), so the
    filter is enforced rather than treated as a soft preference. ``is_currency`` controls money
    formatting; ``is_proxy`` flags goals (followers) the ad data can only approximate.
    """

    key: str
    label: str
    metric_column: str
    affinity_objectives: tuple[str, ...]
    cost_label: str
    volume_label: str
    is_currency: bool = False
    is_proxy: bool = False
    strict_affinity: bool = False
    note: str = ""


GOAL_DEFINITIONS: dict[str, GoalSpec] = {
    "conversions": GoalSpec(
        "conversions", "Conversions", "conversions", ("Sales", "Retargeting"),
        "Cost per conversion", "Conversions",
    ),
    "revenue": GoalSpec(
        "revenue", "Revenue / Sales", "revenue", ("Sales",),
        "Cost per $1 revenue", "Revenue", is_currency=True,
    ),
    "traffic": GoalSpec(
        "traffic", "Website traffic", "clicks", ("Traffic",),
        "Cost per click", "Clicks",
    ),
    "awareness": GoalSpec(
        "awareness", "Brand awareness", "impressions", ("Awareness",),
        "CPM (cost per 1k impressions)", "Impressions",
    ),
    "leads": GoalSpec(
        "leads", "Leads", "conversions", ("Lead Gen",),
        "Cost per lead", "Leads", strict_affinity=True,
    ),
    "app_installs": GoalSpec(
        "app_installs", "App installs", "conversions", ("App Install",),
        "Cost per install", "Installs", strict_affinity=True,
    ),
    "engagement": GoalSpec(
        "engagement", "Followers / engagement", "engagement", ("Awareness", "Traffic"),
        "Cost per engagement", "Engagements", is_proxy=True,
        note="Ad platforms do not report follower growth; this uses ad engagement "
        "(likes/comments/shares/saves) as a directional proxy for community growth.",
    ),
}

# Conversion-type goals can use any audience; upper-funnel goals should not be sold against
# retargeting/lapsed pools (you don't run awareness to lapsed customers, and those audiences are
# too small to absorb real budget).
_UPPER_FUNNEL_POOL = ("Prospecting", "Lookalike", "High Intent")
GOAL_AUDIENCE_POOLS: dict[str, tuple[str, ...]] = {
    "awareness": _UPPER_FUNNEL_POOL,
    "engagement": _UPPER_FUNNEL_POOL,
    "traffic": _UPPER_FUNNEL_POOL,
}
_CONVERSION_GOALS = {"conversions", "revenue", "leads", "app_installs"}

# Competitor sources we can attribute to an ad platform for the under-contested tilt.
_SOURCE_TO_PLATFORM = {
    "Meta Ad Library": "Meta",
    "TikTok Creative Center": "TikTok",
    "YouTube": "YouTube",
    "Reddit": "Reddit",
}


@dataclass
class GoalPlan:
    goal: GoalSpec
    allocation: pd.DataFrame
    summary: dict[str, float] = field(default_factory=dict)
    filter_relaxed: bool = False


ALLOCATION_COLUMNS = [
    "platform",
    "allocation",
    "allocation_share",
    "efficiency",
    "saturation_index",
    "historical_spend",
    "expected_metric",
    "expected_cost_per_metric",
    "recommended_campaign_type",
    "recommended_audience",
    "rows",
]


def list_goal_labels() -> dict[str, str]:
    """Map each goal key to its display label, for building a selectbox."""
    return {spec.label: key for key, spec in GOAL_DEFINITIONS.items()}


def plan_for_goal(
    frame: pd.DataFrame,
    total_budget: float,
    goal_key: str,
    market: str | None = None,
    industry: str | None = None,
    max_platforms: int | None = None,
    min_allocation: float = 0.0,
    platform_tilts: dict[str, float] | None = None,
) -> GoalPlan:
    """Allocate ``total_budget`` across platforms to maximize a chosen goal metric.

    Platforms are scored by goal-metric-per-dollar (efficiency) weighted by saturation headroom and
    optional strategic tilts, budget is split proportionally, and an over-investment decay penalizes
    pushing far past a platform's historical scale. ``min_allocation`` drops platforms that would get
    less than a viable spend and ``max_platforms`` caps the channel count — both renormalize so the
    plan still sums to the budget. For each funded platform it recommends a valid campaign type and a
    goal-appropriate audience.
    """
    if goal_key not in GOAL_DEFINITIONS:
        raise ValueError(f"unknown goal: {goal_key}")
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    goal = GOAL_DEFINITIONS[goal_key]
    if goal.metric_column not in frame.columns:
        raise ValueError(f"frame is missing the goal metric column '{goal.metric_column}'")

    working, filter_relaxed = _apply_filters(frame, goal, market, industry)
    if working.empty:
        return GoalPlan(goal=goal, allocation=_empty_allocation(), summary=_empty_summary(), filter_relaxed=filter_relaxed)

    summary = _summarize_platforms(working, goal.metric_column)
    if summary.empty:
        return GoalPlan(goal=goal, allocation=_empty_allocation(), summary=_empty_summary(), filter_relaxed=filter_relaxed)

    # Base score = efficiency (goal per $) tilted toward platforms with headroom, then by any
    # strategic tilt (e.g. less-contested competitor space).
    headroom = (1.05 - summary["saturation_index"].clip(0.15, 1.0)).clip(0.1, 1.0)
    score = (summary["efficiency"].clip(lower=0) * headroom).replace([np.inf, -np.inf], np.nan).fillna(0)
    if platform_tilts:
        tilt = summary["platform"].map(lambda p: float(platform_tilts.get(p, 1.0))).clip(lower=0.1)
        score = score * tilt
    summary = summary.assign(score=score)

    kept = _select_platforms(summary, total_budget, max_platforms, min_allocation)
    result = kept.copy()
    result["allocation_share"] = _shares_from_score(result["score"])
    result["allocation"] = result["allocation_share"] * total_budget

    result["expected_metric"] = _response(
        result["allocation"].to_numpy(dtype=float),
        result["historical_spend"].to_numpy(dtype=float),
        result["saturation_index"].to_numpy(dtype=float),
        result["efficiency"].to_numpy(dtype=float),
    )
    result["expected_cost_per_metric"] = (
        result["allocation"] / result["expected_metric"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0)

    recommendations = _best_dimension_per_platform(working, goal)
    result = result.merge(recommendations, on="platform", how="left")
    result["recommended_campaign_type"] = result["recommended_campaign_type"].fillna("—")
    result["recommended_audience"] = result["recommended_audience"].fillna("—")

    result = result.sort_values("allocation", ascending=False).reset_index(drop=True)
    allocation = result[ALLOCATION_COLUMNS].copy()
    return GoalPlan(
        goal=goal,
        allocation=allocation,
        summary=summarize_goal_plan(allocation, goal),
        filter_relaxed=filter_relaxed,
    )


def summarize_goal_plan(allocation: pd.DataFrame, goal: GoalSpec) -> dict[str, float]:
    """Roll a per-platform allocation up to plan-level totals."""
    if allocation.empty:
        return _empty_summary()
    budget = float(allocation["allocation"].sum())
    metric = float(allocation["expected_metric"].sum())
    return {
        "budget": budget,
        "expected_metric": metric,
        "blended_cost_per_metric": budget / metric if metric else 0.0,
        "platforms": int(len(allocation)),
    }


def platform_response_curves(allocation: pd.DataFrame, points: int = 25) -> pd.DataFrame:
    """Expected goal volume for each platform as its spend scales 0 → 2× its planned allocation.

    The curve flattening visualizes diminishing returns: where it bends, extra spend stops buying
    proportional volume.
    """
    if allocation.empty:
        return pd.DataFrame(columns=["platform", "spend", "expected_metric"])
    rows = []
    for _, row in allocation.iterrows():
        top = max(float(row["allocation"]) * 2.0, 1.0)
        grid = np.linspace(0.0, top, points)
        metric = _response(
            grid,
            np.full(points, float(row["historical_spend"])),
            np.full(points, float(row["saturation_index"])),
            np.full(points, float(row["efficiency"])),
        )
        rows.append(pd.DataFrame({"platform": row["platform"], "spend": grid, "expected_metric": metric}))
    return pd.concat(rows, ignore_index=True)


def budget_sweep(
    frame: pd.DataFrame,
    goal_key: str,
    base_budget: float,
    multipliers: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0),
    **plan_kwargs,
) -> pd.DataFrame:
    """Re-run the plan across budget levels to expose the efficient frontier.

    Returns one row per budget with expected goal volume, blended cost-per-goal, and the marginal
    metric gained per extra dollar versus the previous (smaller) budget — i.e. what the next dollars
    actually buy.
    """
    rows = []
    for mult in sorted(multipliers):
        budget = base_budget * mult
        plan = plan_for_goal(frame, budget, goal_key, **plan_kwargs)
        rows.append(
            {
                "budget": budget,
                "expected_metric": plan.summary["expected_metric"],
                "blended_cost_per_metric": plan.summary["blended_cost_per_metric"],
            }
        )
    sweep = pd.DataFrame(rows).sort_values("budget").reset_index(drop=True)
    delta_metric = sweep["expected_metric"].diff()
    delta_budget = sweep["budget"].diff()
    sweep["marginal_metric_per_dollar"] = (delta_metric / delta_budget).replace([np.inf, -np.inf], np.nan).fillna(
        sweep["expected_metric"] / sweep["budget"].replace(0, np.nan)
    )
    sweep["marginal_cost_per_metric"] = (delta_budget / delta_metric).replace([np.inf, -np.inf], np.nan).fillna(0)
    return sweep


def compute_platform_tilts(competitor_items: pd.DataFrame, strength: float = 0.15) -> dict[str, float]:
    """Tilt budget toward ad platforms where competitors are *under*-represented (whitespace).

    Competitor item counts are mapped to ad platforms and turned into a pressure share; platforms
    below the average pressure get a tilt > 1, crowded ones < 1. Only platforms attributable to a
    competitor source are tilted — others are left neutral (1.0).
    """
    if competitor_items is None or competitor_items.empty or "source" not in competitor_items.columns:
        return {}
    mapped = competitor_items["source"].map(_SOURCE_TO_PLATFORM).dropna()
    if mapped.empty:
        return {}
    counts = mapped.value_counts()
    shares = counts / counts.sum()
    mean_share = float(shares.mean())
    return {platform: float(1 + strength * (mean_share - share)) for platform, share in shares.items()}


def _response(
    allocation: np.ndarray, historical_spend: np.ndarray, saturation: np.ndarray, efficiency: np.ndarray
) -> np.ndarray:
    """Expected goal volume for a spend level, with over-investment decay. Shared by plan + curves."""
    allocation = np.asarray(allocation, dtype=float)
    hist = np.where(np.asarray(historical_spend, dtype=float) <= 0, np.nan, historical_spend)
    relative_budget = np.nan_to_num(allocation / hist, nan=0.05, posinf=0.05, neginf=0.05)
    sat = np.clip(np.asarray(saturation, dtype=float), 0.15, 0.98)
    decay = 1 / (1 + np.power(relative_budget * 6.0, 0.6) * (1 - sat))
    decay = np.clip(decay, 0.2, 1.05)
    return allocation * np.clip(np.asarray(efficiency, dtype=float), 0, None) * decay


def _shares_from_score(score: pd.Series) -> pd.Series:
    total = float(score.sum())
    if total <= 0:
        return pd.Series(1 / len(score), index=score.index)
    return score / total


def _select_platforms(
    summary: pd.DataFrame, total_budget: float, max_platforms: int | None, min_allocation: float
) -> pd.DataFrame:
    """Pick the platform set to fund: cap the count, then drop sub-viable spends, renormalizing."""
    kept = summary.sort_values("score", ascending=False).reset_index(drop=True)
    if max_platforms and max_platforms > 0:
        kept = kept.head(max_platforms).reset_index(drop=True)

    if min_allocation and min_allocation > 0:
        # Iteratively drop the smallest platform whose renormalized allocation is below the floor.
        while len(kept) > 1:
            shares = _shares_from_score(kept["score"])
            allocations = shares * total_budget
            below = allocations < min_allocation
            if not below.any():
                break
            drop_idx = allocations[below].idxmin()
            kept = kept.drop(index=drop_idx).reset_index(drop=True)
    return kept


def _apply_filters(
    frame: pd.DataFrame, goal: GoalSpec, market: str | None, industry: str | None
) -> tuple[pd.DataFrame, bool]:
    base = frame
    if market and market != "All" and "geo" in base.columns:
        base = base[base["geo"] == market]
    if industry and industry != "All" and "industry" in base.columns:
        base = base[base["industry"] == industry]
    if base.empty:
        return base, False

    if goal.affinity_objectives and "objective" in base.columns:
        affined = base[base["objective"].isin(goal.affinity_objectives)]
        enough = affined["platform"].nunique() >= 2 and len(affined) >= 50
        if goal.strict_affinity:
            # The goal's metric only exists inside these objectives; enforce unless it would starve.
            return (affined, False) if enough else (base, True)
        # Soft preference: use goal-aligned objectives when there's enough data, else fall back.
        return (affined, False) if enough else (base, True)
    return base, False


def _summarize_platforms(frame: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    has_dri = "diminishing_return_index" in frame.columns
    grouped = frame.groupby("platform").agg(
        historical_spend=("spend", "sum"),
        metric_total=(metric_column, "sum"),
        saturation_index=("diminishing_return_index", "mean") if has_dri else ("spend", "size"),
        rows=("spend", "count"),
    ).reset_index()
    if not has_dri:
        grouped["saturation_index"] = 0.5
    grouped["efficiency"] = grouped["metric_total"] / grouped["historical_spend"].replace(0, np.nan)
    grouped["efficiency"] = grouped["efficiency"].replace([np.inf, -np.inf], np.nan).fillna(0)
    grouped["saturation_index"] = grouped["saturation_index"].fillna(0.5)
    return grouped


def _best_dimension_per_platform(frame: pd.DataFrame, goal: GoalSpec) -> pd.DataFrame:
    rows = []
    for platform, group in frame.groupby("platform"):
        rows.append(
            {
                "platform": platform,
                "recommended_campaign_type": _top_creative(group, platform, goal.metric_column),
                "recommended_audience": _recommend_audience(group, goal),
            }
        )
    return pd.DataFrame(rows, columns=["platform", "recommended_campaign_type", "recommended_audience"])


def _ranked_by_efficiency(group: pd.DataFrame, dimension: str, metric_column: str, allowed: set | None = None) -> list[str]:
    if dimension not in group.columns:
        return []
    agg = group.groupby(dimension).agg(metric=(metric_column, "sum"), spend=("spend", "sum"))
    agg = agg[agg["spend"] > 0]
    if allowed is not None:
        agg = agg[agg.index.isin(allowed)]
    if agg.empty:
        return []
    efficiency = (agg["metric"] / agg["spend"]).sort_values(ascending=False)
    return [str(idx) for idx in efficiency.index]


def _top_creative(group: pd.DataFrame, platform: str, metric_column: str) -> str:
    allowed = set(PLATFORM_CREATIVE_FORMATS.get(platform, ()))
    ranked = _ranked_by_efficiency(group, "creative_format", metric_column, allowed=allowed or None)
    return ranked[0] if ranked else "—"


def _recommend_audience(group: pd.DataFrame, goal: GoalSpec) -> str:
    pool = set(GOAL_AUDIENCE_POOLS.get(goal.key, AUDIENCES))
    ranked = _ranked_by_efficiency(group, "audience_segment", goal.metric_column, allowed=pool)
    if not ranked:
        return "—"
    top = ranked[0]
    # Retargeting always looks efficient but can't absorb large budget — pair it with a scalable
    # audience and cap it, for conversion-type goals where retargeting is even eligible.
    if goal.key in _CONVERSION_GOALS and top == "Retargeting":
        next_best = next((a for a in ranked[1:] if a != "Retargeting"), "Prospecting")
        return f"Retargeting + {next_best} (cap retargeting ~30%)"
    return top


def _empty_allocation() -> pd.DataFrame:
    return pd.DataFrame(columns=ALLOCATION_COLUMNS)


def _empty_summary() -> dict[str, float]:
    return {"budget": 0.0, "expected_metric": 0.0, "blended_cost_per_metric": 0.0, "platforms": 0}
