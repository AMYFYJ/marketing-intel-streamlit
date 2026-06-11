from __future__ import annotations

import importlib

import pandas as pd


def test_app_modules_import() -> None:
    modules = [
        "streamlit_app",
        "features.performance_dashboard",
        "features.competitor_intelligence",
        "features.demand_pulse",
        "features.budget_optimizer",
        "utils.layout",
    ]
    for module in modules:
        assert importlib.import_module(module)


def test_competitor_dropdown_options_include_defaults() -> None:
    competitor_intelligence = importlib.import_module("features.competitor_intelligence")

    assert set(competitor_intelligence.DEFAULT_COMPETITORS).issubset(competitor_intelligence.COMPETITOR_OPTIONS)
    assert len(competitor_intelligence.COMPETITOR_OPTIONS) == len(set(competitor_intelligence.COMPETITOR_OPTIONS))


def test_watch_dropdown_options_are_clean_fetch_keywords() -> None:
    competitor_intelligence = importlib.import_module("features.competitor_intelligence")
    all_theme_keywords = {
        theme
        for themes in competitor_intelligence.THEME_OPTIONS_BY_CATEGORY.values()
        for theme in themes
    }
    default_keywords = competitor_intelligence._theme_keywords_from_options(
        competitor_intelligence.DEFAULT_THEME_OPTIONS
    )

    assert set(competitor_intelligence.DEFAULT_THEMES).issubset(all_theme_keywords)
    assert "beauty" in all_theme_keywords
    assert len(competitor_intelligence.THEME_OPTIONS) == len(set(competitor_intelligence.THEME_OPTIONS))
    assert set(default_keywords) == set(competitor_intelligence.DEFAULT_THEMES)
    assert all(competitor_intelligence.THEME_LABEL_SEPARATOR not in keyword for keyword in default_keywords)


def test_content_category_dropdown_uses_clean_theme_labels() -> None:
    competitor_intelligence = importlib.import_module("features.competitor_intelligence")

    assert competitor_intelligence.THEME_OPTIONS[:3] == (
        "Beauty & Personal Care: beauty",
        "Beauty & Personal Care: skincare",
        "Beauty & Personal Care: makeup",
    )
    assert competitor_intelligence._theme_keyword_from_option("Beauty & Personal Care: beauty") == "beauty"
    assert competitor_intelligence._theme_keyword_from_option("beauty") == "beauty"
    assert competitor_intelligence._theme_display_label("Beauty & Personal Care: beauty") == "Beauty"
    assert competitor_intelligence._theme_display_label("Beauty & Personal Care: hair care") == "Hair Care"
    assert competitor_intelligence._theme_display_label("AI & Automation: AI marketing") == "AI Marketing"
    assert competitor_intelligence._theme_display_label("Ads & Media: connected TV") == "Connected TV"
    assert competitor_intelligence._theme_display_label("Commerce & Vertical Plays: SaaS marketing") == "SaaS Marketing"
    assert competitor_intelligence._theme_display_label("Audience Pains: prove ROI") == "Prove ROI"
    assert competitor_intelligence._theme_display_label("Measurement & Analytics: A/B testing") == "A/B Testing"


def test_source_selector_options_include_default_sources() -> None:
    competitor_intelligence = importlib.import_module("features.competitor_intelligence")

    assert set(competitor_intelligence.DEFAULT_SOURCES).issubset(competitor_intelligence.SOURCES)
    assert len(competitor_intelligence.SOURCES) == len(set(competitor_intelligence.SOURCES))
    assert "LinkedIn Ad Library" in competitor_intelligence.SOURCES
    assert "X Ads Repository (EU Only)" in competitor_intelligence.SOURCES
    assert not any("Amazon" in source for source in competitor_intelligence.SOURCES)
    assert competitor_intelligence.DEFAULT_MAX_ITEMS_PER_SOURCE == 15


def test_platform_strategy_summary_shapes_competitor_rows() -> None:
    competitor_intelligence = importlib.import_module("features.competitor_intelligence")
    frame = pd.DataFrame(
        [
            {
                "competitor": "Acme",
                "source": "TikTok Creative Center",
                "creative_format": "Video",
                "cta": "Shop now",
                "campaign_type": "Discount / Promo",
                "title": "Limited time bundle",
                "published_at": pd.Timestamp("2026-02-01T00:00:00Z"),
                "signal_strength": 80.0,
            }
        ]
    )

    summary = competitor_intelligence._platform_strategy_summary(frame)

    assert summary.loc[0, "Competitor"] == "Acme"
    assert summary.loc[0, "Top Format"] == "Video"
    assert summary.loc[0, "Top Campaign Type"] == "Discount / Promo"
    assert "Shop now" in summary.loc[0, "Suggested Test"]


def test_live_result_signature_tracks_current_setup() -> None:
    competitor_intelligence = importlib.import_module("features.competitor_intelligence")

    base = competitor_intelligence._query_signature((), ("beauty",), "US", 15, ("Reddit", "GDELT"))
    changed_sources = competitor_intelligence._query_signature((), ("beauty",), "US", 15, ("GDELT",))
    changed_categories = competitor_intelligence._query_signature((), ("skincare",), "US", 15, ("Reddit", "GDELT"))

    assert base == ((), ("beauty",), "US", 15, ("Reddit", "GDELT"))
    assert base != changed_sources
    assert base != changed_categories


def test_signal_summary_help_covers_cards() -> None:
    competitor_intelligence = importlib.import_module("features.competitor_intelligence")
    expected_cards = {
        "Observed signals",
        "Active sources",
        "Scan scope",
        "Share leader",
        "Top message",
        "Top CTA",
        "Newest signal",
        "Test-next ideas",
        "Avg strength",
    }

    assert set(competitor_intelligence.SIGNAL_SUMMARY_HELP) == expected_cards
    assert all(competitor_intelligence.SIGNAL_SUMMARY_HELP.values())


def test_frontier_cpc_axis_keeps_cents() -> None:
    performance_dashboard = importlib.import_module("features.performance_dashboard")

    assert performance_dashboard._axis_tickformat("cpc") == "$,.2f"
    assert performance_dashboard._axis_tickformat("cpa") == "$,.0f"


def test_performance_diagnostic_helpers_shape_ranked_outputs() -> None:
    performance_dashboard = importlib.import_module("features.performance_dashboard")
    campaign_data = importlib.import_module("data_sources.campaign_data")
    frame = campaign_data.generate_campaign_sample(rows=500, seed=21)

    leaderboard = performance_dashboard._ranked_segments(frame, "platform", "cpa", 5)
    assert len(leaderboard) <= 5
    assert leaderboard["conversions"].gt(0).all()
    assert leaderboard["cpa"].is_monotonic_increasing

    funnel = performance_dashboard._funnel_leakage_matrix(frame, "platform", 5)
    assert list(funnel.columns) == [
        "Reach / Impr.",
        "Click / Impr.",
        "LPV / Click",
        "Cart / LPV",
        "Conv. / Cart",
        "Conv. / Click",
        "Conv. / Impr.",
    ]
    assert len(funnel) <= 5

    normalized = performance_dashboard._normalize_rate_matrix(funnel)
    assert normalized.max().le(1).all()

    help_html = performance_dashboard._funnel_metric_help_html()
    for label in funnel.columns:
        assert label in help_html


def test_top_movers_rank_by_absolute_delta() -> None:
    performance_dashboard = importlib.import_module("features.performance_dashboard")
    campaign_data = importlib.import_module("data_sources.campaign_data")
    frame = campaign_data.generate_campaign_sample(rows=1_000, seed=34)

    max_date = frame["date"].max()
    current = frame[frame["date"] > max_date - pd.Timedelta(days=90)]
    previous = frame[
        (frame["date"] <= max_date - pd.Timedelta(days=90))
        & (frame["date"] > max_date - pd.Timedelta(days=180))
    ]

    movers = performance_dashboard._top_movers(current, previous, "platform", "profit", 5)

    assert len(movers) <= 5
    assert movers["abs_delta"].is_monotonic_decreasing


def test_demand_language_playbook_prioritizes_actionable_phrases() -> None:
    demand_pulse = importlib.import_module("features.demand_pulse")
    frame = pd.DataFrame(
        [
            {
                "priority": "Medium",
                "recommended_action": "Content idea",
                "keyword": "retail media",
                "intent": "Pain",
                "audience_language": "Retail media attribution feels expensive and hard to trust",
                "urgency_score": 92.0,
                "freshness_hours": 2.0,
                "noise_risk": "Low",
                "source": "Reddit",
                "url": "https://example.com/pain",
            },
            {
                "priority": "High",
                "recommended_action": "Test now",
                "keyword": "AI marketing",
                "intent": "Question",
                "audience_language": "How should teams use AI marketing without workflow problems?",
                "urgency_score": 78.0,
                "freshness_hours": 4.0,
                "noise_risk": "Low",
                "source": "GDELT",
                "url": "https://example.com/question",
            },
        ]
    )

    playbook = demand_pulse._build_language_playbook(frame, limit=2)

    assert list(playbook.columns) == [
        "Priority",
        "Action",
        "Keyword",
        "Intent",
        "Reusable phrase",
        "Recommended use",
        "Urgency",
        "Noise risk",
        "Source",
        "URL",
    ]
    assert playbook.loc[0, "Action"] == "Test now"
    assert playbook.loc[0, "Recommended use"] == "How-to or FAQ hook"


def test_demand_language_snapshot_uses_question_mark_fallback() -> None:
    demand_pulse = importlib.import_module("features.demand_pulse")
    phrase = "How should teams use AI marketing without workflow problems?"
    frame = pd.DataFrame(
        [
            {
                "priority": "High",
                "recommended_action": "Test now",
                "keyword": "AI marketing",
                "intent": "Pain",
                "audience_language": phrase,
                "urgency_score": 82.0,
                "freshness_hours": 1.0,
                "noise_risk": "Low",
                "source": "GDELT",
                "url": "https://example.com/question",
            }
        ]
    )

    snapshot = demand_pulse._language_snapshot(frame)

    assert snapshot[2][1] == phrase


def test_demand_pulse_simplified_controls_and_help_coverage() -> None:
    demand_pulse = importlib.import_module("features.demand_pulse")
    expected_cards = {
        "Active keywords",
        "Source coverage",
        "Rising topic",
        "Urgency score",
        "Sentiment shift",
        "Audience language",
        "Next move",
        "Noise risk",
    }

    assert demand_pulse._time_window_days("Past 48 hours") == 2
    assert demand_pulse._time_window_days("Unknown") == 7
    assert demand_pulse.DEFAULT_MAX_ITEMS_PER_SOURCE == 20
    assert set(demand_pulse.DEMAND_BRIEF_HELP) == expected_cards
    assert all(demand_pulse.DEMAND_BRIEF_HELP.values())
    assert "Velocity by Keyword" in demand_pulse.CHART_HELP
    assert "Raw Signal Feed" in demand_pulse.CHART_HELP
    assert demand_pulse._safe_external_url("https://example.com/path") == "https://example.com/path"
    assert demand_pulse._safe_external_url("javascript:alert(1)") == "#"
