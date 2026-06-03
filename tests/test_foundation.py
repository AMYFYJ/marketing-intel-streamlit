from __future__ import annotations

import importlib


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
