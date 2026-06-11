from __future__ import annotations


def format_currency(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_roas(value: float) -> str:
    return f"{value:.2f}x"
