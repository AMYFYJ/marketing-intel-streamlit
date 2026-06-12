from __future__ import annotations

from typing import Iterable

import pandas as pd

# Tokens that should not be plainly capitalized in display labels.
_LABEL_TOKEN_OVERRIDES = {
    "roas": "ROAS",
    "cpa": "CPA",
    "ctr": "CTR",
    "cvr": "CVR",
    "cta": "CTA",
    "url": "URL",
    "id": "ID",
    "saas": "SaaS",
    "of": "of",
    "per": "per",
    "and": "and",
    "vs": "vs",
    "by": "by",
}


def display_label(column: str) -> str:
    """Turn a snake_case column name into a Title Case display label."""
    return " ".join(_LABEL_TOKEN_OVERRIDES.get(token, token.capitalize()) for token in str(column).split("_"))


def display_labels(columns: Iterable[str]) -> dict[str, str]:
    return {column: display_label(column) for column in columns}


def title_case_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename a display dataframe's columns to Title Case labels."""
    return frame.rename(columns=display_labels(frame.columns))


def format_currency(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_roas(value: float) -> str:
    return f"{value:.2f}x"
