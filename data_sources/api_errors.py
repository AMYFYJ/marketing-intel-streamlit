"""Helpers for reporting HTTP API failures without leaking credentials.

requests embeds the full request URL — including query-string secrets such as
``access_token`` or ``key`` — in exception text, so status details shown in
the UI must never contain raw exception strings or request URLs.
"""
from __future__ import annotations

import re

_QUERY_STRING = re.compile(r"\?\S+")


def strip_query_strings(text: str) -> str:
    """Drop URL query strings (where tokens and keys live) from error text."""
    return _QUERY_STRING.sub("", text)


def response_error_detail(response: object, status_code: int) -> str:
    """Summarize an HTTP error from the response body instead of the request URL.

    Meta Graph and Google APIs return ``{"error": {"message", "type", "code"}}``
    bodies that explain the failure far better than the bare status line.
    """
    message = ""
    error_type = None
    code = None
    try:
        payload = response.json()
        error = payload.get("error") if isinstance(payload, dict) else None
        if isinstance(error, dict):
            message = str(error.get("message") or "").strip()
            error_type = error.get("type")
            code = error.get("code")
    except Exception:
        pass
    qualifier = " ".join(str(part) for part in (error_type, f"code {code}" if code is not None else None) if part)
    label = f"HTTP {status_code}" + (f" ({qualifier})" if qualifier else "")
    return f"{label}: {strip_query_strings(message)}" if message else label
