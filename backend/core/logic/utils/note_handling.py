"""Utilities for analyzing custom client notes and related text."""

from __future__ import annotations

import re


def get_client_address_lines(client_info: dict) -> list[str]:
    """Return client's mailing address lines.

    Priority order:
    1. ``client_info['address']``
    2. ``client_info['current_address']`` extracted from the credit report

    The returned list may contain one or two lines. When no address is found,
    an empty list is returned so the caller can render a placeholder line.
    """

    raw = (
        client_info.get("address") or client_info.get("current_address") or ""
    ).strip()
    if not raw:
        return []

    # Normalize separators to detect street vs city/state/zip parts
    raw = raw.replace("\r", "\n")
    parts = [p.strip() for p in re.split(r"\n|,", raw) if p.strip()]

    if len(parts) >= 2:
        line1 = parts[0]
        line2 = ", ".join(parts[1:])
        return [line1, line2]
    return [raw]
