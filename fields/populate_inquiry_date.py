"""Populate inquiry_date from evidence."""
from __future__ import annotations

from typing import Mapping


def populate_inquiry_date(
    ctx: dict,
    evidence: Mapping[str, str] | None = None,
) -> None:
    """Populate ``inquiry_date`` on ``ctx`` if missing."""

    if ctx.get("inquiry_date"):
        return
    evidence = evidence or {}
    date = evidence.get("inquiry_date") or evidence.get("date")
    if date:
        ctx["inquiry_date"] = date
