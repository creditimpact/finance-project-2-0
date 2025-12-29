"""Populate inquiry_creditor_name from evidence."""
from __future__ import annotations

from typing import Mapping


def populate_inquiry_creditor_name(
    ctx: dict,
    evidence: Mapping[str, str] | None = None,
) -> None:
    """Populate ``inquiry_creditor_name`` on ``ctx`` if missing."""

    if ctx.get("inquiry_creditor_name"):
        return
    evidence = evidence or {}
    name = evidence.get("inquiry_creditor_name") or evidence.get("name")
    if name:
        ctx["inquiry_creditor_name"] = name
