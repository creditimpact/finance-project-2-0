"""Populate corrected name from profile or planner corrections."""
from __future__ import annotations

from typing import Mapping


def populate_name(
    ctx: dict,
    profile: Mapping[str, str] | None = None,
    corrections: Mapping[str, str] | None = None,
) -> None:
    """Populate ``name`` on ``ctx`` if missing."""

    if ctx.get("name"):
        return
    corrections = corrections or {}
    profile = profile or {}
    name = corrections.get("name") or profile.get("name")
    if name:
        ctx["name"] = name
