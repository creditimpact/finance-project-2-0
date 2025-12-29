"""Populate corrected address from profile or planner corrections."""
from __future__ import annotations

from typing import Mapping


def populate_address(
    ctx: dict,
    profile: Mapping[str, str] | None = None,
    corrections: Mapping[str, str] | None = None,
) -> None:
    """Populate ``address`` on ``ctx`` if missing."""

    if ctx.get("address"):
        return
    corrections = corrections or {}
    profile = profile or {}
    addr = corrections.get("address") or profile.get("address")
    if addr:
        ctx["address"] = addr
