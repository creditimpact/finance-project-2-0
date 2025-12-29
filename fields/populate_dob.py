"""Populate date of birth from profile or corrections."""
from __future__ import annotations

from typing import Mapping


def populate_dob(
    ctx: dict,
    profile: Mapping[str, str] | None = None,
    corrections: Mapping[str, str] | None = None,
) -> None:
    """Populate ``date_of_birth`` on ``ctx`` if missing."""

    if ctx.get("date_of_birth"):
        return
    corrections = corrections or {}
    profile = profile or {}
    dob = corrections.get("date_of_birth") or profile.get("date_of_birth")
    if dob:
        ctx["date_of_birth"] = dob
