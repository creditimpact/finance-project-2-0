"""Populate masked SSN from profile or corrections."""
from __future__ import annotations

from typing import Mapping


def _mask(ssn: str) -> str:
    if len(ssn) == 4:
        return f"***-**-{ssn}"
    if len(ssn) == 9:
        return f"***-**-{ssn[-4:]}"
    return ssn


def populate_ssn_masked(
    ctx: dict,
    profile: Mapping[str, str] | None = None,
    corrections: Mapping[str, str] | None = None,
) -> None:
    """Populate ``ssn_masked`` on ``ctx`` if missing."""

    if ctx.get("ssn_masked"):
        return
    corrections = corrections or {}
    profile = profile or {}
    ssn = corrections.get("ssn") or corrections.get("ssn_last4")
    ssn = ssn or profile.get("ssn") or profile.get("ssn_last4")
    if ssn:
        ctx["ssn_masked"] = _mask(str(ssn))
