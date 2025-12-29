"""Populate medical_status from evidence."""
from __future__ import annotations

from typing import Mapping


def populate_medical_status(
    ctx: dict,
    evidence: Mapping[str, str] | None = None,
) -> None:
    """Populate ``medical_status`` on ``ctx`` if missing."""

    if ctx.get("medical_status"):
        return
    evidence = evidence or {}
    status = evidence.get("medical_status") or evidence.get("status")
    if status:
        ctx["medical_status"] = status
