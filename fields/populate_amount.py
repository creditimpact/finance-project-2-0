"""Populate amount from evidence."""
from __future__ import annotations

from typing import Mapping


def populate_amount(
    ctx: dict,
    evidence: Mapping[str, object] | None = None,
) -> None:
    """Populate ``amount`` on ``ctx`` if missing."""

    if ctx.get("amount") is not None:
        return
    evidence = evidence or {}
    amount = evidence.get("amount")
    if amount is not None:
        ctx["amount"] = amount
