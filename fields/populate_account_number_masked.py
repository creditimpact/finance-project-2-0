"""Populate account_number_masked from tri-merge evidence."""
from __future__ import annotations

from typing import Mapping


def populate_account_number_masked(
    ctx: dict, tri_merge: Mapping[str, str] | None = None
) -> None:
    """Populate ``account_number_masked`` on ``ctx`` if missing."""

    if ctx.get("account_number_masked"):
        return
    tri_merge = tri_merge or {}
    masked = tri_merge.get("account_number_masked") or tri_merge.get("account_number")
    if masked:
        ctx["account_number_masked"] = masked
