"""Compute days_since_cra_result from an outcome timestamp."""
from __future__ import annotations

from datetime import datetime
from typing import Mapping


def populate_days_since_cra_result(
    ctx: dict,
    outcome: Mapping[str, object] | None = None,
    *,
    now: datetime | str | None = None,
) -> None:
    """Populate ``days_since_cra_result`` on ``ctx`` if missing.

    ``outcome`` may provide a ``timestamp`` field representing the last CRA
    result date. If ``timestamp`` is a string, it must be ISO formatted.
    """

    if ctx.get("days_since_cra_result") is not None:
        return

    if not outcome:
        return

    ts = outcome.get("timestamp") or outcome.get("cra_result_at")
    if not ts:
        return

    if isinstance(ts, str):
        result_time = datetime.fromisoformat(ts)
    else:
        result_time = ts

    if isinstance(now, str):
        now_dt = datetime.fromisoformat(now)
    else:
        now_dt = now
    if now_dt is None:
        return

    ctx["days_since_cra_result"] = (now_dt - result_time).days
