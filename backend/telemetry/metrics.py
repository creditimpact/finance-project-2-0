"""Lightweight telemetry helpers used across the codebase."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Mapping, Protocol


class _EmitCounter(Protocol):
    def __call__(self, name: str, increment: float | Mapping[str, Any] = 1) -> None:
        ...


@lru_cache(maxsize=1)
def _resolve_tracker_emit_counter() -> _EmitCounter | None:
    """Return the analytics tracker ``emit_counter`` if available."""

    try:  # pragma: no cover - defensive import
        from backend.analytics.analytics_tracker import emit_counter as tracker_emit_counter
    except Exception:
        return None
    return tracker_emit_counter


def emit_counter(name: str, value: float | Mapping[str, Any] = 1) -> None:
    """Emit a counter metric with a graceful fallback when analytics is unavailable."""

    tracker_emit_counter = _resolve_tracker_emit_counter()
    if tracker_emit_counter is not None:
        tracker_emit_counter(name, value)
        return

    namespace = os.getenv("ANALYTICS_NAMESPACE", "local")
    logging.info("[METRIC] %s.%s = %s", namespace, name, value)
