"""UI telemetry ingestion helpers."""
from __future__ import annotations

from typing import Any

from .emit import emit as telemetry_emit


def emit(event_type: str, **fields: Any) -> None:
    """Forward a UI telemetry event to the core telemetry emitter."""
    telemetry_emit(event_type, fields)
