from typing import Mapping, Any, Optional, Callable
import time

_emit: Optional[Callable[[str, Mapping[str, Any]], None]] = None

def set_emitter(fn: Optional[Callable[[str, Mapping[str, Any]], None]]) -> None:
    """Swap the global emitter for tests or integration."""
    global _emit
    _emit = fn

def get_emitter() -> Optional[Callable[[str, Mapping[str, Any]], None]]:
    """Return the currently registered emitter, if any."""
    return _emit

def emit(event: str, **fields: Any) -> None:
    """Fire-and-forget; safe if no emitter is registered."""
    if _emit:
        try:
            _emit(event, fields)
        except Exception:
            # Never let telemetry break the app
            pass

class timed:
    """Context manager to measure duration_ms and emit after block."""

    def __init__(self, event: str, **base_fields: Any):
        self.event = event
        self.base = base_fields

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dur = (time.perf_counter() - self.t0) * 1000.0
        emit(self.event, duration_ms=round(dur, 3), **self.base)
