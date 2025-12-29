import logging
from backend.core.config.flags import FLAGS

log = logging.getLogger(__name__)

def increment(name: str, value: int = 1, tags: dict | None = None) -> None:
    """Increment a counter metric."""
    if not getattr(FLAGS, "metrics_enabled", True):
        return
    log.debug("METRIC increment name=%s value=%s tags=%s", name, value, tags or {})

def gauge(name: str, value: float, tags: dict | None = None) -> None:
    """Record a gauge metric."""
    if not getattr(FLAGS, "metrics_enabled", True):
        return
    log.debug("METRIC gauge     name=%s value=%s tags=%s", name, value, tags or {})
