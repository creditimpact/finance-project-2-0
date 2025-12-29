import logging
from typing import Any

logger = logging.getLogger(__name__)


def emit_metric(name: str, value: float, **tags: Any) -> None:
    """Best-effort metric emitter that logs the metric info."""
    try:
        logger.info("metric %s %s %s", name, value, tags)
    except Exception:  # pragma: no cover - best effort
        try:
            logger.exception("emit_metric_failed")
        except Exception:
            pass
