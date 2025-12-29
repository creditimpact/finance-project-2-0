import json
import logging
from datetime import datetime
from typing import Any, Mapping

logger = logging.getLogger("telemetry")


def emit(event: str, payload: Mapping[str, Any] | None = None) -> None:
    """Emit a structured telemetry event.

    Events are logged as a single JSON line with an ISO timestamp. Failures to
    serialize are logged but otherwise ignored to avoid interrupting the caller.
    """
    record = {"event": event, "ts": datetime.utcnow().isoformat() + "Z"}
    if payload:
        record.update(dict(payload))
    try:
        logger.info(json.dumps(record, sort_keys=True))
    except Exception:
        logger.exception("telemetry_emit_failed")
