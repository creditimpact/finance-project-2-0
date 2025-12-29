from __future__ import annotations

import logging
import os
from typing import Any

from backend import config
from backend.api.tasks import app as celery_app
from backend.ai.merge import sender

logger = logging.getLogger(__name__)

_RATE_LIMIT = config.MERGE_CELERY_RATE_LIMIT or os.getenv("MERGE_CELERY_RATE_LIMIT")


@celery_app.task(
    name="backend.ai.merge.tasks.send_merge_packs",
    bind=True,
    rate_limit=_RATE_LIMIT,
)
def send_merge_packs(self, sid: str, runs_root: str | None = None, reason: str | None = None) -> dict[str, Any]:
    task_id = getattr(self.request, "id", None)
    normalized_reason = reason or "manual"
    result = sender.send_merge_packs(
        sid,
        runs_root=runs_root,
        task_id=task_id,
        reason=normalized_reason,
    )
    payload = {
        "sid": result.sid,
        "packs_dir": str(result.packs_dir),
        "index_path": str(result.index_path),
        "glob": result.glob,
        "discovered": result.discovered,
        "sent": result.sent,
        "task_id": result.task_id,
        "reason": result.reason,
    }
    logger.info("MERGE_AUTOSEND_TASK_COMPLETED %s", payload)
    return payload


__all__ = ["send_merge_packs"]
