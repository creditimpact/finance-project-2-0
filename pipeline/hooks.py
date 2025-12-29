"""Pipeline lifecycle hooks for cross-stage orchestration."""

from __future__ import annotations

import logging
import os


logger = logging.getLogger(__name__)


def on_cases_built(sid: str) -> bool:
    """Trigger frontend pack generation when cases are materialised."""

    if os.getenv("FRONTEND_TRIGGER_AFTER_CASES", "1") != "1":
        logger.info("REVIEW_TRIGGER: skip_enqueue sid=%s reason=env_disabled", sid)
        return False

    from backend.api.tasks import enqueue_generate_frontend_packs

    enqueue_generate_frontend_packs(sid)
    return True
