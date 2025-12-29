"""Helpers for writing frontend review responses."""

from __future__ import annotations

import logging
import os

from backend.ai.note_style.tasks import note_style_prepare_and_send_task


log = logging.getLogger(__name__)


def _coerce_min_bytes(value: str | None, default: int) -> int:
    try:
        coerced = int(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return coerced if coerced >= 0 else 0


def maybe_trigger_note_style_on_response_write(sid: str) -> None:
    """Trigger the note style pipeline after a response write if enabled."""

    if os.getenv("NOTE_STYLE_SEND_ON_RESPONSE_WRITE", "0") != "1":
        log.info("NOTE_STYLE_AUTO: skip enqueue sid=%s reason=env_disabled", sid)
        return

    runs_root = os.getenv("RUNS_ROOT", "runs")
    index_path = os.path.join(runs_root, sid, "frontend", "review", "index.json")
    min_bytes = _coerce_min_bytes(os.getenv("NOTE_STYLE_INDEX_MIN_BYTES"), 20)

    try:
        size = os.path.getsize(index_path)
    except OSError:
        log.info("NOTE_STYLE_AUTO: skip enqueue sid=%s reason=review_index_missing", sid)
        return

    if size < min_bytes:
        log.info("NOTE_STYLE_AUTO: skip enqueue sid=%s reason=review_index_missing", sid)
        return

    note_style_prepare_and_send_task.delay(sid)
    log.info("NOTE_STYLE_AUTO: queued_prepare_and_send sid=%s", sid)


def write_client_response(sid: str, account_id: str, payload: dict | None = None) -> None:
    """Log a client response write and trigger follow-on tasks."""

    _ = payload  # Preserve argument for compatibility with existing callers.
    log.info("REVIEW_RESPONSES: wrote_response sid=%s account_id=%s", sid, account_id)
    maybe_trigger_note_style_on_response_write(sid)
