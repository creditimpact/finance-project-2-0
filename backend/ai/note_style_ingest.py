"""Helpers for persisting note_style model outputs."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping

from backend.ai.manifest import update_note_style_stage_status
from backend.ai.note_style.io import note_style_stage_view
from backend.ai.note_style.parse import parse_note_style_response_payload
from backend.ai.note_style_results import (
    complete_note_style_result,
    store_note_style_result,
)
from backend.core.ai.paths import NoteStyleAccountPaths
from backend.note_style.validator import validate_analysis_payload


log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def ingest_note_style_result(
    *,
    sid: str,
    account_id: str,
    runs_root: Path,
    account_paths: NoteStyleAccountPaths,
    pack_payload: Mapping[str, Any],
    response_payload: Any,
) -> Path:
    """Persist the normalized ``response_payload`` for ``account_id``."""

    parsed_response = parse_note_style_response_payload(response_payload)
    response_mode = "unknown"
    if isinstance(response_payload, Mapping):
        mode_value = response_payload.get("mode")
        if isinstance(mode_value, str) and mode_value:
            response_mode = mode_value
    analysis_payload = parsed_response.analysis

    normalized_analysis = validate_analysis_payload(analysis_payload)

    result_payload: MutableMapping[str, Any] = {
        "sid": sid,
        "account_id": str(account_id),
        "analysis": normalized_analysis,
    }

    metrics_payload: MutableMapping[str, Any] | None = None
    note_candidate: Any = None
    if isinstance(pack_payload, Mapping):
        note_candidate = pack_payload.get("note_text")
        if not isinstance(note_candidate, str):
            context = pack_payload.get("context")
            if isinstance(context, Mapping):
                context_note = context.get("note_text")
                if isinstance(context_note, str):
                    note_candidate = context_note
    if isinstance(note_candidate, str):
        metrics_payload = {
            "char_len": len(note_candidate),
            "word_len": len(note_candidate.split()),
        }

    if metrics_payload is not None:
        result_payload["note_metrics"] = metrics_payload

    log.info(
        "NOTE_STYLE_PARSED sid=%s account_id=%s mode=%s source=%s",
        sid,
        account_id,
        response_mode,
        parsed_response.source,
    )

    completed_at = _now_iso()
    result_path = store_note_style_result(
        sid,
        account_id,
        result_payload,
        runs_root=runs_root,
        completed_at=completed_at,
        update_index=False,
    )

    _, totals, _, result_valid, _ = complete_note_style_result(
        sid,
        account_id,
        runs_root=runs_root,
        account_paths=account_paths,
        completed_at=completed_at,
    )

    results_completed = int(totals.get("completed", 0)) if totals else 0
    results_failed = int(totals.get("failed", 0)) if totals else 0
    results_count = results_completed + results_failed

    if result_valid and results_count > 0:
        try:
            view = note_style_stage_view(sid, runs_root=runs_root)
            if view.is_terminal:
                update_note_style_stage_status(
                    sid,
                    runs_root=runs_root,
                    built=view.built_complete,
                    sent=True,
                    failed=view.failed_total > 0,
                    completed_at=completed_at,
                    state=view.state,
                )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "NOTE_STYLE_MANIFEST_STAGE_STATUS_UPDATE_FAILED sid=%s account_id=%s path=%s",
                sid,
                account_id,
                str(result_path),
                exc_info=True,
            )
        else:
            if view.is_terminal:
                log.info(
                    "NOTE_STYLE_MANIFEST_STAGE_STATUS_UPDATED sid=%s account_id=%s results=%d",
                    sid,
                    account_id,
                    results_count,
                )

    return result_path


__all__ = ["ingest_note_style_result"]
