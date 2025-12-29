"""Helpers for coordinating umbrella barrier side-effects."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Mapping, TYPE_CHECKING

from backend import config
from backend.pipeline.runs import RunManifest
from backend.runflow.counters import note_style_stage_counts
from backend.ai.note_style_logging import log_note_style_decision

log = logging.getLogger(__name__)


if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from backend.ai.note_style.io import NoteStyleStageView

def _collect_note_style_metrics(
    run_dir_path: Path,
) -> tuple[tuple[int | None, int | None], "NoteStyleStageView" | None]:
    from backend.ai.note_style.io import note_style_stage_view

    try:
        view = note_style_stage_view(run_dir_path.name, runs_root=run_dir_path.parent)
    except Exception:  # pragma: no cover - defensive logging
        log.debug(
            "NOTE_STYLE_AUTOSEND_METRICS_FAILED path=%s",
            run_dir_path,
            exc_info=True,
        )
        return ((None, None), None)

    packs_total = view.total_expected
    if packs_total == 0:
        return ((0, 0), view)

    terminal_total = view.completed_total + view.failed_total
    if terminal_total == 0:
        terminal_value: int | None = None
    else:
        terminal_value = terminal_total

    return ((packs_total, terminal_value), view)


def _log_autosend_decision(
    *,
    sid: str,
    reason: str,
    sent: bool | None,
    built: int | None,
    terminal: int | None,
    view: "NoteStyleStageView" | None = None,
    level: int = logging.INFO,
    error_code: str | None = None,
    error_type: str | None = None,
) -> None:
    sent_text = "unknown" if sent is None else ("true" if sent else "false")
    built_text = "unknown" if built is None else str(built)
    terminal_text = "unknown" if terminal is None else str(terminal)

    message = (
        "NOTE_STYLE_AUTOSEND_DECISION sid=%s sent=%s built=%s terminal=%s reason=%s"
        % (sid, sent_text, built_text, terminal_text, reason)
    )
    if error_code:
        message += f" error.code={error_code}"
    if error_type:
        message += f" error.type={error_type}"
    log.log(level, message)

    if view is not None:
        decided_status = view.state
        packs_expected = view.total_expected
        packs_built = view.built_total
        packs_completed = view.completed_total
        packs_failed = view.failed_total
    else:
        decided_status = None
        packs_expected = built if built is not None else 0
        packs_built = built if built is not None else 0
        packs_completed = terminal if terminal is not None else 0
        packs_failed = 0

    log_note_style_decision(
        "NOTE_STYLE_AUTOSEND_DECISION",
        logger=log,
        level=level,
        sid=sid,
        reason=reason,
        decided_status=decided_status,
        view=view,
        packs_expected=packs_expected,
        packs_built=packs_built,
        packs_completed=packs_completed,
        packs_failed=packs_failed,
        sent=sent,
        error_code=error_code,
        error_type=error_type,
    )


def _stage_completed(status: Mapping[str, object] | None) -> bool:
    if not isinstance(status, Mapping):
        return False

    if bool(status.get("failed")):
        return True

    if bool(status.get("sent")):
        return True

    completed_at = status.get("completed_at")
    if isinstance(completed_at, str) and completed_at.strip():
        return True

    return False


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default

    lowered = raw.strip().lower()
    if lowered in {"", "0", "false", "no", "off"}:
        return False

    return True


def _normalize_runs_root_arg(runs_root: Path | None) -> str | None:
    if runs_root is None:
        return None
    try:
        return os.fspath(runs_root)
    except TypeError:
        return str(runs_root)


def schedule_merge_autosend(sid: str, *, run_dir: Path | str) -> None:
    """Schedule merge autosend when stage automation is enabled."""

    sid_text = str(sid or "").strip()
    if not sid_text:
        log.info("MERGE_AUTOSEND_STAGE_SKIP sid=%s reason=invalid_sid", sid)
        return

    run_dir_path = Path(run_dir)
    if not _env_flag_enabled("MERGE_AUTOSEND", default=config.MERGE_AUTOSEND):
        log.debug(
            "MERGE_AUTOSEND_STAGE_SKIP sid=%s reason=autosend_disabled", sid_text
        )
        return

    if not _env_flag_enabled(
        "MERGE_STAGE_AUTORUN", default=config.MERGE_STAGE_AUTORUN
    ):
        log.debug(
            "MERGE_AUTOSEND_STAGE_SKIP sid=%s reason=stage_autorun_disabled",
            sid_text,
        )
        return

    try:
        from backend.ai.merge.sender import schedule_stage_autosend
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "MERGE_AUTOSEND_STAGE_IMPORT_FAILED sid=%s", sid_text, exc_info=True
        )
        return

    try:
        schedule_stage_autosend(sid_text, run_dir=run_dir_path)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "MERGE_AUTOSEND_STAGE_SCHEDULE_FAILED sid=%s", sid_text, exc_info=True
        )


def schedule_note_style_after_validation(
    sid: str,
    *,
    run_dir: Path | str,
) -> None:
    """Schedule note_style autosend after validation completes for ``sid``."""

    sid_text = str(sid or "").strip()
    if not sid_text:
        _log_autosend_decision(
            sid="<missing>",
            reason="invalid_sid",
            sent=None,
            built=None,
            terminal=None,
        )
        return

    run_dir_path = Path(run_dir)
    runs_root_path = run_dir_path.parent

    metrics_cache: tuple[int | None, int | None] | None = None
    view_cache: "NoteStyleStageView" | None = None

    def _metrics() -> tuple[int | None, int | None]:
        nonlocal metrics_cache, view_cache
        if metrics_cache is None:
            metrics_cache, view_cache = _collect_note_style_metrics(run_dir_path)
        return metrics_cache

    def _view() -> "NoteStyleStageView" | None:
        _metrics()
        return view_cache

    if not config.NOTE_STYLE_ENABLED:
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="disabled_feature",
            sent=None,
            built=built_total,
            terminal=terminal_total,
            view=_view(),
        )
        return

    if not _env_flag_enabled("NOTE_STYLE_AUTOSEND", default=True):
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="disabled_env",
            sent=None,
            built=built_total,
            terminal=terminal_total,
            view=_view(),
        )
        return

    if not _env_flag_enabled("NOTE_STYLE_STAGE_AUTORUN", default=True):
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="stage_autorun_disabled",
            sent=None,
            built=built_total,
            terminal=terminal_total,
            view=_view(),
        )
        return

    if not _env_flag_enabled("NOTE_STYLE_SEND_ON_RESPONSE_WRITE", default=True):
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="send_on_write_disabled",
            sent=None,
            built=built_total,
            terminal=terminal_total,
            view=_view(),
        )
        return

    try:
        manifest = RunManifest.load_or_create(run_dir_path / "manifest.json", sid_text)
    except Exception as exc:  # pragma: no cover - defensive logging
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="manifest_error",
            sent=None,
            built=built_total,
            terminal=terminal_total,
            level=logging.ERROR,
            error_code="manifest_load",
            error_type=type(exc).__name__,
            view=_view(),
        )
        log.debug(
            "NOTE_STYLE_AUTOSEND_MANIFEST_LOAD_FAILED sid=%s path=%s",
            sid_text,
            run_dir_path,
            exc_info=True,
        )
        return

    from backend.ai.note_style.io import note_style_stage_view

    validation_status = manifest.get_ai_stage_status("validation")
    note_style_status = manifest.get_ai_stage_status("note_style")
    view = note_style_stage_view(sid_text, runs_root=runs_root_path)
    view_cache = view
    metrics_cache = (
        view.total_expected,
        (view.completed_total + view.failed_total)
        if (view.completed_total or view.failed_total)
        else (0 if view.total_expected == 0 else None),
    )

    sent_flag = bool(note_style_status.get("sent")) or view.is_terminal

    if not _stage_completed(validation_status):
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="validation_pending",
            sent=sent_flag,
            built=built_total,
            terminal=terminal_total,
            view=view,
        )
        return

    if not view.has_expected:
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="empty",
            sent=sent_flag,
            built=built_total,
            terminal=terminal_total,
            view=view,
        )

    pending_total = len(view.pending_results)
    ready_total = len(view.ready_to_send)

    if view.has_expected and (pending_total == 0 or view.is_terminal):
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="already_complete",
            sent=sent_flag,
            built=built_total,
            terminal=terminal_total,
            view=view,
        )
        return

    if view.has_expected and ready_total == 0:
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="packs_not_ready",
            sent=sent_flag,
            built=built_total,
            terminal=terminal_total,
            view=view,
        )
        return

    try:
        from backend.runflow.manifest import resolve_note_style_stage_paths

        paths = resolve_note_style_stage_paths(runs_root_path, sid_text, create=False)
    except Exception as exc:  # pragma: no cover - defensive logging
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="path_error",
            sent=sent_flag,
            built=built_total,
            terminal=terminal_total,
            level=logging.ERROR,
            error_code="path_resolve",
            error_type=type(exc).__name__,
            view=view,
        )
        log.debug(
            "NOTE_STYLE_STAGE_PATH_RESOLVE_FAILED sid=%s runs_root=%s",
            sid_text,
            runs_root_path,
            exc_info=True,
        )
        return

    runs_root_arg = _normalize_runs_root_arg(runs_root_path)

    try:
        from backend.ai.note_style.tasks import note_style_prepare_and_send_task
    except Exception as exc:  # pragma: no cover - defensive logging
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="task_import_error",
            sent=sent_flag,
            built=built_total,
            terminal=terminal_total,
            level=logging.ERROR,
            error_code="task_import",
            error_type=type(exc).__name__,
            view=view,
        )
        log.warning(
            "NOTE_STYLE_AUTOSEND_IMPORT_FAILED sid=%s", sid_text, exc_info=True
        )
        return

    built_total, terminal_total = _metrics()

    log.info(
        "NOTE_STYLE_AUTOSEND_READY sid=%s packs=%s", sid_text, pending_total
    )

    try:
        if runs_root_arg is None:
            note_style_prepare_and_send_task.delay(sid_text)
        else:
            note_style_prepare_and_send_task.delay(
                sid_text, runs_root=runs_root_arg
            )
    except Exception as exc:  # pragma: no cover - defensive logging
        built_total, terminal_total = _metrics()
        _log_autosend_decision(
            sid=sid_text,
            reason="enqueue_failed",
            sent=sent_flag,
            built=built_total,
            terminal=terminal_total,
            level=logging.ERROR,
            error_code="schedule",
            error_type=type(exc).__name__,
            view=view,
        )
        log.warning(
            "NOTE_STYLE_AUTOSEND_SCHEDULE_FAILED sid=%s packs=%s",
            sid_text,
            pending_total,
            exc_info=True,
        )
        return

    built_total, terminal_total = _metrics()
    _log_autosend_decision(
        sid=sid_text,
        reason="enqueued",
        sent=sent_flag,
        built=built_total,
        terminal=terminal_total,
        view=view,
    )
    log.info(
        "NOTE_STYLE_AUTOSEND_AFTER_VALIDATION sid=%s packs=%s",
        sid_text,
        pending_total,
    )


__all__ = ["schedule_merge_autosend", "schedule_note_style_after_validation"]
