"""Orchestration helpers for the note_style AI stage."""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend import config
from backend.ai.manifest import ensure_note_style_section
from backend.ai.note_style.schema import NoteStyleResult, to_json, validate_result
from backend.ai.note_style_stage import (
    build_note_style_pack_for_account,
    discover_note_style_response_accounts,
)
from backend.ai.note_style_logging import log_structured_event
from backend.core.runflow import runflow_barriers_refresh
from backend.runflow.decider import record_stage, reconcile_umbrella_barriers


log = logging.getLogger(__name__)

_DEBOUNCE_MS_ENV = "NOTE_STYLE_DEBOUNCE_MS"
_DEFAULT_DEBOUNCE_MS = 500


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        env_value = os.getenv("RUNS_ROOT")
        return Path(env_value) if env_value else Path("runs")
    return Path(runs_root)


def _debounce_delay_seconds() -> float:
    raw = os.getenv(_DEBOUNCE_MS_ENV)
    if raw is None:
        return _DEFAULT_DEBOUNCE_MS / 1000.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _DEFAULT_DEBOUNCE_MS / 1000.0
    if value <= 0:
        return 0.0
    return value / 1000.0


def prepare_and_send(
    sid: str, *, runs_root: Path | str | None = None
) -> Mapping[str, Any]:
    """Discover, build, and send note_style packs for ``sid``."""

    sid_text = str(sid or "").strip()
    if not sid_text:
        raise ValueError("sid is required")

    if not config.NOTE_STYLE_ENABLED:
        log.info("NOTE_STYLE_DISABLED sid=%s", sid_text)
        return {
            "sid": sid_text,
            "accounts_discovered": 0,
            "packs_built": 0,
            "skipped": 0,
            "processed_accounts": [],
            "statuses": {},
        }

    runs_root_path = _resolve_runs_root(runs_root)
    ensure_note_style_section(sid_text, runs_root=runs_root_path)

    accounts = discover_note_style_response_accounts(
        sid_text, runs_root=runs_root_path
    )
    log.info(
        "NOTE_STYLE_PREPARE sid=%s discovered=%s", sid_text, len(accounts)
    )

    built = 0
    skipped = 0
    scheduled: list[str] = []
    statuses: dict[str, Mapping[str, Any]] = {}

    for account in accounts:
        result = build_note_style_pack_for_account(
            sid_text, account.account_id, runs_root=runs_root_path
        )
        statuses[account.account_id] = dict(result)
        status_text = str(result.get("status") or "").lower()
        if status_text == "completed":
            built += 1
            scheduled.append(account.account_id)
        elif status_text.startswith("skipped"):
            skipped += 1

    if built > 0:
        try:
            schedule_send_for_sid(
                sid_text,
                runs_root=runs_root_path,
                trigger="prepare",
                account_ids=tuple(scheduled),
            )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "NOTE_STYLE_SEND_TASK_SCHEDULE_FAILED sid=%s accounts=%s",
                sid_text,
                scheduled,
                exc_info=True,
            )

    if not accounts:
        try:
            record_stage(
                sid_text,
                "note_style",
                status="success",
                counts={"packs_total": 0},
                empty_ok=True,
                metrics={"packs_total": 0},
                results={"results_total": 0, "completed": 0, "failed": 0},
                runs_root=runs_root_path,
            )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "NOTE_STYLE_PREPARE_STAGE_RECORD_FAILED sid=%s", sid_text, exc_info=True
            )

    if not built:
        try:
            runflow_barriers_refresh(sid_text)
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "NOTE_STYLE_PREPARE_BARRIERS_REFRESH_FAILED sid=%s",
                sid_text,
                exc_info=True,
            )
        try:
            reconcile_umbrella_barriers(sid_text, runs_root=runs_root_path)
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "NOTE_STYLE_PREPARE_BARRIERS_RECONCILE_FAILED sid=%s",
                sid_text,
                exc_info=True,
        )

    log.info(
        "NOTE_STYLE_PREPARE_DONE sid=%s discovered=%s built=%s sent=%s skipped=%s",
        sid_text,
        len(accounts),
        built,
        len(scheduled),
        skipped,
    )

    return {
        "sid": sid_text,
        "accounts_discovered": len(accounts),
        "packs_built": built,
        "skipped": skipped,
        "processed_accounts": list(scheduled),
        "statuses": statuses,
    }


def schedule_send_for_sid(
    sid: str,
    *,
    runs_root: Path | str | None = None,
    trigger: str | None = None,
    account_ids: Sequence[str] | None = None,
) -> None:
    """Schedule the asynchronous send/ingest work for ``sid`` packs."""

    sid_text = str(sid or "").strip()
    if not sid_text:
        return

    if not config.NOTE_STYLE_ENABLED:
        log.info("NOTE_STYLE_DISABLED sid=%s", sid_text)
        return

    if not config.NOTE_STYLE_AUTOSEND:
        log.info("NOTE_STYLE_AUTOSEND_DISABLED sid=%s", sid_text)
        return

    if runs_root is None:
        runs_root_arg: str | None = None
    else:
        try:
            runs_root_arg = os.fspath(runs_root)
        except TypeError:
            runs_root_arg = str(runs_root)

    normalized_accounts: Sequence[str] | None
    if account_ids:
        normalized_accounts = tuple(str(value).strip() for value in account_ids if str(value).strip())
    else:
        normalized_accounts = None

    now_bucket = int(math.floor(time.time()))
    task_id = f"note-style.send:{sid_text}:{now_bucket}"

    log_structured_event(
        "NOTE_STYLE_SEND_TASK_ENQUEUE",
        logger=log,
        sid=sid_text,
        runs_root=runs_root_arg,
        trigger=trigger or "explicit",
        accounts=list(normalized_accounts) if normalized_accounts else None,
    )

    from backend.ai.note_style.tasks import note_style_send_sid_task

    try:
        note_style_send_sid_task.apply_async(
            args=(sid_text,),
            kwargs={"runs_root": runs_root_arg},
            task_id=task_id,
            expires=300,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        if exc.__class__.__name__ == "DuplicateTaskError":
            log_structured_event(
                "NOTE_STYLE_SEND_TASK_ENQUEUE_DUPLICATE",
                logger=log,
                sid=sid_text,
                runs_root=runs_root_arg,
                trigger=trigger or "explicit",
                accounts=list(normalized_accounts) if normalized_accounts else None,
            )
            return
        log.warning(
            "NOTE_STYLE_SEND_TASK_ENQUEUE_FAILED sid=%s", sid_text, exc_info=True
        )
        raise


def schedule_send_for_account(
    sid: str,
    account_id: str,
    *,
    runs_root: Path | str | None = None,
) -> None:
    """Backward-compatible wrapper scheduling send work for ``sid``."""

    sid_text = str(sid or "").strip()
    account_text = str(account_id or "").strip()
    if not sid_text or not account_text:
        return

    if not config.NOTE_STYLE_ENABLED:
        log.info(
            "NOTE_STYLE_DISABLED sid=%s account_id=%s", sid_text, account_text
        )
        return

    if not config.NOTE_STYLE_AUTOSEND:
        log.info("NOTE_STYLE_AUTOSEND_DISABLED sid=%s", sid_text)
        return

    if runs_root is None:
        runs_root_arg: str | None = None
    else:
        try:
            runs_root_arg = os.fspath(runs_root)
        except TypeError:
            runs_root_arg = str(runs_root)

    log.info(
        "NOTE_STYLE_ACCOUNT_TASK_ENQUEUE sid=%s account_id=%s", sid_text, account_text
    )

    schedule_send_for_sid(
        sid_text,
        runs_root=runs_root_arg,
        trigger="account",
        account_ids=(account_text,),
    )


def schedule_prepare_and_send(
    sid: str, *, runs_root: Path | str | None = None
) -> None:
    """Schedule :func:`prepare_and_send` for ``sid`` with debounce."""

    sid_text = str(sid or "").strip()
    if not sid_text:
        return

    if not config.NOTE_STYLE_ENABLED:
        log.info("NOTE_STYLE_DISABLED sid=%s", sid_text)
        return

    if not config.NOTE_STYLE_AUTOSEND:
        log.info("NOTE_STYLE_AUTOSEND_DISABLED sid=%s", sid_text)
        return

    delay = max(_debounce_delay_seconds(), 0.0)
    now_wall = time.time()
    task_id: str | None = None
    expires: float | None = None

    if delay > 0:
        bucket = int(math.floor(now_wall / delay)) if delay else int(now_wall)
        task_id = f"note-style.prepare:{sid_text}:{bucket}"
        expires = delay + 60
    else:
        expires = 60

    runs_root_arg: str | None
    if runs_root is None:
        runs_root_arg = None
    else:
        try:
            runs_root_arg = os.fspath(runs_root)
        except TypeError:
            runs_root_arg = str(runs_root)

    log_structured_event(
        "NOTE_STYLE_TASK_ENQUEUE",
        logger=log,
        sid=sid_text,
        delay_seconds=delay if delay > 0 else 0,
        runs_root=runs_root_arg,
        task_id=task_id,
    )

    from backend.ai.note_style.tasks import note_style_prepare_and_send_task

    apply_kwargs: dict[str, object] = {
        "args": (sid_text,),
        "kwargs": {"runs_root": runs_root_arg},
    }
    if delay > 0:
        apply_kwargs["countdown"] = delay
    if task_id:
        apply_kwargs["task_id"] = task_id
    if expires:
        apply_kwargs["expires"] = expires

    try:
        note_style_prepare_and_send_task.apply_async(**apply_kwargs)
    except Exception as exc:
        if exc.__class__.__name__ == "DuplicateTaskError":
            log_structured_event(
                "NOTE_STYLE_TASK_ENQUEUE_DUPLICATE",
                logger=log,
                sid=sid_text,
                runs_root=runs_root_arg,
                task_id=task_id,
            )
            return
        log_structured_event(
            "NOTE_STYLE_TASK_ENQUEUE_FAILED",
            logger=log,
            sid=sid_text,
            runs_root=runs_root_arg,
            task_id=task_id,
            error=str(exc),
            level=logging.ERROR,
        )
        log.exception("NOTE_STYLE_TASK_ENQUEUE_FAILED sid=%s", sid_text)
        raise


__all__ = [
    "prepare_and_send",
    "schedule_prepare_and_send",
    "schedule_send_for_sid",
    "schedule_send_for_account",
    "NoteStyleResult",
    "validate_result",
    "to_json",
]

