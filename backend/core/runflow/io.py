"""Helper utilities for emitting runflow stage lifecycle records."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Mapping, MutableMapping, Optional

from backend.core.runflow import runflow_end_stage, runflow_start_stage
from backend.core.runflow.env_snapshot import log_stage_env_snapshot


def _normalize_summary(summary: Optional[Mapping[str, object]]) -> dict[str, object]:
    if not summary:
        return {}
    normalized: dict[str, object] = {}
    for key, value in summary.items():
        normalized[str(key)] = value
    return normalized


def _short_message(message: str, *, limit: int = 200) -> str:
    compact = " ".join(str(message).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "\u2026"


def _traceback_tail(
    payload: str,
    *,
    limit: int = 500,
    max_lines: int = 30,
) -> str:
    text = str(payload or "").strip()
    if not text:
        return ""

    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    trimmed = "\n".join(lines)

    if len(trimmed) <= limit:
        return trimmed
    return trimmed[-limit:]


def format_exception_tail(
    exc: BaseException,
    *,
    limit: int = 500,
    max_lines: int = 30,
) -> str:
    """Return a compact traceback tail for ``exc`` suitable for runflow."""

    formatted = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    return _traceback_tail(formatted, limit=limit, max_lines=max_lines)


def exception_hint(exc: BaseException) -> Optional[str]:
    """Return a concise hint for ``exc`` when the cause is obvious."""

    if isinstance(exc, FileNotFoundError):
        filename = getattr(exc, "filename", None)
        if not filename and exc.args:
            # Attempt to recover from default strerror message payload.
            message = str(exc)
            if "bureaus.json" in message:
                filename = "bureaus.json"
        if filename:
            name = Path(str(filename)).name or str(filename)
            return f"missing {name}"
        return "missing file"

    if isinstance(exc, PermissionError):
        filename = getattr(exc, "filename", None)
        if filename:
            name = Path(str(filename)).name or str(filename)
            return f"permission denied for {name}"
        return "permission denied"

    return None


def compose_hint(default: Optional[str], exc: Optional[BaseException]) -> Optional[str]:
    """Combine ``default`` and any obvious hint derived from ``exc``."""

    default_hint = (default or "").strip()
    specific = exception_hint(exc) if exc is not None else None

    if specific and default_hint:
        return f"{default_hint}: {specific}"

    if specific:
        return specific

    return default_hint or None


def runflow_stage_start(stage: str, *, sid: str, substage: str = "default") -> None:
    """Mark ``stage`` as started for ``sid`` in runflow outputs."""

    extra: Optional[MutableMapping[str, object]]
    substage_name = substage.strip()
    if substage_name == "default":
        log_stage_env_snapshot(stage, sid=sid)
    if substage_name:
        extra = {"substage": substage_name}
    else:
        extra = None
    runflow_start_stage(sid, stage, extra=extra)


def runflow_stage_end(
    stage: str,
    *,
    sid: str,
    status: str = "success",
    summary: Optional[Mapping[str, object]] = None,
    empty_ok: Optional[bool] = None,
) -> None:
    """Finalize ``stage`` for ``sid`` with the provided ``summary``."""

    summary_payload = _normalize_summary(summary)
    if empty_ok:
        summary_payload.setdefault("empty_ok", True)

    stage_status = None
    if empty_ok and status != "error":
        stage_status = "empty"

    runflow_end_stage(
        sid,
        stage,
        status=status,
        summary=summary_payload or None,
        stage_status=stage_status,
        empty_ok=bool(empty_ok),
    )


def runflow_stage_error(
    stage: str,
    *,
    sid: str,
    error_type: str,
    message: str,
    traceback_tail: str,
    hint: Optional[str] = None,
    summary: Optional[Mapping[str, object]] = None,
) -> None:
    """Record an error outcome for ``stage`` on ``sid``."""

    error_payload: dict[str, object] = {
        "type": str(error_type or "Error"),
        "message": _short_message(message),
        "traceback_tail": _traceback_tail(traceback_tail),
    }
    if hint:
        error_payload["hint"] = hint

    summary_payload = _normalize_summary(summary)
    summary_payload["error"] = error_payload

    runflow_stage_end(
        stage,
        sid=sid,
        status="error",
        summary=summary_payload,
        empty_ok=False,
    )


__all__ = [
    "runflow_stage_start",
    "runflow_stage_end",
    "runflow_stage_error",
    "format_exception_tail",
    "exception_hint",
    "compose_hint",
]
