"""Helpers to log environment flag snapshots for runflow stages."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from backend.core.paths.frontend_review import get_frontend_review_paths

_LOG = logging.getLogger(__name__)

_MASK_KEYWORDS: tuple[str, ...] = ("KEY", "TOKEN", "SECRET", "PASSWORD")


def _coerce_flag(name: str, *, default: bool) -> tuple[bool, str]:
    raw = os.getenv(name)
    if raw is None:
        return default, "<unset>"

    if isinstance(raw, str):
        raw_value = raw.strip()
    else:
        raw_value = str(raw).strip()

    lowered = raw_value.lower()
    if lowered in {"1", "true", "yes", "on", "y"}:
        return True, raw_value
    if lowered in {"0", "false", "no", "off", "n", ""}:
        return False, raw_value

    try:
        numeric = float(raw_value)
    except (TypeError, ValueError):
        return default, raw_value

    return (numeric != 0.0), raw_value


def _resolve_runs_root() -> Path:
    raw = os.getenv("RUNS_ROOT")
    base = Path(raw).expanduser() if raw else Path("runs")
    try:
        return base.resolve()
    except OSError:
        return base


def _sanitize_env_value(name: str, value: Any) -> str:
    if value is None:
        return "<unset>"

    text = str(value)
    if any(keyword in name for keyword in _MASK_KEYWORDS):
        return "<redacted>"

    trimmed = text.strip()
    return trimmed or "<empty>"


def _collect_flag_snapshot() -> dict[str, dict[str, Any]]:
    flags: dict[str, dict[str, Any]] = {}
    for name, default in (
        ("ENABLE_FRONTEND_PACKS", True),
        ("FRONTEND_TRIGGER_AFTER_CASES", True),
        ("FRONTEND_STAGE_AUTORUN", True),
        ("REVIEW_STAGE_AUTORUN", True),
        ("FRONTEND_REVIEW_CREATE_EMPTY_INDEX", False),
        ("VALIDATION_AUTOSEND_ENABLED", True),
        ("ENABLE_VALIDATION_SENDER", False),
        ("AUTO_VALIDATION_SEND", False),
        ("VALIDATION_SEND_ON_BUILD", False),
    ):
        enabled, raw = _coerce_flag(name, default=default)
        flags[name] = {"enabled": enabled, "raw": raw}
    return flags


def _collect_validation_env() -> dict[str, str]:
    entries: dict[str, str] = {}
    for name, value in sorted(os.environ.items()):
        if not name.startswith("VALIDATION_"):
            continue
        entries[name] = _sanitize_env_value(name, value)
    return entries


def _collect_template_paths(root: Path) -> dict[str, Any]:
    base = root / "<sid>"
    canonical = get_frontend_review_paths(str(base))

    return {
        "runs_root": str(root),
        "templates": {
            "merge_packs": str((base / "ai_packs" / "merge" / "packs").resolve()),
            "merge_results": str((base / "ai_packs" / "merge" / "results").resolve()),
            "validation_packs": str((base / "ai_packs" / "validation" / "packs").resolve()),
            "validation_results": str((base / "ai_packs" / "validation" / "results").resolve()),
            "frontend_dir": str(Path(canonical["frontend_base"]).resolve()),
        },
    }


def _resolve_validation_override(
    env_name: str, default: Path, *, root: Path, sid: str
) -> Path:
    raw = os.getenv(env_name)
    if not raw:
        return default.resolve()

    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        try:
            return candidate.resolve()
        except OSError:
            return candidate

    return (root / sid / candidate).resolve()


def _collect_stage_paths(root: Path, stage: str, sid: str) -> dict[str, Any]:
    payload: dict[str, Any] = {"runs_root": str(root)}
    normalized_stage = stage.strip().lower()
    normalized_sid = str(sid)

    if normalized_stage == "merge":
        base = (root / normalized_sid / "ai_packs" / "merge").resolve()
        payload["merge"] = {
            "base": str(base),
            "packs_dir": str((base / "packs").resolve()),
            "results_dir": str((base / "results").resolve()),
            "log_file": str((base / "logs.txt").resolve()),
            "index_file": str((base / "index.json").resolve()),
        }
    elif normalized_stage == "validation":
        base = (root / normalized_sid / "ai_packs" / "validation").resolve()
        packs_dir = _resolve_validation_override(
            "VALIDATION_PACKS_DIR", base / "packs", root=root, sid=normalized_sid
        )
        results_dir = _resolve_validation_override(
            "VALIDATION_RESULTS_DIR", base / "results", root=root, sid=normalized_sid
        )
        payload["validation"] = {
            "base": str(base),
            "packs_dir": str(packs_dir),
            "results_dir": str(results_dir),
            "log_file": str((base / "logs.txt").resolve()),
            "index_file": str((base / "index.json").resolve()),
        }
    elif normalized_stage == "frontend":
        canonical = get_frontend_review_paths(str(root / normalized_sid))
        review_base = Path(canonical["review_dir"])
        packs_dir = Path(canonical["packs_dir"])
        responses_dir = Path(canonical["responses_dir"])
        index_path = Path(canonical["index"])
        legacy_index = Path(canonical.get("legacy_index", canonical["index"]))
        base = review_base.resolve()
        payload["frontend"] = {
            "base": str(base),
            "packs_dir": str(packs_dir.resolve()),
            "responses_dir": str(responses_dir.resolve()),
            "accounts_dir": str(packs_dir.resolve()),
            "index_file": str(index_path.resolve()),
            "legacy_index_file": str(legacy_index.resolve()),
        }
    else:  # pragma: no cover - defensive logging
        payload["stage"] = normalized_stage

    return payload


def collect_process_snapshot(stage: str | None = None, sid: str | None = None) -> dict[str, Any]:
    """Return a structured snapshot of flag and path state."""

    runs_root = _resolve_runs_root()
    snapshot: dict[str, Any] = {
        "runs_root": str(runs_root),
        "flags": _collect_flag_snapshot(),
        "validation_env": _collect_validation_env(),
    }

    if stage is not None:
        snapshot["stage"] = stage
    if sid is not None:
        snapshot["sid"] = str(sid)

    if stage is not None and sid is not None:
        snapshot["paths"] = _collect_stage_paths(runs_root, stage, sid)
    else:
        snapshot["paths"] = _collect_template_paths(runs_root)

    return snapshot


def log_worker_env_snapshot(context: str = "celery_worker") -> None:
    """Emit a process-scope snapshot for Celery worker startup."""

    snapshot = collect_process_snapshot()
    _LOG.info(
        "CELERY_ENV_SNAPSHOT context=%s data=%s",
        context,
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True),
    )


def log_stage_env_snapshot(stage: str, *, sid: str) -> None:
    """Emit the environment snapshot for a specific runflow stage."""

    snapshot = collect_process_snapshot(stage=stage, sid=sid)
    _LOG.info(
        "RUNFLOW_STAGE_ENV stage=%s sid=%s data=%s",
        stage,
        sid,
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True),
    )


__all__ = [
    "collect_process_snapshot",
    "log_stage_env_snapshot",
    "log_worker_env_snapshot",
]
