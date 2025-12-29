"""Helpers for synchronizing runflow decisions with the run manifest."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from backend import config
from backend.core.ai.paths import NoteStylePaths
from backend.core.paths import normalize_stage_path, normalize_worker_path
from backend.core.paths.frontend_review import ensure_frontend_review_dirs
from backend.pipeline.runs import RUNS_ROOT_ENV, RunManifest, persist_manifest


log = logging.getLogger(__name__)


_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[/\\]")


# Lazily imported to avoid circular imports during module initialization.
schedule_prepare_and_send = None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _normalize_manifest_path_value(
    value: Path | str | None, *, run_dir: Path
) -> str | None:
    """Return a normalized string representation for manifest paths."""

    if value is None:
        return None

    try:
        text = os.fspath(value)
    except TypeError:
        return None

    sanitized = str(text).strip()
    if not sanitized:
        return None

    sanitized = sanitized.replace("\\", "/")

    if _WINDOWS_DRIVE_PATTERN.match(sanitized):
        sanitized = sanitized[2:]

    run_dir_text = str(run_dir).strip().replace("\\", "/")
    if _WINDOWS_DRIVE_PATTERN.match(run_dir_text):
        run_dir_text = run_dir_text[2:]

    run_dir_lower = run_dir_text.lower()
    sanitized_lower = sanitized.lower()
    if run_dir_lower and sanitized_lower.startswith(run_dir_lower):
        suffix = sanitized[len(run_dir_text) :]
        sanitized = run_dir_text + suffix

    return sanitized


def _is_within_directory(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        return False
    return True


def _canonical_note_style_stage_paths(run_dir: Path) -> dict[str, Path]:
    base_dir = (run_dir / config.NOTE_STYLE_STAGE_DIR).resolve()
    packs_dir = (run_dir / config.NOTE_STYLE_PACKS_DIR).resolve()
    results_dir = (run_dir / config.NOTE_STYLE_RESULTS_DIR).resolve()
    index_file = (base_dir / "index.json").resolve()
    log_file = (base_dir / "logs.txt").resolve()

    return {
        "base_dir": base_dir,
        "packs_dir": packs_dir,
        "results_dir": results_dir,
        "index_file": index_file,
        "log_file": log_file,
    }


def _normalize_note_style_stage_path(
    value: Any,
    *,
    run_dir: Path,
    fallback: Path,
    key: str | None = None,
) -> Path:
    if value is None:
        return fallback

    try:
        text = os.fspath(value)
    except TypeError:
        text = str(value)

    sanitized = str(text).strip()
    if not sanitized:
        return fallback

    sanitized = sanitized.replace("\\", "/")

    try:
        candidate = normalize_worker_path(run_dir, sanitized)
    except ValueError:
        _log_manifest_path_fallback(run_dir, key=key, value=sanitized, reason="normalize")
        return fallback

    if not _is_within_directory(candidate, run_dir):
        _log_manifest_path_fallback(
            run_dir, key=key, value=sanitized, reason="outside_run_dir"
        )
        return fallback

    try:
        resolved = candidate.resolve()
    except OSError:
        resolved = candidate

    if key is not None:
        sanitized_lower = sanitized.lower()
        fallback_lower = str(fallback).strip().replace("\\", "/").lower()
        if resolved == fallback and sanitized_lower != fallback_lower:
            _log_manifest_path_fallback(
                run_dir, key=key, value=sanitized, reason="fallback"
            )

    return resolved


def _log_manifest_path_fallback(
    run_dir: Path, *, key: str | None, value: str, reason: str
) -> None:
    if key is None:
        return
    log.debug(
        "NOTE_STYLE_MANIFEST_PATH_FALLBACK run_dir=%s key=%s value=%s reason=%s",
        run_dir,
        key,
        value,
        reason,
    )


def _extract_stage_value(stage_section: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = stage_section.get(key)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _load_note_style_manifest_stage(run_dir: Path) -> dict[str, Any] | None:
    manifest_path = run_dir / "manifest.json"

    try:
        raw = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.debug(
            "NOTE_STYLE_MANIFEST_STAGE_READ_FAILED run_dir=%s path=%s",
            run_dir,
            manifest_path,
            exc_info=True,
        )
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.debug(
            "NOTE_STYLE_MANIFEST_STAGE_INVALID_JSON run_dir=%s path=%s",
            run_dir,
            manifest_path,
            exc_info=True,
        )
        return None

    if not isinstance(payload, dict):
        return None

    ai_section = payload.get("ai")
    if not isinstance(ai_section, dict):
        return None

    packs_section = ai_section.get("packs")
    if not isinstance(packs_section, dict):
        return None

    note_style_section = packs_section.get("note_style")
    if not isinstance(note_style_section, dict):
        return None

    return note_style_section


def resolve_note_style_stage_paths(
    runs_root: Path | str, sid: str, *, create: bool = False
) -> NoteStylePaths:
    runs_root_path = Path(runs_root).resolve()
    run_dir = (runs_root_path / sid).resolve()

    canonical = _canonical_note_style_stage_paths(run_dir)
    base_dir = canonical["base_dir"]
    packs_dir = canonical["packs_dir"]
    results_dir = canonical["results_dir"]
    index_file = canonical["index_file"]
    log_file = canonical["log_file"]

    if config.NOTE_STYLE_USE_MANIFEST_PATHS:
        stage_section = _load_note_style_manifest_stage(run_dir)
        if stage_section is not None:
            base_value = _extract_stage_value(stage_section, "base", "dir")
            base_dir = _normalize_note_style_stage_path(
                base_value, run_dir=run_dir, fallback=base_dir, key="base"
            )

            if base_dir == canonical["base_dir"]:
                packs_fallback = packs_dir
                results_fallback = results_dir
                index_fallback = index_file
                log_fallback = log_file
            else:
                packs_fallback = (base_dir / "packs").resolve()
                results_fallback = (base_dir / "results").resolve()
                index_fallback = (base_dir / "index.json").resolve()
                log_fallback = (base_dir / "logs.txt").resolve()

            packs_value = _extract_stage_value(stage_section, "packs", "packs_dir")
            results_value = _extract_stage_value(
                stage_section, "results", "results_dir"
            )
            index_value = _extract_stage_value(stage_section, "index")
            logs_value = _extract_stage_value(stage_section, "logs")

            packs_dir = _normalize_note_style_stage_path(
                packs_value,
                run_dir=run_dir,
                fallback=packs_fallback,
                key="packs",
            )
            results_dir = _normalize_note_style_stage_path(
                results_value,
                run_dir=run_dir,
                fallback=results_fallback,
                key="results",
            )
            index_file = _normalize_note_style_stage_path(
                index_value,
                run_dir=run_dir,
                fallback=index_fallback,
                key="index",
            )
            log_file = _normalize_note_style_stage_path(
                logs_value,
                run_dir=run_dir,
                fallback=log_fallback,
                key="logs",
            )

    results_raw_dir = (base_dir / "results_raw").resolve()
    debug_dir = (base_dir / "debug").resolve()

    if create:
        for directory in (base_dir, packs_dir, results_dir, results_raw_dir, debug_dir):
            directory.mkdir(parents=True, exist_ok=True)

    return NoteStylePaths(
        base=base_dir.resolve(),
        packs_dir=packs_dir.resolve(),
        results_dir=results_dir.resolve(),
        results_raw_dir=results_raw_dir,
        debug_dir=debug_dir,
        index_file=index_file.resolve(strict=False),
        log_file=log_file.resolve(strict=False),
    )


def _resolve_manifest(
    sid: str,
    *,
    manifest: Optional[RunManifest] = None,
    runs_root: Optional[Path | str] = None,
) -> RunManifest | None:
    if manifest is not None:
        return manifest

    if runs_root is not None:
        if isinstance(runs_root, Path):
            base = runs_root.resolve()
        else:
            text = str(runs_root or "").strip()
            sanitized = text.replace("\\", "/")
            if len(sanitized) >= 2 and sanitized[1] == ":":
                try:
                    base = normalize_stage_path(Path("/"), sanitized)
                except ValueError:
                    base = Path("runs").resolve()
            else:
                candidate = Path(sanitized)
                if candidate.is_absolute():
                    base = candidate.resolve()
                else:
                    base = (Path.cwd() / candidate).resolve()
        os.environ.setdefault(RUNS_ROOT_ENV, str(base))
        try:
            return RunManifest.for_sid(sid, allow_create=False, runs_root=base)
        except FileNotFoundError:
            log.warning("RUN_DIR_CREATE_BLOCKED sid=%s caller=%s", sid, "runflow._resolve")
            return None

    try:
        return RunManifest.for_sid(sid, allow_create=False)
    except FileNotFoundError:
        log.warning("RUN_DIR_CREATE_BLOCKED sid=%s caller=%s", sid, "runflow._resolve")
        return None


def update_manifest_state(
    sid: str,
    state: str,
    *,
    manifest: Optional[RunManifest] = None,
    runs_root: Optional[Path | str] = None,
) -> RunManifest | None:
    """Update the manifest ``status`` field for ``sid`` to ``state``.

    Parameters
    ----------
    sid:
        The session identifier whose manifest should be updated.
    state:
        The new status string to persist into the manifest.
    manifest:
        Optional pre-loaded manifest instance. When provided, it is updated
        in-place and returned without reloading from disk.
    runs_root:
        Optional runs root override used when ``manifest`` is not supplied.
    """

    target_manifest = _resolve_manifest(
        sid, manifest=manifest, runs_root=runs_root
    )

    if target_manifest is None:
        log.warning(
            "RUN_DIR_CREATE_BLOCKED sid=%s caller=%s", sid, "runflow.update_state"
        )
        return None

    previous_state = str(target_manifest.data.get("run_state") or "")

    target_manifest.data["status"] = str(state)
    target_manifest.data["run_state"] = str(state)
    persist_manifest(target_manifest)

    state_text = str(state)
    if (
        state_text == "AWAITING_CUSTOMER_INPUT"
        and previous_state != "AWAITING_CUSTOMER_INPUT"
    ):
        try:
            base_dir = target_manifest.path.resolve().parent.parent
        except Exception:
            base_dir = None

        effective_runs_root = runs_root if runs_root is not None else base_dir

        if config.NOTE_STYLE_ENABLED:
            global schedule_prepare_and_send
            if schedule_prepare_and_send is None:
                from backend.ai.note_style import schedule_prepare_and_send as _schedule_prepare_and_send

                schedule_prepare_and_send = _schedule_prepare_and_send

            try:
                schedule_prepare_and_send(sid, runs_root=effective_runs_root)
            except Exception:  # pragma: no cover - defensive logging
                log.warning(
                    "NOTE_STYLE_PREPARE_SCHEDULE_STATE_FAILED sid=%s state=%s",
                    sid,
                    state_text,
                    exc_info=True,
                )

    return target_manifest


def update_manifest_frontend(
    sid: str,
    *,
    packs_dir: Optional[Path | str],
    packs_count: int,
    built: bool,
    last_built_at: Optional[str],
    manifest: Optional[RunManifest] = None,
    runs_root: Optional[Path | str] = None,
) -> RunManifest | None:
    target_manifest = _resolve_manifest(
        sid, manifest=manifest, runs_root=runs_root
    )

    if target_manifest is None:
        log.warning(
            "RUN_DIR_CREATE_BLOCKED sid=%s caller=%s", sid, "runflow.update_frontend"
        )
        return None

    run_dir = target_manifest.path.parent
    canonical_paths = ensure_frontend_review_dirs(str(run_dir))

    run_dir_path = run_dir.resolve()

    frontend_base_path = Path(canonical_paths["frontend_base"]).resolve()
    review_dir_path = Path(canonical_paths["review_dir"]).resolve()
    packs_dir_path = Path(canonical_paths["packs_dir"]).resolve()
    responses_dir_path = Path(canonical_paths["responses_dir"]).resolve()
    index_path = Path(canonical_paths["index"]).resolve()
    legacy_index_value = canonical_paths.get("legacy_index")
    legacy_index_path = Path(legacy_index_value).resolve() if legacy_index_value else None

    packs_count_glob = len(list(packs_dir_path.glob("idx-*.json")))
    packs_count_param = int(packs_count or 0)
    packs_count_value = max(packs_count_glob, packs_count_param)

    responses_count = len(list(responses_dir_path.glob("*.json")))
    now_iso = _now_iso()

    last_built_value: str | None
    if built:
        last_built_value = (
            str(last_built_at) if last_built_at else now_iso
        )
    else:
        last_built_value = str(last_built_at) if last_built_at else None

    base_value = _normalize_manifest_path_value(frontend_base_path, run_dir=run_dir_path)
    dir_value = _normalize_manifest_path_value(review_dir_path, run_dir=run_dir_path)
    packs_value = _normalize_manifest_path_value(packs_dir_path, run_dir=run_dir_path)
    results_value = _normalize_manifest_path_value(responses_dir_path, run_dir=run_dir_path)
    index_value = _normalize_manifest_path_value(index_path, run_dir=run_dir_path)
    legacy_index_normalized = _normalize_manifest_path_value(
        legacy_index_path, run_dir=run_dir_path
    )

    target_manifest.data["frontend"] = {
        "base": base_value,
        "dir": dir_value,
        "packs": packs_value,
        "packs_dir": packs_value,
        "results": results_value,
        "results_dir": results_value,
        "index": index_value,
        "legacy_index": legacy_index_normalized,
        "built": bool(built),
        "packs_count": packs_count_value,
        "counts": {
            "packs": packs_count_value,
            "responses": responses_count,
        },
        "last_built_at": last_built_value,
        "last_responses_at": now_iso,
    }

    persist_manifest(target_manifest)
    if built:
        log.info(
            "FRONTEND_BUILT sid=%s packs=%s index=%s",
            sid,
            packs_count_value,
            index_value or str(index_path),
        )
    return target_manifest


def _ensure_stage_status_payload(manifest: RunManifest, stage_key: str) -> dict:
    data = manifest.data
    if not isinstance(data, dict):
        data = {}
        manifest.data = data

    ai_section = data.setdefault("ai", {})
    if not isinstance(ai_section, dict):
        ai_section = {}
        data["ai"] = ai_section

    packs_section = ai_section.setdefault("packs", {})
    if not isinstance(packs_section, dict):
        packs_section = {}
        ai_section["packs"] = packs_section

    stage_section = packs_section.setdefault(stage_key, {})
    if not isinstance(stage_section, dict):
        stage_section = {}
        packs_section[stage_key] = stage_section

    status_payload = stage_section.setdefault("status", {})
    if not isinstance(status_payload, dict):
        status_payload = {}
        stage_section["status"] = status_payload

    status_payload.setdefault("built", False)
    status_payload.setdefault("sent", False)
    status_payload.setdefault("completed_at", None)

    return status_payload


def update_manifest_ai_stage_result(
    sid: str,
    stage: str,
    *,
    manifest: Optional[RunManifest] = None,
    runs_root: Optional[Path | str] = None,
    completed_at: Optional[str] = None,
) -> RunManifest | None:
    """Mark ``stage`` as sent with ``completed_at`` inside the manifest."""

    stage_key = str(stage).strip().lower()
    if not stage_key:
        raise ValueError("stage is required")

    target_manifest = _resolve_manifest(
        sid, manifest=manifest, runs_root=runs_root
    )

    if target_manifest is None:
        log.warning(
            "RUN_DIR_CREATE_BLOCKED sid=%s caller=%s", sid, "runflow.update_ai_stage"
        )
        return None

    status_payload = _ensure_stage_status_payload(target_manifest, stage_key)
    stage_status = target_manifest.ensure_ai_stage_status(stage_key)

    existing_completed = status_payload.get("completed_at")
    if isinstance(existing_completed, str) and existing_completed.strip():
        timestamp = existing_completed.strip()
    else:
        timestamp_candidate = str(completed_at).strip() if completed_at else ""
        timestamp = timestamp_candidate or _now_iso()

    changed = False

    if not bool(status_payload.get("sent")):
        status_payload["sent"] = True
        changed = True
    elif status_payload.get("sent") is not True:
        status_payload["sent"] = True
        changed = True

    if status_payload.get("completed_at") != timestamp:
        status_payload["completed_at"] = timestamp
        changed = True

    if not bool(stage_status.get("sent")):
        stage_status["sent"] = True
        changed = True
    elif stage_status.get("sent") is not True:
        stage_status["sent"] = True
        changed = True

    if stage_status.get("completed_at") != timestamp:
        stage_status["completed_at"] = timestamp
        changed = True

    if changed:
        persist_manifest(target_manifest)

    return target_manifest


def update_note_style_stage_status(
    sid: str,
    *,
    manifest: Optional[RunManifest] = None,
    runs_root: Optional[Path | str] = None,
    built: Optional[bool] = None,
    sent: Optional[bool] = None,
    state: Optional[str] = None,
    completed_at: Optional[str] = None,
) -> RunManifest | None:
    """Update the note_style stage status fields and persist the manifest."""

    target_manifest = _resolve_manifest(
        sid, manifest=manifest, runs_root=runs_root
    )

    if target_manifest is None:
        log.warning(
            "RUN_DIR_CREATE_BLOCKED sid=%s caller=%s", sid, "runflow.update_note_style"
        )
        return None

    status_payload = _ensure_stage_status_payload(target_manifest, "note_style")
    stage_status = target_manifest.ensure_ai_stage_status("note_style")

    sentinel = object()

    def _assign(mapping: dict[str, Any], key: str, value: Any) -> bool:
        current = mapping.get(key, sentinel)
        if current is sentinel or current != value:
            mapping[key] = value
            return True
        return False

    changed = False
    if built is not None:
        built_value = bool(built)
        changed |= _assign(status_payload, "built", built_value)
        if isinstance(stage_status, dict):
            changed |= _assign(stage_status, "built", built_value)
    if sent is not None:
        sent_value = bool(sent)
        changed |= _assign(status_payload, "sent", sent_value)
        if isinstance(stage_status, dict):
            changed |= _assign(stage_status, "sent", sent_value)
    if state is not None:
        normalized_state = str(state).strip() or None
        changed |= _assign(status_payload, "state", normalized_state)
        if isinstance(stage_status, dict):
            changed |= _assign(stage_status, "state", normalized_state)
    if completed_at is not None:
        completed_value = str(completed_at).strip() if completed_at is not None else ""
        if not completed_value:
            completed_value = _now_iso()
        changed |= _assign(status_payload, "completed_at", completed_value)
        if isinstance(stage_status, dict):
            changed |= _assign(stage_status, "completed_at", completed_value)

    if changed:
        persist_manifest(target_manifest)

    return target_manifest


__all__ = [
    "update_manifest_state",
    "update_manifest_frontend",
    "update_manifest_ai_stage_result",
    "update_note_style_stage_status",
    "resolve_note_style_stage_paths",
]

