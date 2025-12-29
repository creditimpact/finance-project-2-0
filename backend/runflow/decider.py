"""Simple runflow state machine for deciding the next pipeline action."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Literal, Mapping, Optional, Sequence, Tuple

from backend import config
from backend.core.ai.paths import (
    ensure_merge_paths,
    ensure_note_style_paths,
    merge_result_glob_pattern,
)
from backend.core.io.json_io import _atomic_write_json
from backend.core.runflow import (
    _apply_umbrella_barriers,
    runflow_decide_step,
    runflow_end_stage,
    runflow_refresh_umbrella_barriers,
    runflow_step,
)
from backend.pipeline.runs import RunManifest
from backend.runflow.counters import (
    _has_review_attachments as _frontend_has_review_attachments,
    frontend_answers_counters as _frontend_answers_counters,
    stage_counts as _stage_counts_from_disk,
)
from backend.validation.index_schema import load_validation_index

from backend.frontend.packs.config import load_frontend_stage_config

StageStatus = Literal[
    "success",
    "error",
    "built",
    "published",
    "in_progress",
    "empty",
    "pending",
]
RunState = Literal[
    "INIT",
    "VALIDATING",
    "AWAITING_CUSTOMER_INPUT",
    "COMPLETE_NO_ACTION",
    "ERROR",
]
StageName = Literal["merge", "validation", "frontend", "note_style"]


log = logging.getLogger(__name__)

_RUNFLOW_FILENAME = "runflow.json"
def _log_note_style_decision(*args: Any, **kwargs: Any) -> None:
    from backend.ai.note_style_logging import log_note_style_decision as _log

    _log(*args, **kwargs)


_STATUS_NORMALIZATION: Dict[str, str] = {
    "completed": "completed",
    "complete": "completed",
    "done": "completed",
    "success": "completed",
    "succeeded": "completed",
    "ok": "completed",
    "finished": "completed",
    "built": "completed",
    "published": "completed",
    "sent": "completed",  # Validation V2: sent = results available = completed
    "failed": "failed",
    "failure": "failed",
    "error": "failed",
    "errored": "failed",
    "aborted": "failed",
    "rejected": "failed",
    "skipped": "skipped",
    "skipped_low_signal": "skipped",
}

_UNKNOWN_STATUS_WARNINGS: set[tuple[str, str]] = set()


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _validation_v2_mode_enabled() -> bool:
    """Return True when Validation V2 orchestrator mode is active.

    Gate legacy validation writers from mutating stages.validation when this is True.
    Signals:
      - VALIDATION_ORCHESTRATOR_MODE (primary)
      - VALIDATION_AUTOSEND_ENABLED/VALIDATION_AUTOSEND (legacy aliases)
    """
    if _env_enabled("VALIDATION_ORCHESTRATOR_MODE", False):
        return True
    # Accept either spelling for autosend feature toggle
    if _env_enabled("VALIDATION_AUTOSEND_ENABLED", False):
        return True
    if _env_enabled("VALIDATION_AUTOSEND", False):
        return True
    return False


def _env_non_negative_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except Exception:
        return default
    return value if value >= 0 else default


_MERGE_SKIP_COUNTS_ENABLED = _env_enabled("MERGE_SKIP_COUNTS_ENABLED", True)
_MERGE_ZERO_PACKS_SIGNAL_ENABLED = _env_enabled("MERGE_ZERO_PACKS_SIGNAL", True)
_RUNFLOW_EMIT_ZERO_PACKS_STEP = _env_enabled("RUNFLOW_EMIT_ZERO_PACKS_STEP", True)
_RUNFLOW_MERGE_ZERO_PACKS_FASTPATH = _env_enabled("RUNFLOW_MERGE_ZERO_PACKS_FASTPATH", True)
_UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG = _env_enabled(
    "UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG", True
)
_VALIDATION_INDEX_MIN_BYTES = _env_non_negative_int("VALIDATION_INDEX_MIN_BYTES", 20)
_VALIDATION_FASTPATH_WATCHDOG_LOCK_SECONDS = _env_non_negative_int(
    "VALIDATION_FASTPATH_WATCHDOG_LOCK_SECONDS", 90
)
_VALIDATION_FASTPATH_WATCHDOG_EVENT_SECONDS = _env_non_negative_int(
    "VALIDATION_FASTPATH_WATCHDOG_EVENT_SECONDS", 180
)


def _umbrella_barriers_enabled() -> bool:
    return _env_enabled("UMBRELLA_BARRIERS_ENABLED", True)


def _strict_validation_enabled() -> bool:
    return _env_enabled("UMBRELLA_BARRIERS_STRICT_VALIDATION", True)


def _review_explanation_required() -> bool:
    return _env_enabled("UMBRELLA_BARRIERS_REVIEW_REQUIRE_EXPLANATION", True)


def _review_attachment_required() -> bool:
    return _env_enabled("UMBRELLA_BARRIERS_REVIEW_REQUIRE_FILE", False)


def _validation_autosend_enabled() -> bool:
    return _env_enabled("VALIDATION_AUTOSEND", True)


def _merge_required() -> bool:
    return _env_enabled("MERGE_REQUIRED", True)


def _umbrella_require_merge() -> bool:
    return _env_enabled("UMBRELLA_REQUIRE_MERGE", True)


def _umbrella_require_style() -> bool:
    return _env_enabled("UMBRELLA_REQUIRE_STYLE", False)


def _barrier_event_logging_enabled() -> bool:
    return _env_enabled("UMBRELLA_BARRIERS_LOG", True)


def _runflow_events_enabled() -> bool:
    return _env_enabled("RUNFLOW_EVENTS", False)


def _document_verifier_enabled() -> bool:
    return _env_enabled("DOCUMENT_VERIFIER_ENABLED", False)


def _default_runs_root() -> Path:
    root_env = os.getenv("RUNS_ROOT")
    return Path(root_env) if root_env else Path("runs")


def _now_iso() -> str:
    """Return an ISO-8601 timestamp in UTC with second precision."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _resolve_runs_root(runs_root: Optional[str | Path]) -> Path:
    if runs_root is None:
        return _default_runs_root()
    return Path(runs_root)


def _runflow_path(sid: str, runs_root: Optional[str | Path]) -> Path:
    base = _resolve_runs_root(runs_root) / sid
    base.mkdir(parents=True, exist_ok=True)
    return base / _RUNFLOW_FILENAME


def _resolve_merge_index_path(
    sid: str,
    run_dir: Path,
    manifest_payload: Mapping[str, Any] | None = None,
) -> Path:
    candidate: Optional[str] = None

    if isinstance(manifest_payload, Mapping):
        ai_payload = manifest_payload.get("ai")
        if isinstance(ai_payload, Mapping):
            packs_payload = ai_payload.get("packs")
            if isinstance(packs_payload, Mapping):
                manifest_index = packs_payload.get("index")
                if isinstance(manifest_index, str) and manifest_index.strip():
                    candidate = manifest_index.strip()

    if not candidate:
        env_override = os.getenv("MERGE_INDEX_PATH")
        if isinstance(env_override, str) and env_override.strip():
            candidate = env_override.strip()

    if not candidate:
        candidate = str(config.MERGE_INDEX_PATH)

    index_path = Path(candidate)
    if not index_path.is_absolute():
        index_path = (run_dir / index_path).resolve()

    return index_path


def _validation_fastpath_lock_path(run_dir: Path) -> Path:
    return run_dir / ".locks" / "validation_fastpath.lock"


def _fastpath_has_lock(run_dir: Path) -> bool:
    lock_path = _validation_fastpath_lock_path(run_dir)
    try:
        return lock_path.exists()
    except OSError:
        log.debug(
            "VALIDATION_FASTPATH_LOCK_CHECK_FAILED sid=%s path=%s",
            _run_dir_sid(run_dir),
            lock_path,
            exc_info=True,
        )
        return False


def _acquire_validation_fastpath_lock(run_dir: Path) -> Path | None:
    lock_path = _validation_fastpath_lock_path(run_dir)
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    except OSError:
        log.warning(
            "VALIDATION_FASTPATH_LOCK_WRITE_FAILED sid=%s path=%s",
            _run_dir_sid(run_dir),
            lock_path,
            exc_info=True,
        )
        return None

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(_now_iso())
    except OSError:
        log.warning(
            "VALIDATION_FASTPATH_LOCK_WRITE_ABORT sid=%s path=%s",
            _run_dir_sid(run_dir),
            lock_path,
            exc_info=True,
        )
        try:
            lock_path.unlink()
        except OSError:
            log.debug(
                "VALIDATION_FASTPATH_LOCK_CLEANUP_FAILED sid=%s path=%s",
                _run_dir_sid(run_dir),
                lock_path,
                exc_info=True,
            )
        return None

    return lock_path


def release_validation_fastpath_lock(run_dir: Path) -> None:
    lock_path = _validation_fastpath_lock_path(run_dir)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return
    except OSError:
        log.debug(
            "VALIDATION_FASTPATH_LOCK_RELEASE_FAILED sid=%s path=%s",
            _run_dir_sid(run_dir),
            lock_path,
            exc_info=True,
        )


def _enqueue_validation_fastpath(
    sid: str,
    run_dir: Path,
    *,
    merge_zero_packs: bool,
    payload: Mapping[str, Any] | None = None,
) -> bool:
    if not merge_zero_packs:
        return False

    if not _validation_autosend_enabled():
        return False

    lock_path = _acquire_validation_fastpath_lock(run_dir)
    if lock_path is None:
        if _fastpath_has_lock(run_dir):
            return True
        return False

    try:
        from celery import chain
        from backend.pipeline import auto_ai_tasks
    except Exception:
        log.warning(
            "VALIDATION_FASTPATH_IMPORT_FAILED sid=%s",
            sid,
            exc_info=True,
        )
        try:
            lock_path.unlink()
        except OSError:
            log.debug(
                "VALIDATION_FASTPATH_LOCK_RELEASE_FAILED sid=%s path=%s",
                sid,
                lock_path,
                exc_info=True,
            )
        return False

    initial_payload: dict[str, Any] = {
        "sid": sid,
        "runs_root": str(run_dir.parent),
        "merge_zero_packs": bool(merge_zero_packs),
    }
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            if isinstance(key, str):
                initial_payload[key] = value

    try:
        workflow = chain(
            auto_ai_tasks.validation_build_packs.s(initial_payload),
            # Send must precede compact so results are emitted before pruning.
            auto_ai_tasks.validation_send.s(),
            auto_ai_tasks.validation_compact.s(),
        )
        workflow.apply_async(queue="validation")
    except Exception:
        log.warning(
            "VALIDATION_FASTPATH_QUEUE_FAILED sid=%s",
            sid,
            exc_info=True,
        )
        try:
            lock_path.unlink()
        except OSError:
            log.debug(
                "VALIDATION_FASTPATH_LOCK_RELEASE_FAILED sid=%s path=%s",
                sid,
                lock_path,
                exc_info=True,
            )
        return False

    runflow_step(
        sid,
        "validation",
        "fastpath_send",
        status="queued",
        out={"merge_zero_packs": True},
    )
    log.info("VALIDATION_FASTPATH_ENQUEUED sid=%s", sid)
    return True

def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_boolish(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _validation_index_ready(run_dir: Path) -> bool:
    index_path = _resolve_validation_index_path(run_dir)
    try:
        if not index_path.is_file():
            return False
        return index_path.stat().st_size >= _VALIDATION_INDEX_MIN_BYTES
    except OSError:
        log.debug(
            "VALIDATION_INDEX_STAT_FAILED sid=%s path=%s",
            _run_dir_sid(run_dir),
            index_path,
            exc_info=True,
        )
        return False


def _runflow_events_path(run_dir: Path) -> Path:
    return run_dir / "runflow_events.jsonl"


def _last_validation_sender_event_timestamp(run_dir: Path, event_name: str) -> Optional[datetime]:
    events_path = _runflow_events_path(run_dir)
    try:
        if not events_path.exists():
            return None
        lines = events_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        log.debug(
            "VALIDATION_EVENTS_READ_FAILED sid=%s path=%s",
            _run_dir_sid(run_dir),
            events_path,
            exc_info=True,
        )
        return None

    for raw in reversed(lines):
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        if payload.get("event") != event_name:
            continue
        timestamp = _parse_timestamp(payload.get("ts"))
        if timestamp is not None:
            return timestamp
    return None


def _rotate_validation_fastpath_lock(run_dir: Path) -> Path:
    lock_path = _validation_fastpath_lock_path(run_dir)
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        log.debug(
            "VALIDATION_FASTPATH_LOCK_ROTATE_UNLINK_FAILED sid=%s path=%s",
            _run_dir_sid(run_dir),
            lock_path,
            exc_info=True,
        )

    timestamp = _now_iso()
    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(timestamp, encoding="utf-8")
    except OSError:
        log.warning(
            "VALIDATION_FASTPATH_LOCK_ROTATE_FAILED sid=%s path=%s",
            _run_dir_sid(run_dir),
            lock_path,
            exc_info=True,
        )
    return lock_path


def _stage_merge_zero_flag(stage_payload: Mapping[str, Any]) -> bool:
    candidates: list[Any] = [stage_payload.get("merge_zero_packs")]
    metrics_payload = stage_payload.get("metrics")
    if isinstance(metrics_payload, Mapping):
        candidates.append(metrics_payload.get("merge_zero_packs"))
    summary_payload = stage_payload.get("summary")
    if isinstance(summary_payload, Mapping):
        candidates.append(summary_payload.get("merge_zero_packs"))
        summary_metrics = summary_payload.get("metrics")
        if isinstance(summary_metrics, Mapping):
            candidates.append(summary_metrics.get("merge_zero_packs"))

    for candidate in candidates:
        boolish = _coerce_boolish(candidate)
        if boolish is None:
            continue
        if boolish:
            return True
    return False


def _stage_merge_zero_flag_for_sid(
    run_dir: Path, 
    runflow_data: Mapping[str, Any] | None = None,
    validation_stage: Mapping[str, Any] | None = None
) -> bool:
    """Read merge_zero_packs preferring merge stage/umbrella, fallback to validation for legacy."""
    # Load runflow if not provided
    if runflow_data is None:
        runflow_path = run_dir / "runflow.json"
        try:
            raw = runflow_path.read_text(encoding="utf-8")
            runflow_data = json.loads(raw)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            runflow_data = {}
    
    if not isinstance(runflow_data, Mapping):
        runflow_data = {}
    
    # Check umbrella barriers first
    umbrella = runflow_data.get("umbrella_barriers")
    if isinstance(umbrella, Mapping):
        merge_zero = umbrella.get("merge_zero_packs")
        if isinstance(merge_zero, bool):
            return merge_zero
    
    # Check merge stage
    stages = runflow_data.get("stages")
    if isinstance(stages, Mapping):
        merge_stage = stages.get("merge")
        if isinstance(merge_stage, Mapping):
            if _stage_merge_zero_flag(merge_stage):
                return True
        
        # Check validation stage from runflow_data if available
        validation_from_data = stages.get("validation")
        if isinstance(validation_from_data, Mapping):
            if _stage_merge_zero_flag(validation_from_data):
                return True
    
    # Fallback to validation_stage parameter (for test contexts)
    if isinstance(validation_stage, Mapping):
        if _stage_merge_zero_flag(validation_stage):
            return True
    
    return False


def _validation_stuck_fastpath(
    sid: str,
    run_dir: Path,
    snapshot: Mapping[str, Any],
) -> Optional[dict[str, Any]]:
    stages_payload = snapshot.get("stages")
    if not isinstance(stages_payload, Mapping):
        return None

    validation_stage = stages_payload.get("validation")
    if not isinstance(validation_stage, Mapping):
        return None

    status_raw = validation_stage.get("status")
    status_normalized = str(status_raw).strip().lower() if isinstance(status_raw, str) else ""
    if status_normalized not in {"built", "in_progress"}:
        return None

    if not _stage_merge_zero_flag(validation_stage):
        return None

    now = datetime.now(timezone.utc)

    lock_path = _validation_fastpath_lock_path(run_dir)
    lock_exists = False
    lock_age_seconds: Optional[float] = None
    try:
        stat = lock_path.stat()
    except FileNotFoundError:
        stat = None
    except OSError:
        stat = None
        log.debug(
            "VALIDATION_FASTPATH_LOCK_STAT_FAILED sid=%s path=%s",
            sid,
            lock_path,
            exc_info=True,
        )

    if stat is not None:
        lock_exists = True
        lock_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        lock_age_seconds = (now - lock_mtime).total_seconds()

    last_sender_started = _last_validation_sender_event_timestamp(
        run_dir, "validation.sender.started"
    )

    stale_lock = (
        lock_exists
        and lock_age_seconds is not None
        and lock_age_seconds >= _VALIDATION_FASTPATH_WATCHDOG_LOCK_SECONDS
    )

    stale_events = (
        last_sender_started is None
        or (now - last_sender_started).total_seconds() >= _VALIDATION_FASTPATH_WATCHDOG_EVENT_SECONDS
    )

    if not stale_lock and not stale_events:
        return None

    context: dict[str, Any] = {
        "lock_path": str(lock_path),
        "lock_exists": lock_exists,
        "lock_age_seconds": lock_age_seconds,
        "stale_lock": stale_lock,
        "stale_events": stale_events,
    }
    if last_sender_started is not None:
        context["last_sender_started"] = last_sender_started.isoformat().replace("+00:00", "Z")
    else:
        context["last_sender_started"] = None

    return context


def _watchdog_trigger_validation_fastpath(
    sid: str,
    run_dir: Path,
    snapshot: dict[str, Any],
    context: Mapping[str, Any],
) -> bool:
    try:
        from celery import chain
        from backend.pipeline import auto_ai_tasks
    except Exception:
        log.warning(
            "VALIDATION_FASTPATH_WATCHDOG_IMPORT_FAILED sid=%s",
            sid,
            exc_info=True,
        )
        return False

    lock_path = _rotate_validation_fastpath_lock(run_dir)

    payload = {
        "sid": sid,
        "runs_root": str(run_dir.parent),
        "merge_zero_packs": True,
        "fastpath": True,
    }

    workflow = chain(
        auto_ai_tasks.validation_build_packs.s(payload),
        # Watchdog re-enqueue follows the same ordering guarantees: send before compact.
        auto_ai_tasks.validation_send.s(),
        auto_ai_tasks.validation_compact.s(),
    )
    try:
        workflow.apply_async(queue="validation")
    except Exception:
        log.warning(
            "VALIDATION_FASTPATH_WATCHDOG_QUEUE_FAILED sid=%s",
            sid,
            exc_info=True,
        )
        return False

    log.info(
        "VALIDATION_FASTPATH_WATCHDOG_REENQUEUE sid=%s lock_path=%s lock_age=%.2f last_sender_started=%s",
        sid,
        lock_path,
        float(context.get("lock_age_seconds") or 0.0),
        context.get("last_sender_started"),
    )

    stages_payload = snapshot.setdefault("stages", {})
    if isinstance(stages_payload, Mapping) and not isinstance(stages_payload, dict):
        stages_payload = dict(stages_payload)
        snapshot["stages"] = stages_payload

    validation_stage = stages_payload.get("validation")
    if not isinstance(validation_stage, dict):
        validation_stage = dict(validation_stage) if isinstance(validation_stage, Mapping) else {}
        stages_payload["validation"] = validation_stage

    validation_stage["status"] = "in_progress"
    validation_stage["sent"] = False
    validation_stage["last_at"] = _now_iso()
    validation_stage.setdefault("empty_ok", False)
    validation_stage.setdefault("metrics", {})
    metrics_payload = validation_stage.get("metrics")
    if isinstance(metrics_payload, Mapping) and not isinstance(metrics_payload, dict):
        metrics_payload = dict(metrics_payload)
        validation_stage["metrics"] = metrics_payload
    if isinstance(metrics_payload, dict):
        metrics_payload.setdefault("merge_zero_packs", True)

    return True
def _maybe_enqueue_validation_fastpath(sid: str, run_dir: Path, snapshot: dict[str, Any]) -> bool:
    stages_payload = snapshot.get("stages")
    if isinstance(stages_payload, dict):
        stages = stages_payload
    elif isinstance(stages_payload, Mapping):
        stages = dict(stages_payload)
        snapshot["stages"] = stages
    else:
        stages = {}
        snapshot["stages"] = stages

    merge_stage_payload = stages.get("merge")
    if isinstance(merge_stage_payload, dict):
        merge_stage = merge_stage_payload
    elif isinstance(merge_stage_payload, Mapping):
        merge_stage = dict(merge_stage_payload)
        stages["merge"] = merge_stage
    else:
        return False

    # ── DEFENSIVE: CHECK merge_ai_applied FOR NON-ZERO-PACKS ──────────────
    # Fastpath is only for zero-packs cases. If merge_ai_applied is missing,
    # defer to normal orchestrator flow to avoid timing bug.
    merge_empty_ok = _stage_empty_ok(merge_stage)
    if not merge_empty_ok:
        merge_ai_applied = merge_stage.get("merge_ai_applied", False)
        if not merge_ai_applied:
            log.info(
                "VALIDATION_FASTPATH_SKIP sid=%s reason=merge_not_ai_applied empty_ok=%s",
                sid,
                merge_empty_ok,
            )
            return False
    # ───────────────────────────────────────────────────────────────────────

    merge_zero_candidates: list[Any] = [merge_stage.get("merge_zero_packs")]
    metrics_payload = merge_stage.get("metrics")
    if isinstance(metrics_payload, Mapping):
        merge_zero_candidates.append(metrics_payload.get("merge_zero_packs"))
    summary_payload = merge_stage.get("summary")
    if isinstance(summary_payload, Mapping):
        merge_zero_candidates.append(summary_payload.get("merge_zero_packs"))
        summary_metrics_payload = summary_payload.get("metrics")
        if isinstance(summary_metrics_payload, Mapping):
            merge_zero_candidates.append(summary_metrics_payload.get("merge_zero_packs"))

    merge_zero_flag = False
    merge_zero_known = False
    for candidate in merge_zero_candidates:
        boolish = _coerce_boolish(candidate)
        if boolish is None:
            continue
        merge_zero_known = True
        if boolish:
            merge_zero_flag = True
            break
    if not merge_zero_flag:
        if not merge_zero_known:
            pairs_scored = _coerce_int(merge_stage.get("pairs_scored"))
            if pairs_scored is None:
                pairs_scored = _coerce_int(merge_stage.get("scored_pairs"))
            if pairs_scored is None and isinstance(summary_payload, Mapping):
                pairs_scored = _coerce_int(summary_payload.get("pairs_scored"))
            if pairs_scored is None and isinstance(metrics_payload, Mapping):
                pairs_scored = _coerce_int(metrics_payload.get("pairs_scored"))

            packs_created = _coerce_int(merge_stage.get("packs_created"))
            if packs_created is None:
                packs_created = _coerce_int(merge_stage.get("pack_files"))
            if packs_created is None and isinstance(summary_payload, Mapping):
                packs_created = _coerce_int(summary_payload.get("packs_created"))
            if packs_created is None and isinstance(metrics_payload, Mapping):
                packs_created = _coerce_int(metrics_payload.get("created_packs"))

            if (
                pairs_scored is not None
                and packs_created is not None
                and pairs_scored > 0
                and packs_created == 0
            ):
                merge_zero_flag = True

    if not merge_zero_flag:
        return False

    validation_stage_payload = stages.get("validation")
    if isinstance(validation_stage_payload, dict):
        validation_stage = validation_stage_payload
    elif isinstance(validation_stage_payload, Mapping):
        validation_stage = dict(validation_stage_payload)
        stages["validation"] = validation_stage
    else:
        validation_stage = {}
        stages["validation"] = validation_stage

    sent_flag = bool(validation_stage.get("sent"))

    metrics_existing = validation_stage.get("metrics")
    summary_existing = validation_stage.get("summary")

    results_payload = validation_stage.get("results")
    if isinstance(results_payload, Mapping):
        results = dict(results_payload)
    else:
        results = {}

    total = _coerce_int(results.get("results_total"))
    if total is None:
        total = _coerce_int(results.get("total"))
    if total is None:
        total = 0

    if total <= 0 and isinstance(metrics_existing, Mapping):
        metrics_total = _coerce_int(metrics_existing.get("packs_total"))
        if metrics_total is not None and metrics_total > 0:
            total = metrics_total
    if total <= 0 and isinstance(summary_existing, Mapping):
        summary_metrics = summary_existing.get("metrics")
        if isinstance(summary_metrics, Mapping):
            summary_total = _coerce_int(summary_metrics.get("packs_total"))
            if summary_total is not None and summary_total > 0:
                total = summary_total

    completed = _coerce_int(results.get("completed"))
    if completed is None:
        completed = 0

    results_incomplete = completed < total

    if sent_flag and not results_incomplete:
        return False

    if _fastpath_has_lock(run_dir):
        return False

    if not _validation_index_ready(run_dir):
        return False

    skip_counts: dict[str, int] = {}
    skip_reason: Optional[str] = None

    def _accumulate_skip(source: Mapping[str, Any]) -> None:
        nonlocal skip_counts, skip_reason
        reason_candidate = source.get("skip_reason_top")
        if isinstance(reason_candidate, str) and reason_candidate.strip():
            if not skip_reason:
                skip_reason = reason_candidate.strip()
        counts_candidate = source.get("skip_counts")
        if isinstance(counts_candidate, Mapping):
            for key, value in counts_candidate.items():
                if not isinstance(key, str):
                    continue
                coerced = _coerce_int(value)
                if coerced is None:
                    continue
                skip_counts[key] = coerced

    # Ensure local payload variables are defined before use
    metrics_payload = validation_stage.get("metrics")
    summary_payload = validation_stage.get("summary")

    if isinstance(metrics_payload, Mapping):
        _accumulate_skip(metrics_payload)
    if isinstance(summary_payload, Mapping):
        _accumulate_skip(summary_payload)
        summary_metrics_payload = summary_payload.get("metrics")
        if isinstance(summary_metrics_payload, Mapping):
            _accumulate_skip(summary_metrics_payload)

    enqueue_payload: dict[str, Any] = {"merge_zero_fastpath": True}
    if skip_counts:
        enqueue_payload["skip_counts"] = dict(skip_counts)
    if skip_reason:
        enqueue_payload["skip_reason_top"] = skip_reason

    enqueued = _enqueue_validation_fastpath(
        sid,
        run_dir,
        merge_zero_packs=True,
        payload=enqueue_payload,
    )

    if not enqueued:
        return False

    now_iso = _now_iso()

    validation_stage["sent"] = True
    validation_status_raw = validation_stage.get("status")
    validation_status = str(validation_status_raw or "").strip().lower()
    if validation_status in {"", "empty", "built", "pending"}:
        validation_stage["status"] = "in_progress"
    elif validation_status not in {"success", "error", "in_progress"}:
        validation_stage["status"] = "in_progress"

    validation_stage["last_at"] = now_iso

    metrics_dict = validation_stage.get("metrics")
    if isinstance(metrics_dict, Mapping):
        metrics_dict = dict(metrics_dict)
    else:
        metrics_dict = {}
    metrics_dict["merge_zero_packs"] = True
    validation_stage["metrics"] = metrics_dict

    summary_dict = validation_stage.get("summary")
    if isinstance(summary_dict, Mapping):
        summary_dict = dict(summary_dict)
    else:
        summary_dict = {}
    summary_dict["merge_zero_packs"] = True

    summary_metrics_dict = summary_dict.get("metrics")
    if isinstance(summary_metrics_dict, Mapping):
        summary_metrics_dict = dict(summary_metrics_dict)
    else:
        summary_metrics_dict = {}
    summary_metrics_dict["merge_zero_packs"] = True
    summary_dict["metrics"] = summary_metrics_dict
    validation_stage["summary"] = summary_dict

    validation_stage["results"] = results
    stages["validation"] = validation_stage

    if snapshot.get("run_state") == "INIT":
        snapshot["run_state"] = "VALIDATING"

    log.info(
        "VALIDATION_FASTPATH_AUTO_ENQUEUE sid=%s merge_zero_packs=%s",
        sid,
        True,
    )

    return True


_STAGE_STATUS_PRIORITY: dict[str, int] = {
    "error": 100,
    "published": 90,
    "success": 80,
    "in_progress": 50,
    "built": 40,
    "empty": 30,
    "pending": 20,
}


_RUN_STATE_PRIORITY: dict[str, int] = {
    "ERROR": 100,
    "AWAITING_CUSTOMER_INPUT": 80,
    "COMPLETE_NO_ACTION": 70,
    "VALIDATING": 60,
    "INIT": 10,
}


def _normalize_stage_status_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    return ""


def _prefer_stage_status(existing: Any, candidate: Any) -> Any:
    candidate_text = candidate if isinstance(candidate, str) else ""
    existing_text = existing if isinstance(existing, str) else ""

    candidate_normalized = _normalize_stage_status_value(candidate_text)
    existing_normalized = _normalize_stage_status_value(existing_text)

    candidate_rank = _STAGE_STATUS_PRIORITY.get(candidate_normalized, -1)
    existing_rank = _STAGE_STATUS_PRIORITY.get(existing_normalized, -1)

    if candidate_rank >= existing_rank:
        return candidate if candidate_text else existing
    return existing


def _merge_nested_mapping(
    existing: Mapping[str, Any] | None, incoming: Mapping[str, Any]
) -> dict[str, Any]:
    merged: dict[str, Any]
    if isinstance(existing, Mapping):
        merged = dict(existing)
    else:
        merged = {}

    for key, value in incoming.items():
        merged[str(key)] = value

    return merged


def _merge_stage_snapshot(
    existing: Mapping[str, Any] | None, incoming: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge a single stage snapshot.

    For validation promotion snapshots (incoming contains _writer == 'validation_promotion'),
    the metrics and summary subtrees are treated as authoritative and replace existing values.
    All other mappings (including results) continue to use deep merge semantics.
    The internal '_writer' marker is stripped from the merged output.
    """
    merged: dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
    is_validation_promotion = incoming.get("_writer") == "validation_promotion"

    for key, value in incoming.items():
        # Drop internal marker
        if key == "_writer":
            continue

        if key == "status":
            existing_status = merged.get("status")
            merged["status"] = _prefer_stage_status(existing_status, value)
            continue

        if key in {"metrics", "results", "summary"} and isinstance(value, Mapping):
            if is_validation_promotion and key in {"metrics", "summary"}:
                merged[key] = dict(value)
            else:
                merged[key] = _merge_nested_mapping(
                    merged.get(key) if isinstance(merged.get(key), Mapping) else None, value
                )
            continue

        if key == "empty_ok":
            incoming_bool = bool(value) if isinstance(value, (bool, int, float)) else value
            existing_value = merged.get("empty_ok")
            if isinstance(incoming_bool, bool):
                if isinstance(existing_value, (bool, int, float)):
                    merged["empty_ok"] = bool(existing_value) or incoming_bool
                else:
                    merged["empty_ok"] = incoming_bool
            else:
                merged["empty_ok"] = incoming_bool
            continue

        if key == "last_at" and isinstance(value, str):
            existing_last = merged.get("last_at")
            if not isinstance(existing_last, str) or value >= existing_last:
                merged["last_at"] = value
            continue

        merged[key] = value

    return merged


def _normalize_run_state(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in _RUN_STATE_PRIORITY:
            return normalized
    return ""


def _prefer_run_state(existing: Any, candidate: Any) -> str:
    candidate_state = _normalize_run_state(candidate)
    if not candidate_state:
        normalized_existing = _normalize_run_state(existing)
        return normalized_existing or "INIT"

    existing_state = _normalize_run_state(existing)
    if not existing_state:
        return candidate_state

    existing_rank = _RUN_STATE_PRIORITY.get(existing_state, -1)
    candidate_rank = _RUN_STATE_PRIORITY.get(candidate_state, -1)

    if candidate_rank >= existing_rank:
        return candidate_state
    return existing_state


def _merge_runflow_snapshots(
    existing: Mapping[str, Any] | None, incoming: Mapping[str, Any]
) -> dict[str, Any]:
    if isinstance(existing, Mapping):
        merged: dict[str, Any] = dict(existing)
    else:
        merged = {}

    for key, value in incoming.items():
        if key == "stages" and isinstance(value, Mapping):
            existing_stages = merged.get("stages")
            if isinstance(existing_stages, Mapping):
                stages: dict[str, Any] = dict(existing_stages)
            else:
                stages = {}

            for stage_name, stage_payload in value.items():
                if not isinstance(stage_name, str):
                    continue
                existing_stage = stages.get(stage_name)
                if isinstance(stage_payload, Mapping):
                    stages[stage_name] = _merge_stage_snapshot(existing_stage, stage_payload)
                else:
                    stages[stage_name] = stage_payload

            merged["stages"] = stages
            continue

        if key == "umbrella_barriers" and isinstance(value, Mapping):
            existing_barriers = merged.get("umbrella_barriers")
            if isinstance(existing_barriers, Mapping):
                barriers = dict(existing_barriers)
            else:
                barriers = {}
            for barrier_key, barrier_value in value.items():
                barriers[barrier_key] = barrier_value
            merged["umbrella_barriers"] = barriers
            continue

        if key == "umbrella_ready":
            if isinstance(value, bool):
                merged["umbrella_ready"] = value
            elif value is not None:
                merged["umbrella_ready"] = bool(value)
            continue

        if key == "run_state":
            merged["run_state"] = _prefer_run_state(merged.get("run_state"), value)
            continue

        if key == "updated_at" and isinstance(value, str):
            existing_updated = merged.get("updated_at")
            if not isinstance(existing_updated, str) or value >= existing_updated:
                merged["updated_at"] = value
            continue

        merged[key] = value

    if "sid" not in merged and isinstance(incoming.get("sid"), str):
        merged["sid"] = incoming["sid"].strip()

    return merged


def _next_snapshot_version(*values: Any) -> int:
    candidates: list[int] = []
    for value in values:
        coerced = _coerce_int(value)
        if coerced is not None:
            candidates.append(coerced)

    if not candidates:
        return 1

    return max(candidates) + 1


def _ensure_merge_zero_pack_metadata(
    sid: str,
    run_dir: Path,
    snapshot: dict[str, Any],
    *,
    manifest_payload: Mapping[str, Any] | None = None,
) -> bool:
    if not _UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG:
        return False

    stages_payload = snapshot.get("stages")
    if isinstance(stages_payload, dict):
        stages = stages_payload
    elif isinstance(stages_payload, Mapping):
        stages = dict(stages_payload)
        snapshot["stages"] = stages
    else:
        return False

    merge_stage_payload = stages.get("merge")
    if isinstance(merge_stage_payload, dict):
        merge_stage = merge_stage_payload
    elif isinstance(merge_stage_payload, Mapping):
        merge_stage = dict(merge_stage_payload)
        stages["merge"] = merge_stage
    else:
        return False

    metrics_payload = merge_stage.get("metrics")
    if isinstance(metrics_payload, dict):
        metrics = metrics_payload
    elif isinstance(metrics_payload, Mapping):
        metrics = dict(metrics_payload)
        merge_stage["metrics"] = metrics
    else:
        metrics = {}
        merge_stage["metrics"] = metrics

    summary_payload = merge_stage.get("summary")
    if isinstance(summary_payload, dict):
        summary = summary_payload
    elif isinstance(summary_payload, Mapping):
        summary = dict(summary_payload)
        merge_stage["summary"] = summary
    else:
        summary = {}
        merge_stage["summary"] = summary

    existing_flag = metrics.get("merge_zero_packs")
    summary_flag = summary.get("merge_zero_packs")
    if isinstance(existing_flag, bool) and existing_flag:
        flag_present = True
    elif isinstance(summary_flag, bool) and summary_flag:
        flag_present = True
    else:
        flag_present = False

    manifest_data = manifest_payload
    if manifest_data is None:
        manifest_path = run_dir / "manifest.json"
        manifest_data = _load_json_mapping(manifest_path)

    index_path = _resolve_merge_index_path(sid, run_dir, manifest_data)
    if not index_path.exists():
        return False

    index_payload = _load_json_mapping(index_path)
    if not isinstance(index_payload, Mapping):
        return False

    totals_payload = index_payload.get("totals")
    if not isinstance(totals_payload, Mapping):
        return False

    merge_zero_raw = totals_payload.get("merge_zero_packs")
    merge_zero_flag = bool(merge_zero_raw is True)

    if not merge_zero_flag and flag_present:
        return False

    pairs_scored_value = _coerce_int(totals_payload.get("scored_pairs"))
    created_packs_value = _coerce_int(totals_payload.get("created_packs"))
    reason_value = totals_payload.get("reason")
    reason_text = reason_value.strip() if isinstance(reason_value, str) else None
    skip_reason_value = totals_payload.get("skip_reason_top")
    skip_reason_text = skip_reason_value.strip() if isinstance(skip_reason_value, str) else None

    skip_counts_payload = totals_payload.get("skip_counts")
    normalized_skip_counts: dict[str, int] = {}
    if isinstance(skip_counts_payload, Mapping):
        for key, value in skip_counts_payload.items():
            if not isinstance(key, str):
                continue
            coerced = _coerce_int(value)
            if coerced is None:
                continue
            normalized_skip_counts[key] = coerced

    metrics_updated = False

    if pairs_scored_value is not None:
        metrics["pairs_scored"] = pairs_scored_value
        metrics.setdefault("scored_pairs", pairs_scored_value)
        metrics_updated = True

    if created_packs_value is not None:
        metrics["created_packs"] = created_packs_value
        metrics_updated = True

    if normalized_skip_counts:
        metrics["skip_counts"] = dict(normalized_skip_counts)
        metrics_updated = True

    if skip_reason_text:
        metrics["skip_reason_top"] = skip_reason_text
        metrics_updated = True

    if reason_text:
        metrics["reason"] = reason_text
        metrics_updated = True

    if merge_zero_flag:
        metrics["merge_zero_packs"] = True
        summary["merge_zero_packs"] = True
        merge_stage["merge_zero_packs"] = True
        metrics_updated = True

    if pairs_scored_value is not None:
        summary.setdefault("pairs_scored", pairs_scored_value)
    if created_packs_value is not None:
        summary.setdefault("packs_created", created_packs_value)
    if normalized_skip_counts:
        summary.setdefault("skip_counts", dict(normalized_skip_counts))
    if skip_reason_text:
        summary.setdefault("skip_reason_top", skip_reason_text)
    if reason_text:
        summary.setdefault("reason", reason_text)
        summary.setdefault("merge_reason", reason_text)

    barriers_payload = snapshot.get("umbrella_barriers")
    if isinstance(barriers_payload, dict):
        barriers = barriers_payload
    elif isinstance(barriers_payload, Mapping):
        barriers = dict(barriers_payload)
        snapshot["umbrella_barriers"] = barriers
    else:
        barriers = {
            "merge_ready": False,
            "validation_ready": False,
            "review_ready": False,
            "style_ready": False,
            "all_ready": False,
        }
        snapshot["umbrella_barriers"] = barriers

    if merge_zero_flag:
        barriers["merge_zero_packs"] = True

    return metrics_updated or merge_zero_flag


def _run_dir_sid(run_dir: Path) -> str:
    name = run_dir.name
    if name:
        return name
    stem = run_dir.stem
    if stem:
        return stem
    return str(run_dir)


def _warn_unknown_status(stage: str, run_dir: Path, status: str) -> None:
    sid = _run_dir_sid(run_dir)
    key = (stage, sid)
    if key in _UNKNOWN_STATUS_WARNINGS:
        return
    _UNKNOWN_STATUS_WARNINGS.add(key)
    log.warning(
        "RUNFLOW_UNKNOWN_STATUS stage=%s sid=%s status=%s",
        stage,
        sid,
        status,
    )


def _normalize_terminal_status(
    status: Any, *, stage: str, run_dir: Path
) -> Optional[str]:
    if isinstance(status, str):
        normalized = status.strip().lower()
    elif isinstance(status, (bytes, bytearray)):
        try:
            normalized = bytes(status).decode("utf-8", errors="ignore").strip().lower()
        except Exception:  # pragma: no cover - defensive
            normalized = ""
    else:
        normalized = ""

    if not normalized:
        return None

    mapped = _STATUS_NORMALIZATION.get(normalized)
    if mapped is not None:
        return mapped

    _warn_unknown_status(stage, run_dir, normalized)
    return None


def _default_umbrella_barriers() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "merge_ready": False,
        "validation_ready": False,
        "strategy_ready": False,
        "strategy_required": False,
        "review_ready": False,
        "style_ready": False,
        "all_ready": False,
        "checked_at": None,
    }

    if _document_verifier_enabled():
        payload["document_ready"] = False

    return payload


def _normalise_umbrella_barriers(payload: Any) -> dict[str, Any]:
    result = _default_umbrella_barriers()
    if isinstance(payload, Mapping):
        for key in (
            "merge_ready",
            "validation_ready",
            "strategy_ready",
            "strategy_required",
            "review_ready",
            "style_ready",
            "all_ready",
        ):
            value = payload.get(key)
            if isinstance(value, bool):
                result[key] = value
        checked_at = payload.get("checked_at")
        if isinstance(checked_at, str):
            result["checked_at"] = checked_at

        document_ready = payload.get("document_ready")
        if isinstance(document_ready, bool):
            result["document_ready"] = document_ready
    return result


def _load_runflow(path: Path, sid: str) -> dict[str, Any]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        data: dict[str, Any] = {
            "sid": sid,
            "run_state": "INIT",
            "stages": {},
            "updated_at": _now_iso(),
            "umbrella_barriers": _default_umbrella_barriers(),
        }
        return data
    except OSError:
        log.warning("RUNFLOW_READ_FAILED sid=%s path=%s", sid, path, exc_info=True)
        return {
            "sid": sid,
            "run_state": "INIT",
            "stages": {},
            "updated_at": _now_iso(),
            "umbrella_barriers": _default_umbrella_barriers(),
            # Marker to indicate runflow was unavailable due to an IO error
            "runflow_unavailable": True,
        }

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("RUNFLOW_PARSE_FAILED sid=%s path=%s", sid, path, exc_info=True)
        payload = {}

    if not isinstance(payload, Mapping):
        payload = {}

    stages = payload.get("stages")
    if not isinstance(stages, Mapping):
        stages = {}

    run_state = payload.get("run_state")
    if not isinstance(run_state, str) or run_state not in {
        "INIT",
        "VALIDATING",
        "AWAITING_CUSTOMER_INPUT",
        "COMPLETE_NO_ACTION",
        "ERROR",
    }:
        run_state = "INIT"

    umbrella_barriers = _normalise_umbrella_barriers(payload.get("umbrella_barriers"))

    umbrella_ready_value = payload.get("umbrella_ready")
    if isinstance(umbrella_ready_value, bool):
        umbrella_ready = umbrella_ready_value
    else:
        umbrella_ready = bool(umbrella_barriers.get("all_ready"))

    snapshot_version_value = _coerce_int(payload.get("snapshot_version"))
    snapshot_version = snapshot_version_value if snapshot_version_value is not None else 0

    last_writer_value = payload.get("last_writer")
    last_writer = last_writer_value if isinstance(last_writer_value, str) else ""

    return {
        "sid": sid,
        "run_state": run_state,
        "stages": dict(stages),
        "updated_at": str(payload.get("updated_at") or _now_iso()),
        "umbrella_barriers": umbrella_barriers,
        "umbrella_ready": umbrella_ready,
        "snapshot_version": snapshot_version,
        "last_writer": last_writer,
    }


def get_runflow_snapshot(
    sid: str,
    runs_root: Optional[str | Path] = None,
) -> dict[str, Any]:
    """Return a deep copy of the runflow snapshot for ``sid``."""

    path = _runflow_path(sid, runs_root)
    snapshot = _load_runflow(path, sid)
    return json.loads(json.dumps(snapshot))


def record_stage(
    sid: str,
    stage: StageName,
    *,
    status: StageStatus,
    counts: Dict[str, int],
    empty_ok: bool,
    notes: Optional[str] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    results: Optional[Mapping[str, Any]] = None,
    runs_root: Optional[str | Path] = None,
    refresh_barriers: bool = True,
) -> dict[str, Any]:
    """Persist ``stage`` information under ``runs/<sid>/runflow.json``."""

    path = _runflow_path(sid, runs_root)
    base_dir = path.parent
    data = _load_runflow(path, sid)

    stage_payload: dict[str, Any] = {
        "status": status,
        "empty_ok": bool(empty_ok),
        "last_at": _now_iso(),
    }

    stages_snapshot = data.get("stages")
    existing_stage: Mapping[str, Any] | None
    if isinstance(stages_snapshot, Mapping):
        existing_stage = stages_snapshot.get(stage)
    else:
        existing_stage = None

    existing_stage_metrics: dict[str, Any] = {}
    existing_stage_results: dict[str, Any] = {}
    existing_summary_metrics: dict[str, Any] = {}
    existing_summary_results: dict[str, Any] = {}
    if isinstance(existing_stage, Mapping):
        existing_metrics_payload = existing_stage.get("metrics")
        if isinstance(existing_metrics_payload, Mapping):
            existing_stage_metrics = dict(existing_metrics_payload)
        existing_results_payload = existing_stage.get("results")
        if isinstance(existing_results_payload, Mapping):
            existing_stage_results = dict(existing_results_payload)
        summary_payload_existing = existing_stage.get("summary")
        if isinstance(summary_payload_existing, Mapping):
            summary_metrics_existing = summary_payload_existing.get("metrics")
            if isinstance(summary_metrics_existing, Mapping):
                existing_summary_metrics = dict(summary_metrics_existing)
            summary_results_existing = summary_payload_existing.get("results")
            if isinstance(summary_results_existing, Mapping):
                existing_summary_results = dict(summary_results_existing)

    normalized_counts: dict[str, int] = {}
    for key, value in (counts or {}).items():
        coerced = _coerce_int(value)
        if coerced is not None:
            normalized_counts[str(key)] = coerced

    def _normalize_mapping(payload: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        normalized: dict[str, Any] = {}
        if not isinstance(payload, Mapping):
            return normalized
        for key, value in payload.items():
            str_key = str(key)
            if isinstance(value, Mapping):
                nested: dict[str, Any] = {}
                for nested_key, nested_value in value.items():
                    nested_coerced = _coerce_int(nested_value)
                    if nested_coerced is not None:
                        nested[str(nested_key)] = nested_coerced
                    elif isinstance(nested_value, bool):
                        nested[str(nested_key)] = bool(nested_value)
                    elif isinstance(nested_value, str):
                        nested[str(nested_key)] = nested_value
                normalized[str_key] = nested
                continue
            if isinstance(value, bool):
                normalized[str_key] = bool(value)
                continue
            if isinstance(value, str):
                normalized[str_key] = value
                continue
            coerced_value = _coerce_int(value)
            if coerced_value is not None:
                normalized[str_key] = coerced_value
        return normalized

    normalized_metrics = _normalize_mapping(metrics)
    merge_reason: Optional[str] = None
    merge_skip_reason_top: Optional[str] = None
    merge_zero_flag: Optional[bool] = None
    merge_skip_counts: dict[str, int] = {}
    if stage == "merge":
        reason_candidate = normalized_metrics.get("merge_reason")
        if isinstance(reason_candidate, str) and reason_candidate.strip():
            merge_reason = reason_candidate.strip()
        elif isinstance(normalized_metrics.get("reason"), str):
            legacy_reason = str(normalized_metrics["reason"]).strip()
            if legacy_reason:
                merge_reason = legacy_reason
        skip_candidate = normalized_metrics.get("skip_reason_top")
        if isinstance(skip_candidate, str) and skip_candidate.strip():
            merge_skip_reason_top = skip_candidate.strip()
        skip_counts_candidate = normalized_metrics.get("skip_counts")
        if isinstance(skip_counts_candidate, Mapping):
            for key, value in skip_counts_candidate.items():
                coerced_skip = _coerce_int(value)
                if coerced_skip is None:
                    continue
                merge_skip_counts[str(key)] = coerced_skip
        merge_zero_candidate = normalized_metrics.get("merge_zero_packs")
        if merge_zero_candidate is not None:
            merge_zero_flag = bool(merge_zero_candidate)
    normalized_results = _normalize_mapping(results)

    if stage == "frontend":
        answers_metrics = _frontend_answers_counters(
            base_dir, attachments_required=_review_attachment_required()
        )
        answers_required = _coerce_int(answers_metrics.get("answers_required"))
        answers_received = _coerce_int(answers_metrics.get("answers_received"))
        if answers_required is not None:
            normalized_metrics["answers_required"] = answers_required
        if answers_received is not None:
            normalized_metrics["answers_received"] = answers_received

    disk_counts = _stage_counts_from_disk(stage, base_dir)
    for key, value in disk_counts.items():
        coerced_disk = _coerce_int(value)
        if coerced_disk is None:
            continue
        existing = normalized_counts.get(str(key))
        if isinstance(existing, int) and existing > 0 and coerced_disk == 0:
            continue
        normalized_counts[str(key)] = coerced_disk

    for key, value in normalized_counts.items():
        stage_payload[key] = value

    merged_stage_metrics = dict(existing_stage_metrics)
    if normalized_metrics:
        merged_stage_metrics.update(normalized_metrics)
    if merged_stage_metrics:
        stage_payload["metrics"] = merged_stage_metrics

    merged_stage_results = dict(existing_stage_results)
    if normalized_results:
        merged_stage_results.update(normalized_results)
    if merged_stage_results:
        stage_payload["results"] = merged_stage_results

    if notes:
        stage_payload["notes"] = str(notes)

    stages = data.setdefault("stages", {})
    if not isinstance(stages, dict):
        stages = {}
        data["stages"] = stages

    # V2 authority: in orchestrator mode, treat generic validation writes as telemetry-only.
    # Only V2-aware paths (refresh_validation_stage_from_index, reconcile_umbrella_barriers,
    # validation_orchestrator finalize) are allowed to control stages.validation.*
    if stage == "validation" and _validation_v2_mode_enabled():
        # Preserve existing validation stage; do not overwrite with legacy payload.
        # We still proceed to promotion below which rehydrates authoritative fields from disk.
        pass
    else:
        stages[stage] = stage_payload

    if "umbrella_barriers" not in data or not isinstance(data["umbrella_barriers"], dict):
        data["umbrella_barriers"] = _default_umbrella_barriers()

    stage_names = set(stages.keys()) if isinstance(stages, dict) else set()

    if "merge" in stage_names:
        _merge_updated, merge_promoted, merge_log = _apply_merge_stage_promotion(data, base_dir)
        if merge_promoted:
            log.info(
                "MERGE_STAGE_PROMOTED sid=%s result_files=%s",
                sid,
                merge_log.get("result_files"),
            )

    if "validation" in stage_names:
        (
            _validation_updated,
            validation_promoted,
            validation_log,
        ) = _apply_validation_stage_promotion(data, base_dir)
        if validation_promoted:
            log.info(
                "VALIDATION_STAGE_PROMOTED sid=%s total=%s completed=%s failed=%s",
                sid,
                validation_log.get("total"),
                validation_log.get("completed"),
                validation_log.get("failed"),
            )

    if "frontend" in stage_names:
        (
            _frontend_updated,
            frontend_promoted,
            frontend_log,
        ) = _apply_frontend_stage_promotion(data, base_dir)
        if frontend_promoted:
            log.info(
                "FRONTEND_STAGE_PROMOTED sid=%s answers_required=%s answers_received=%s",
                sid,
                frontend_log.get("answers_required"),
                frontend_log.get("answers_received"),
            )

    if "note_style" in stage_names:
        (
            _note_style_updated,
            note_style_promoted,
            note_style_log,
        ) = _apply_note_style_stage_promotion(data, base_dir)
        if note_style_promoted:
            log.info(
                "NOTE_STYLE_STAGE_PROMOTED sid=%s total=%s completed=%s failed=%s",
                sid,
                note_style_log.get("total"),
                note_style_log.get("completed"),
                note_style_log.get("failed"),
            )

    latest_snapshot = _load_runflow(path, sid)
    data = _merge_runflow_snapshots(latest_snapshot, data)

    if "umbrella_barriers" not in data or not isinstance(data["umbrella_barriers"], Mapping):
        data["umbrella_barriers"] = _default_umbrella_barriers()

    if status == "error":
        data["run_state"] = "ERROR"
    elif stage == "validation" and data.get("run_state") == "INIT":
        data["run_state"] = "VALIDATING"

    data["snapshot_version"] = _next_snapshot_version(
        latest_snapshot.get("snapshot_version"), data.get("snapshot_version")
    )
    data["last_writer"] = "record_stage"

    timestamp = _now_iso()
    data["updated_at"] = timestamp

    log.info(
        "RUNFLOW_RECORD sid=%s stage=%s status=%s counts=%s empty_ok=%s",
        sid,
        stage,
        status,
        dict(normalized_counts),
        empty_ok,
    )

    summary_payload: dict[str, Any] = {str(key): value for key, value in normalized_counts.items()}

    def _apply_merge_summary_metrics(target: dict[str, Any], metrics_payload: Mapping[str, Any]) -> None:
        merge_zero_candidate = metrics_payload.get("merge_zero_packs")
        if merge_zero_candidate is not None:
            target["merge_zero_packs"] = bool(merge_zero_candidate)
        skip_reason_candidate = metrics_payload.get("skip_reason_top")
        if isinstance(skip_reason_candidate, str) and skip_reason_candidate.strip():
            target["skip_reason_top"] = skip_reason_candidate.strip()
        skip_counts_candidate = metrics_payload.get("skip_counts")
        if isinstance(skip_counts_candidate, Mapping):
            normalized_skip_counts: dict[str, int] = {}
            for skip_key, skip_val in skip_counts_candidate.items():
                if not isinstance(skip_key, str):
                    continue
                coerced_skip = _coerce_int(skip_val)
                if coerced_skip is None:
                    continue
                normalized_skip_counts[skip_key] = coerced_skip
            if normalized_skip_counts:
                target["skip_counts"] = normalized_skip_counts

    metrics_source_for_summary: Mapping[str, Any] | None
    if merged_stage_metrics:
        metrics_source_for_summary = merged_stage_metrics
    elif normalized_metrics:
        metrics_source_for_summary = normalized_metrics
    else:
        metrics_source_for_summary = None

    summary_metrics_copy = (
        json.loads(json.dumps(metrics_source_for_summary)) if metrics_source_for_summary else {}
    )
    if normalized_metrics and stage == "merge":
        _apply_merge_summary_metrics(summary_payload, normalized_metrics)
        _apply_merge_summary_metrics(stage_payload, normalized_metrics)

    if existing_summary_metrics or summary_metrics_copy:
        merged_summary_metrics = dict(existing_summary_metrics)
        merged_summary_metrics.update(summary_metrics_copy)
        if merged_summary_metrics:
            summary_payload["metrics"] = merged_summary_metrics

    summary_results_copy = dict(normalized_results) if normalized_results else {}
    if existing_summary_results or summary_results_copy:
        merged_summary_results = dict(existing_summary_results)
        merged_summary_results.update(summary_results_copy)
        if merged_summary_results:
            summary_payload["results"] = merged_summary_results
    summary_payload["empty_ok"] = bool(empty_ok)
    if notes:
        summary_payload["notes"] = str(notes)

    if stage == "merge":
        merge_zero_from_metrics = normalized_metrics.get("merge_zero_packs") if isinstance(normalized_metrics, Mapping) else None
        if merge_zero_from_metrics is not None and "merge_zero_packs" not in summary_payload:
            summary_payload["merge_zero_packs"] = bool(merge_zero_from_metrics)
        if merge_zero_from_metrics is not None and "merge_zero_packs" not in stage_payload:
            stage_payload["merge_zero_packs"] = bool(merge_zero_from_metrics)

        if "pairs_scored" in normalized_counts:
            summary_payload["pairs_scored"] = normalized_counts["pairs_scored"]
        if "packs_created" in normalized_counts:
            summary_payload["packs_created"] = normalized_counts["packs_created"]

        if "pairs_scored" not in summary_payload:
            summary_pairs = _coerce_int(stage_payload.get("pairs_scored"))
            if summary_pairs is None:
                summary_pairs = _coerce_int(normalized_counts.get("pairs_scored"))
            if summary_pairs is not None:
                summary_payload["pairs_scored"] = summary_pairs
        if "packs_created" not in summary_payload:
            summary_created = _coerce_int(stage_payload.get("packs_created"))
            if summary_created is None:
                summary_created = _coerce_int(normalized_counts.get("packs_created"))
            if summary_created is not None:
                summary_payload["packs_created"] = summary_created

        skip_reason_from_metrics = normalized_metrics.get("skip_reason_top") if isinstance(normalized_metrics, Mapping) else None
        if isinstance(skip_reason_from_metrics, str) and skip_reason_from_metrics.strip():
            summary_payload.setdefault("skip_reason_top", skip_reason_from_metrics.strip())
            stage_payload.setdefault("skip_reason_top", skip_reason_from_metrics.strip())

        merge_reason_from_metrics = normalized_metrics.get("merge_reason") if isinstance(normalized_metrics, Mapping) else None
        if isinstance(merge_reason_from_metrics, str) and merge_reason_from_metrics.strip():
            normalized_reason = merge_reason_from_metrics.strip()
            summary_payload["merge_reason"] = normalized_reason
            stage_payload["merge_reason"] = normalized_reason
            summary_payload.setdefault("reason", normalized_reason)
            stage_payload.setdefault("reason", normalized_reason)

        skip_counts_from_metrics = normalized_metrics.get("skip_counts") if isinstance(normalized_metrics, Mapping) else None
        if isinstance(skip_counts_from_metrics, Mapping):
            normalized_skip_counts: dict[str, int] = {}
            for skip_key, skip_val in skip_counts_from_metrics.items():
                if not isinstance(skip_key, str):
                    continue
                coerced_skip = _coerce_int(skip_val)
                if coerced_skip is None:
                    continue
                normalized_skip_counts[skip_key] = coerced_skip
            if normalized_skip_counts:
                summary_payload.setdefault("skip_counts", dict(normalized_skip_counts))
                stage_payload.setdefault("skip_counts", dict(normalized_skip_counts))

        if merge_zero_flag is None:
            if isinstance(existing_stage, Mapping):
                existing_zero = existing_stage.get("merge_zero_packs")
                if existing_zero is not None:
                    merge_zero_flag = bool(existing_zero)
                else:
                    existing_summary_payload = existing_stage.get("summary")
                    if isinstance(existing_summary_payload, Mapping):
                        existing_summary_zero = existing_summary_payload.get("merge_zero_packs")
                        if existing_summary_zero is not None:
                            merge_zero_flag = bool(existing_summary_zero)
                        else:
                            existing_metrics_payload = existing_summary_payload.get("metrics")
                            if isinstance(existing_metrics_payload, Mapping):
                                merge_zero_candidate = existing_metrics_payload.get("merge_zero_packs")
                                if merge_zero_candidate is not None:
                                    merge_zero_flag = bool(merge_zero_candidate)
        if merge_zero_flag is not None:
            stage_payload["merge_zero_packs"] = bool(merge_zero_flag)
            summary_payload["merge_zero_packs"] = bool(merge_zero_flag)

        stage_metrics_for_summary = stage_payload.get("metrics")
        if isinstance(stage_metrics_for_summary, Mapping):
            summary_payload.setdefault("metrics", json.loads(json.dumps(stage_metrics_for_summary)))

        if not merge_skip_counts and isinstance(existing_stage, Mapping):
            existing_skip_counts = existing_stage.get("skip_counts")
            if isinstance(existing_skip_counts, Mapping):
                for key, value in existing_skip_counts.items():
                    coerced_skip = _coerce_int(value)
                    if coerced_skip is not None:
                        merge_skip_counts[str(key)] = coerced_skip
            if not merge_skip_counts:
                existing_summary_payload = existing_stage.get("summary")
                if isinstance(existing_summary_payload, Mapping):
                    summary_skip_counts = existing_summary_payload.get("skip_counts")
                    if isinstance(summary_skip_counts, Mapping):
                        for key, value in summary_skip_counts.items():
                            coerced_skip = _coerce_int(value)
                            if coerced_skip is not None:
                                merge_skip_counts[str(key)] = coerced_skip

        if merge_skip_counts:
            stage_payload["skip_counts"] = dict(merge_skip_counts)
            summary_payload["skip_counts"] = dict(merge_skip_counts)

        if merge_reason:
            stage_payload["merge_reason"] = merge_reason
            summary_payload["merge_reason"] = merge_reason
            summary_payload.setdefault("reason", merge_reason)
            stage_payload.setdefault("reason", merge_reason)
        if merge_skip_reason_top:
            stage_payload.setdefault("skip_reason_top", merge_skip_reason_top)
            summary_payload.setdefault("skip_reason_top", merge_skip_reason_top)

        _ensure_merge_zero_pack_metadata(sid, base_dir, data)

    if stage == "merge":
        result_files_value = _coerce_int(summary_payload.get("result_files"))
        if result_files_value is None:
            stage_result_files = _coerce_int(stage_payload.get("result_files"))
            if stage_result_files is None and isinstance(existing_stage, Mapping):
                stage_result_files = _coerce_int(existing_stage.get("result_files"))
            if stage_result_files is not None:
                summary_payload["result_files"] = stage_result_files

    if isinstance(existing_stage, Mapping):
        existing_summary = existing_stage.get("summary")
        if isinstance(existing_summary, Mapping):
            merged_summary = dict(existing_summary)
            merged_summary.update(summary_payload)
            summary_payload = merged_summary

    stage_status_override: Optional[str] = None
    if summary_payload:
        stage_payload["summary"] = dict(summary_payload)

    if stage == "frontend" and status != "error":
        packs_value: Optional[int] = None
        for key in ("packs_count", "packs"):
            packs_value = normalized_counts.get(key)
            if packs_value is not None:
                break
        if packs_value == 0:
            stage_status_override = "empty"

    final_stage_info: dict[str, Any] | None = None
    stages_after_merge = data.get("stages")
    if isinstance(stages_after_merge, Mapping):
        stage_entry = stages_after_merge.get(stage)
        if isinstance(stage_entry, dict):
            final_stage_info = stage_entry
        elif isinstance(stage_entry, Mapping):
            final_stage_info = dict(stage_entry)
            if isinstance(stages_after_merge, dict):
                stages_after_merge[stage] = final_stage_info

    final_stage_status: str = status
    if final_stage_info is not None:
        status_candidate = final_stage_info.get("status")
        if isinstance(status_candidate, str) and status_candidate.strip():
            final_stage_status = status_candidate

        existing_summary_payload = final_stage_info.get("summary")
        if not isinstance(existing_summary_payload, Mapping) and summary_payload:
            final_stage_info["summary"] = dict(summary_payload)
            existing_summary_payload = final_stage_info.get("summary")

        if isinstance(existing_summary_payload, Mapping) and existing_summary_payload:
            if stage == "merge":
                summary_target = dict(existing_summary_payload)
                summary_pairs_val = summary_target.get("pairs_scored")
                if summary_pairs_val is None:
                    stage_pairs = _coerce_int(final_stage_info.get("pairs_scored"))
                    if stage_pairs is None:
                        stage_pairs = _coerce_int(normalized_counts.get("pairs_scored"))
                    if stage_pairs is not None:
                        summary_target["pairs_scored"] = stage_pairs
                summary_packs_val = summary_target.get("packs_created")
                if summary_packs_val is None:
                    stage_packs = _coerce_int(final_stage_info.get("packs_created"))
                    if stage_packs is None:
                        stage_packs = _coerce_int(normalized_counts.get("packs_created"))
                    if stage_packs is not None:
                        summary_target["packs_created"] = stage_packs

                stage_metrics_payload = final_stage_info.get("metrics")
                if isinstance(stage_metrics_payload, Mapping):
                    stage_metrics_copy = json.loads(json.dumps(stage_metrics_payload))
                    existing_summary_metrics = summary_target.get("metrics")
                    if isinstance(existing_summary_metrics, Mapping):
                        merged_summary_metrics = dict(existing_summary_metrics)
                        merged_summary_metrics.update(stage_metrics_copy)
                        summary_target["metrics"] = merged_summary_metrics
                    else:
                        summary_target["metrics"] = stage_metrics_copy

                summary_reason_value = summary_target.get("merge_reason")
                if not isinstance(summary_reason_value, str) or not summary_reason_value.strip():
                    candidate_reason: Optional[str] = None
                    stage_reason_value = stage_payload.get("merge_reason")
                    if isinstance(stage_reason_value, str) and stage_reason_value.strip():
                        candidate_reason = stage_reason_value.strip()
                    elif isinstance(stage_metrics_payload, Mapping):
                        stage_metrics_reason = stage_metrics_payload.get("merge_reason")
                        if isinstance(stage_metrics_reason, str) and stage_metrics_reason.strip():
                            candidate_reason = stage_metrics_reason.strip()
                    if candidate_reason is None:
                        summary_metrics_payload = summary_target.get("metrics")
                        if isinstance(summary_metrics_payload, Mapping):
                            summary_metrics_reason = summary_metrics_payload.get("merge_reason")
                            if isinstance(summary_metrics_reason, str) and summary_metrics_reason.strip():
                                candidate_reason = summary_metrics_reason.strip()
                    if candidate_reason:
                        summary_target["merge_reason"] = candidate_reason
                        summary_target.setdefault("reason", candidate_reason)
                        stage_payload["merge_reason"] = candidate_reason
                        stage_payload.setdefault("reason", candidate_reason)
                        final_stage_info["merge_reason"] = candidate_reason
                        final_stage_info.setdefault("reason", candidate_reason)

                existing_summary_payload = summary_target
                final_stage_info["summary"] = dict(summary_target)

            summary_payload = dict(existing_summary_payload)

        empty_ok_value = final_stage_info.get("empty_ok")
        if isinstance(empty_ok_value, (bool, int, float)):
            empty_ok = bool(empty_ok_value)

    barrier_event: Optional[dict[str, Any]] = None
    umbrella_ready_value: Optional[bool] = None
    barrier_result = _apply_umbrella_barriers(
        data,
        sid=sid,
        timestamp=timestamp,
    )
    if barrier_result is not None:
        (
            barriers_payload,
            _merge_ready_state,
            _validation_ready_state,
            _review_ready_state,
            all_ready_state,
            _barrier_ts,
        ) = barrier_result
        barrier_event = dict(barriers_payload)
        umbrella_ready_value = bool(all_ready_state)
    else:
        existing_barriers_payload = data.get("umbrella_barriers")
        if isinstance(existing_barriers_payload, Mapping):
            barrier_event = dict(existing_barriers_payload)
        umbrella_ready_existing = data.get("umbrella_ready")
        if isinstance(umbrella_ready_existing, bool):
            umbrella_ready_value = umbrella_ready_existing

    if stage == "merge":
        merge_barrier_flag: Optional[bool] = None
        stage_merge_zero = stage_payload.get("merge_zero_packs")
        if isinstance(stage_merge_zero, (bool, int, float)) and not isinstance(stage_merge_zero, str):
            merge_barrier_flag = bool(stage_merge_zero)
        elif isinstance(summary_payload, Mapping):
            summary_merge_zero = summary_payload.get("merge_zero_packs")
            if isinstance(summary_merge_zero, (bool, int, float)) and not isinstance(summary_merge_zero, str):
                merge_barrier_flag = bool(summary_merge_zero)
        if merge_barrier_flag is None and isinstance(summary_payload, Mapping):
            summary_metrics_payload = summary_payload.get("metrics")
            if isinstance(summary_metrics_payload, Mapping):
                metrics_merge_zero = summary_metrics_payload.get("merge_zero_packs")
                if isinstance(metrics_merge_zero, (bool, int, float)) and not isinstance(metrics_merge_zero, str):
                    merge_barrier_flag = bool(metrics_merge_zero)
        if merge_barrier_flag is None:
            pairs_scored_val = _coerce_int(summary_payload.get("pairs_scored")) if isinstance(summary_payload, Mapping) else None
            if pairs_scored_val is None:
                pairs_scored_val = _coerce_int(stage_payload.get("pairs_scored"))
            packs_created_val = _coerce_int(summary_payload.get("packs_created")) if isinstance(summary_payload, Mapping) else None
            if packs_created_val is None:
                packs_created_val = _coerce_int(stage_payload.get("packs_created"))
            if (
                pairs_scored_val is not None
                and packs_created_val is not None
                and pairs_scored_val > 0
                and packs_created_val == 0
            ):
                merge_barrier_flag = True

        if merge_barrier_flag is not None:
            if barrier_event is None:
                barrier_event = {}
            barrier_event["merge_zero_packs"] = bool(merge_barrier_flag)
            existing_barriers_payload = data.get("umbrella_barriers")
            if isinstance(existing_barriers_payload, Mapping):
                existing_barriers_payload = dict(existing_barriers_payload)
                existing_barriers_payload["merge_zero_packs"] = bool(merge_barrier_flag)
                data["umbrella_barriers"] = existing_barriers_payload
            else:
                data["umbrella_barriers"] = {
                    "merge_zero_packs": bool(merge_barrier_flag),
                }

    _atomic_write_json(path, data)

    runflow_end_stage(
        sid,
        stage,
        status=final_stage_status,
        summary=summary_payload if summary_payload else None,
        stage_status=stage_status_override,
        empty_ok=empty_ok,
        barriers=barrier_event,
        umbrella_ready=umbrella_ready_value,
        refresh_barriers=False,
    )

    if refresh_barriers:
        runflow_refresh_umbrella_barriers(sid)

    return data


def _persist_runflow_data(
    sid: str,
    snapshot: Mapping[str, Any],
    *,
    runs_root: Optional[str | Path],
    last_writer: str,
) -> dict[str, Any]:
    path = _runflow_path(sid, runs_root)
    baseline = _load_runflow(path, sid)
    merged_snapshot = _merge_runflow_snapshots(baseline, dict(snapshot))
    merged_snapshot["snapshot_version"] = _next_snapshot_version(
        baseline.get("snapshot_version"), merged_snapshot.get("snapshot_version")
    )
    timestamp = _now_iso()
    merged_snapshot["last_writer"] = last_writer
    merged_snapshot["updated_at"] = timestamp
    _atomic_write_json(path, merged_snapshot)
    return merged_snapshot


def record_stage_force(
    sid: str,
    snapshot: Mapping[str, Any],
    *,
    runs_root: Optional[str | Path] = None,
    last_writer: str = "record_stage_force",
    refresh_barriers: bool = False,
) -> dict[str, Any]:
    merged_snapshot = _persist_runflow_data(
        sid,
        snapshot,
        runs_root=runs_root,
        last_writer=last_writer,
    )

    if refresh_barriers:
        try:
            reconcile_umbrella_barriers(sid, runs_root=runs_root)
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "RUNFLOW_FORCE_BARRIERS_FAILED sid=%s",
                sid,
                exc_info=True,
            )

    return merged_snapshot


def finalize_merge_stage(
    sid: str,
    *,
    runs_root: Optional[str | Path] = None,
    notes: Optional[str] = None,
) -> dict[str, Any]:
    """Mark the merge stage as complete using authoritative on-disk data."""

    base_root = _resolve_runs_root(runs_root)
    base_dir = base_root / sid
    merge_paths = ensure_merge_paths(base_root, sid, create=False)
    merge_dir = merge_paths.base

    runflow_path = base_root / sid / _RUNFLOW_FILENAME
    existing_runflow = _load_runflow(runflow_path, sid)
    previous_status = _stage_status(existing_runflow.get("stages"), "merge")

    manifest_index: Optional[Path] = None
    manifest_payload: Mapping[str, Any] | None = None
    manifest_path = base_dir / "manifest.json"
    try:
        manifest = RunManifest(manifest_path).load()
    except FileNotFoundError:
        manifest = None
    except Exception:  # pragma: no cover - defensive logging
        manifest = None
        log.debug(
            "RUNFLOW_MANIFEST_LOAD_FAILED sid=%s path=%s",
            sid,
            manifest_path,
            exc_info=True,
        )
    else:
        manifest_payload = manifest.data if isinstance(manifest.data, Mapping) else None
        try:
            manifest_index = manifest.get_ai_index_path()
        except Exception:  # pragma: no cover - defensive logging
            manifest_index = None
            log.debug(
                "RUNFLOW_MANIFEST_INDEX_RESOLVE_FAILED sid=%s path=%s",
                sid,
                manifest_path,
                exc_info=True,
            )

    if manifest_payload is None:
        manifest_payload = _load_json_mapping(manifest_path)

    preferred_index = _resolve_merge_index_path(sid, base_dir, manifest_payload)
    index_candidates: list[Path] = []
    candidate_set: set[Path] = set()

    def _append_candidate(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in candidate_set:
            candidate_set.add(resolved)
            index_candidates.append(resolved)

    _append_candidate(preferred_index)
    if isinstance(manifest_index, Path):
        _append_candidate(manifest_index)
    if merge_paths.index_file not in candidate_set:
        _append_candidate(merge_paths.index_file)

    index_path = index_candidates[-1]
    for candidate in index_candidates:
        if candidate.exists():
            index_path = candidate
            break

    verbose_logging = _env_enabled("RUNFLOW_VERBOSE", False)
    if verbose_logging:
        try:
            log.info(
                "MERGE_INDEX_RESOLVE sid=%s path=%s exists=%s",
                sid,
                str(index_path),
                index_path.exists(),
            )
        except Exception:  # pragma: no cover - defensive logging
            log.debug(
                "MERGE_INDEX_RESOLVE_LOG_FAILED sid=%s path=%s",
                sid,
                str(index_path),
                exc_info=True,
            )

    index_payload = _load_json_mapping(index_path)
    if not isinstance(index_payload, Mapping):
        legacy_index_path = merge_dir / "index.json"
        if legacy_index_path != index_path:
            legacy_payload = _load_json_mapping(legacy_index_path)
        else:
            legacy_payload = None
        if isinstance(legacy_payload, Mapping):
            index_payload = legacy_payload
        else:
            index_payload = {}

    totals_payload = index_payload.get("totals")
    totals = dict(totals_payload) if isinstance(totals_payload, Mapping) else {}

    def _maybe_int(value: Any) -> Optional[int]:
        coerced = _coerce_int(value)
        return coerced if coerced is not None else None

    def _maybe_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return None

    totals_pairs_scored = _maybe_int(totals.get("scored_pairs"))
    totals_created_packs = _maybe_int(totals.get("created_packs"))

    skip_counts_payload = totals.get("skip_counts")
    skip_counts_override: dict[str, int] = {}
    if isinstance(skip_counts_payload, Mapping):
        for k, v in skip_counts_payload.items():
            coerced = _coerce_int(v)
            if coerced is None:
                continue
            skip_counts_override[str(k)] = coerced

    skip_reason_top_value = totals.get("skip_reason_top")
    skip_reason_top_text: Optional[str]
    if isinstance(skip_reason_top_value, str) and skip_reason_top_value.strip():
        skip_reason_top_text = skip_reason_top_value.strip()
    else:
        skip_reason_top_text = None

    reason_candidate = totals.get("reason")
    merge_reason_text: Optional[str]
    if isinstance(reason_candidate, str) and reason_candidate.strip():
        merge_reason_text = reason_candidate.strip()
    else:
        merge_reason_text = None

    merge_zero_override: Optional[bool] = _maybe_bool(totals.get("merge_zero_packs"))

    merge_zero_totals_flag = merge_zero_override is True
    zero_pack_success = (
        merge_zero_totals_flag
        and totals_created_packs == 0
        and totals_pairs_scored is not None
        and totals_pairs_scored > 0
    )

    pairs_payload = index_payload.get("pairs")
    pairs_count = len(pairs_payload) if isinstance(pairs_payload, Sequence) else None

    metrics: dict[str, Any] = {}
    for key in (
        "scored_pairs",
        "matches_strong",
        "matches_weak",
        "conflicts",
        "skipped",
        "packs_built",
        "created_packs",
        "topn_limit",
        "normalized_accounts",
    ):
        coerced = _maybe_int(totals.get(key))
        if coerced is not None:
            metrics[key] = coerced

    fallback_scored = _maybe_int(index_payload.get("scored_pairs"))
    if fallback_scored is not None:
        metrics.setdefault("scored_pairs", fallback_scored)

    fallback_created = _maybe_int(index_payload.get("created_packs"))
    if fallback_created is not None:
        metrics.setdefault("created_packs", fallback_created)

    if totals_pairs_scored is not None:
        metrics["pairs_scored"] = totals_pairs_scored
        metrics["scored_pairs"] = totals_pairs_scored
    else:
        metrics.setdefault("pairs_scored", metrics.get("scored_pairs", 0))
        metrics.setdefault("scored_pairs", metrics.get("pairs_scored", 0))

    if totals_created_packs is not None:
        metrics["created_packs"] = totals_created_packs
    else:
        metrics.setdefault("created_packs", metrics.get("packs_built", fallback_created or 0))

    if isinstance(skip_counts_payload, Mapping):
        metrics["skip_counts"] = dict(skip_counts_override)

    if skip_reason_top_text is not None:
        metrics["skip_reason_top"] = skip_reason_top_text

    if merge_reason_text is not None:
        metrics["merge_reason"] = merge_reason_text
        metrics["reason"] = merge_reason_text

    if merge_zero_override is not None:
        metrics["merge_zero_packs"] = bool(merge_zero_override)

    if pairs_count is not None:
        metrics["pairs_index_entries"] = pairs_count

    results_dir = merge_paths.results_dir
    result_glob = merge_result_glob_pattern()
    try:
        result_files_total = sum(
            1 for path in results_dir.rglob(result_glob) if path.is_file()
        )
    except OSError:
        result_files_total = 0

    if result_files_total == 0:
        fallback_pattern = "pair-*.result.json"
        try:
            result_files_total = sum(
                1 for path in results_dir.glob(fallback_pattern) if path.is_file()
            )
        except OSError:
            pass

    if result_files_total == 0:
        fallback_pattern = "pair-*.result.json"
        try:
            result_files_total = sum(
                1 for path in results_dir.glob(fallback_pattern) if path.is_file()
            )
        except OSError:
            pass

    if result_files_total == 0:
        fallback_pattern = "pair-*.result.json"
        try:
            result_files_total = sum(
                1 for path in results_dir.glob(fallback_pattern) if path.is_file()
            )
        except OSError:
            pass

    packs_dir = merge_paths.packs_dir
    pack_glob = config.MERGE_PACK_GLOB or "pair_*.jsonl"
    try:
        pack_files_total = sum(
            1 for path in packs_dir.glob(pack_glob) if path.is_file()
        )
    except OSError:
        pack_files_total = 0

    metrics["result_files"] = result_files_total
    metrics["pack_files"] = pack_files_total

    scored_pairs_value = _maybe_int(metrics.get("pairs_scored"))
    if scored_pairs_value is None:
        scored_pairs_value = _maybe_int(metrics.get("scored_pairs"))
    if scored_pairs_value is None:
        scored_pairs_value = 0
    metrics["pairs_scored"] = scored_pairs_value
    metrics.setdefault("scored_pairs", scored_pairs_value)

    expected_candidates: list[int] = []
    if totals_created_packs is not None:
        expected_candidates.append(totals_created_packs)
    packs_built_total = _maybe_int(totals.get("packs_built"))
    if packs_built_total is not None:
        expected_candidates.append(packs_built_total)
    total_packs_total = _maybe_int(totals.get("total_packs"))
    if total_packs_total is not None:
        expected_candidates.append(total_packs_total)
    if fallback_created is not None:
        expected_candidates.append(fallback_created)
    # NOTE: Do NOT use pairs_count for expected calculation.
    # pairs_count represents bidirectional lookup entries (e.g., [7,10] and [10,7]),
    # not physical pack files. Using it causes expected=2 when only 1 result exists.
    # Rely on created_packs/packs_built which correctly reflect physical files.

    expected_total: Optional[int]
    if expected_candidates:
        expected_total = max(expected_candidates)
    else:
        expected_total = None

    if zero_pack_success:
        expected_total = 0
        ready_counts_match = True
    else:
        ready_counts_match = result_files_total == pack_files_total
        if expected_total is not None:
            ready_counts_match = ready_counts_match and result_files_total == expected_total

    if not ready_counts_match:
        raise RuntimeError(
            "merge stage artifacts not ready: results=%s packs=%s expected=%s"
            % (result_files_total, pack_files_total, expected_total)
        )

    created_packs_value = (
        totals_created_packs
        if totals_created_packs is not None
        else result_files_total
    )
    metrics["created_packs"] = created_packs_value

    counts: dict[str, int] = {
        "pairs_scored": scored_pairs_value,
        "packs_created": created_packs_value,
        "result_files": result_files_total,
    }

    empty_ok = zero_pack_success or created_packs_value == 0 or scored_pairs_value == 0

    results_payload = {"result_files": result_files_total}

    merge_zero_flag = merge_zero_override
    if merge_zero_flag is None:
        merge_zero_flag = created_packs_value == 0 and scored_pairs_value > 0
    metrics["merge_zero_packs"] = bool(merge_zero_flag)

    if isinstance(skip_counts_payload, Mapping) and not metrics.get("skip_counts"):
        metrics["skip_counts"] = dict(skip_counts_override)

    record_stage(
        sid,
        "merge",
        status="success",
        counts=counts,
        empty_ok=empty_ok,
        metrics=metrics,
        results=results_payload,
        runs_root=base_root,
        notes=notes,
        refresh_barriers=False,
    )

    # ── MERGE_AI_APPLIED FLAG ──────────────────────────────────────────────
    # Mark that merge AI results have been fully applied to runflow.
    # This flag is used by _compute_umbrella_barriers to ensure validation
    # cannot start before merge AI decisions are applied (non-zero-packs case).
    # Zero-packs fast path is unaffected (empty_ok check takes precedence).
    runflow_data = _load_runflow(runflow_path, sid)
    if runflow_data is not None:
        stages = _ensure_stages_dict(runflow_data)
        merge_stage = stages.get("merge")
        if isinstance(merge_stage, dict):
            merge_stage["merge_ai_applied"] = True
            merge_stage["merge_ai_applied_at"] = _now_iso()
            _atomic_write_json(runflow_path, runflow_data)
            log.info("MERGE_AI_APPLIED sid=%s", sid)
        else:
            log.warning("MERGE_AI_APPLIED_SKIP sid=%s reason=merge_stage_not_dict", sid)
    else:
        log.warning("MERGE_AI_APPLIED_SKIP sid=%s reason=runflow_not_found", sid)
    # ───────────────────────────────────────────────────────────────────────

    if _RUNFLOW_EMIT_ZERO_PACKS_STEP and metrics.get("merge_zero_packs"):
        runflow_step(
            sid,
            "merge",
            "merge_zero_packs",
            status="success",
            out={
                "skip_reason_top": metrics.get("skip_reason_top"),
                "skip_counts": metrics.get("skip_counts", {}),
            },
        )

    if previous_status != "success":
        log.info("MERGE_STAGE_PROMOTED sid=%s result_files=%s", sid, result_files_total)

    runflow_refresh_umbrella_barriers(sid)

    # ── VALIDATION CHAIN TRIGGER ───────────────────────────────────────────
    # After merge finalization and barrier refresh, attempt to trigger validation chain.
    # The helper will check merge_ready and only trigger if validation is needed.
    try:
        from backend.pipeline.auto_ai import maybe_trigger_validation_chain_if_merge_ready
        maybe_trigger_validation_chain_if_merge_ready(
            sid,
            runs_root=base_root,
            flag_env=None,  # Will use os.environ by default
        )
    except Exception:
        log.error("VALIDATION_CHAIN_POST_MERGE_TRIGGER_FAILED sid=%s", sid, exc_info=True)
    # ───────────────────────────────────────────────────────────────────────

    return {
        "counts": counts,
        "metrics": metrics,
        "results": results_payload,
        "empty_ok": empty_ok,
    }


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("RUNFLOW_JSON_READ_FAILED path=%s", path, exc_info=True)
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("RUNFLOW_JSON_PARSE_FAILED path=%s", path, exc_info=True)
        return None

    if isinstance(payload, Mapping):
        return payload

    return None


def _get_stage_info(
    stages: Mapping[str, Any] | None, stage: str
) -> Mapping[str, Any] | None:
    if not isinstance(stages, Mapping):
        return None
    candidate = stages.get(stage)
    if isinstance(candidate, Mapping):
        return candidate
    return None


def _stage_has_counters(stage_info: Mapping[str, Any] | None) -> bool:
    if not isinstance(stage_info, Mapping):
        return False

    skip_keys = {
        "status",
        "empty_ok",
        "last_at",
        "notes",
        "metrics",
        "results",
        "error",
        "summary",
    }

    for key, value in stage_info.items():
        if key in skip_keys:
            continue
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return True
    return False


def _stage_metrics_value(stage_info: Mapping[str, Any] | None, key: str) -> Optional[int]:
    if not isinstance(stage_info, Mapping):
        return None
    metrics = stage_info.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    return _coerce_int(metrics.get(key))


def _stage_status(steps: Mapping[str, Any] | None, stage: str) -> str:
    if not isinstance(steps, Mapping):
        return ""

    stage_info = steps.get(stage)
    if not isinstance(stage_info, Mapping):
        return ""

    status = stage_info.get("status")
    if isinstance(status, str):
        return status.strip().lower()

    return ""


def _stage_empty_ok(stage_info: Mapping[str, Any] | None) -> bool:
    """Return True if stage declares empty_ok truthy."""
    if not isinstance(stage_info, Mapping):
        return False
    raw = stage_info.get("empty_ok")
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return bool(raw)
    return False


def _ensure_stages_dict(data: Mapping[str, Any] | dict[str, Any]) -> dict[str, Any]:
    if isinstance(data, dict):
        stages = data.get("stages")
    else:
        stages = data.get("stages") if isinstance(data, Mapping) else None

    if isinstance(stages, dict):
        return stages

    if isinstance(stages, Mapping):
        stage_mapping = dict(stages)
    else:
        stage_mapping = {}

    if isinstance(data, dict):
        data["stages"] = stage_mapping
    return stage_mapping


def _apply_merge_stage_promotion(
    data: dict[str, Any], run_dir: Path
) -> tuple[bool, bool, dict[str, int]]:
    (
        result_files_total,
        pack_files_total,
        expected_total,
        ready,
    ) = _merge_artifacts_progress(run_dir)

    log_context = {"result_files": result_files_total}
    if not ready:
        return (False, False, log_context)

    stages = _ensure_stages_dict(data)
    existing = stages.get("merge") if isinstance(stages, Mapping) else None
    stage_payload = dict(existing) if isinstance(existing, Mapping) else {}

    previous_status = _stage_status(stages, "merge")

    stage_payload["status"] = "success"
    stage_payload["last_at"] = _now_iso()

    empty_ok = result_files_total == 0 or expected_total == 0
    stage_payload["empty_ok"] = bool(empty_ok)

    stage_payload["result_files"] = result_files_total
    stage_payload["pack_files"] = pack_files_total

    if expected_total is not None:
        stage_payload["expected_packs"] = expected_total

    summary_payload = stage_payload.get("summary")
    if isinstance(summary_payload, Mapping):
        summary = dict(summary_payload)
    else:
        summary = {}

    summary.update(
        {
            "result_files": result_files_total,
            "pack_files": pack_files_total,
            "empty_ok": bool(empty_ok),
        }
    )

    if expected_total is not None:
        summary["expected_packs"] = expected_total

    disk_counts = _stage_counts_from_disk("merge", run_dir)
    for key, value in disk_counts.items():
        coerced = _coerce_int(value)
        if coerced is None:
            continue
        stage_payload[str(key)] = coerced
        summary[str(key)] = coerced

    results_payload = stage_payload.get("results")
    if isinstance(results_payload, Mapping):
        results = dict(results_payload)
    else:
        results = {}
    results["result_files"] = result_files_total
    stage_payload["results"] = results

    metrics_payload = stage_payload.get("metrics")
    if isinstance(metrics_payload, Mapping):
        merge_zero_candidate = metrics_payload.get("merge_zero_packs")
        if merge_zero_candidate is not None:
            merge_zero_value = bool(merge_zero_candidate)
            stage_payload["merge_zero_packs"] = merge_zero_value
            summary["merge_zero_packs"] = merge_zero_value
        skip_reason_candidate = metrics_payload.get("skip_reason_top")
        if isinstance(skip_reason_candidate, str) and skip_reason_candidate.strip():
            normalized_reason = skip_reason_candidate.strip()
            stage_payload.setdefault("skip_reason_top", normalized_reason)
            summary.setdefault("skip_reason_top", normalized_reason)
        skip_counts_candidate = metrics_payload.get("skip_counts")
        if isinstance(skip_counts_candidate, Mapping):
            normalized_skip_counts: dict[str, int] = {}
            for skip_key, skip_val in skip_counts_candidate.items():
                if not isinstance(skip_key, str):
                    continue
                coerced_skip = _coerce_int(skip_val)
                if coerced_skip is None:
                    continue
                normalized_skip_counts[skip_key] = coerced_skip
            if normalized_skip_counts:
                stage_payload.setdefault("skip_counts", dict(normalized_skip_counts))
                existing_summary_skip_counts = summary.get("skip_counts")
                if isinstance(existing_summary_skip_counts, Mapping):
                    merged_counts = dict(existing_summary_skip_counts)
                    merged_counts.update(normalized_skip_counts)
                    summary["skip_counts"] = merged_counts
                else:
                    summary["skip_counts"] = dict(normalized_skip_counts)

    stage_payload["summary"] = summary

    stages["merge"] = stage_payload

    promoted = previous_status != "success"
    return (True, promoted, log_context)


def _apply_validation_stage_promotion(
    data: dict[str, Any], run_dir: Path
) -> tuple[bool, bool, dict[str, int]]:
    total, completed, failed, ready = _validation_results_progress(run_dir)
    log_context = {"total": total, "completed": completed, "failed": failed}

    if not ready or completed != total:
        return (False, False, log_context)

    stages = _ensure_stages_dict(data)
    existing = stages.get("validation") if isinstance(stages, Mapping) else None
    stage_payload = dict(existing) if isinstance(existing, Mapping) else {}
    # Mark this snapshot as coming from validation promotion for authoritative merge
    stage_payload["_writer"] = "validation_promotion"

    previous_status = _stage_status(stages, "validation")

    # Core validation status
    stage_payload["status"] = "success"
    stage_payload["last_at"] = _now_iso()
    stage_payload["empty_ok"] = bool(total == 0)
    stage_payload["sent"] = True
    
    if previous_status != "success":
        stage_payload["ready_latched"] = True
        stage_payload["ready_latched_at"] = _now_iso()

    # Results block - validation-specific
    results_payload = {
        "results_total": total,
        "completed": completed,
        "failed": failed,
    }
    stage_payload["results"] = results_payload

    # Get findings_count from disk
    disk_counts = _stage_counts_from_disk("validation", run_dir)
    findings_count = 0
    for key, value in disk_counts.items():
        if key == "findings_count":
            coerced = _coerce_int(value)
            if coerced is not None:
                findings_count = coerced
                stage_payload["findings_count"] = findings_count

    # Read manifest for V2 validation flags
    manifest_path = run_dir / "manifest.json"
    validation_ai_required = False
    validation_ai_applied = False
    validation_ai_completed = False
    packs_total = 0
    accounts_eligible = 0
    packs_skipped = 0
    
    try:
        manifest = RunManifest(manifest_path).load()
        manifest_data = manifest.data if hasattr(manifest, 'data') else {}
        if isinstance(manifest_data, Mapping):
            ai_section = manifest_data.get("ai", {})
            if isinstance(ai_section, Mapping):
                status_section = ai_section.get("status", {})
                if isinstance(status_section, Mapping):
                    validation_status = status_section.get("validation", {})
                    if isinstance(validation_status, Mapping):
                        validation_ai_required = bool(validation_status.get("validation_ai_required", False))
                        validation_ai_applied = bool(validation_status.get("validation_ai_applied", False))
                        validation_ai_completed = bool(validation_status.get("completed", False))
                        
                        results_total = validation_status.get("results_total")
                        if isinstance(results_total, (int, float)) and not isinstance(results_total, bool):
                            packs_total = int(results_total)
    except Exception:
        log.debug(
            "VALIDATION_STAGE_MANIFEST_READ_FAILED sid=%s path=%s",
            run_dir.name,
            manifest_path,
            exc_info=True,
        )
    # Fallback: if manifest did not declare requirement but existing metrics had it set
    existing_metrics_snapshot = stage_payload.get("metrics")
    if (not validation_ai_required) and isinstance(existing_metrics_snapshot, Mapping):
        if bool(existing_metrics_snapshot.get("validation_ai_required")):
            validation_ai_required = True
    
    # Build metrics - validation-specific only, no merge fields
    missing_results = max(0, total - completed - failed)
    metrics_data: dict[str, Any] = {
        "packs_total": packs_total if packs_total > 0 else total,
        "results_total": total,
        "missing_results": missing_results,
        "accounts_eligible": accounts_eligible,
        "packs_skipped": packs_skipped,
        "validation_ai_required": validation_ai_required,
        "validation_ai_completed": validation_ai_completed or (total > 0 and completed == total),
        "validation_ai_applied": validation_ai_applied,
    }
    
    # Preserve selected counters from existing metrics but intentionally drop merge pollution
    existing_metrics = stage_payload.get("metrics")
    if isinstance(existing_metrics, Mapping):
        if "accounts_eligible" in existing_metrics:
            accounts_eligible = _coerce_int(existing_metrics.get("accounts_eligible")) or 0
            metrics_data["accounts_eligible"] = accounts_eligible
        if "packs_skipped" in existing_metrics:
            packs_skipped = _coerce_int(existing_metrics.get("packs_skipped")) or 0
            metrics_data["packs_skipped"] = packs_skipped
    
    stage_payload["metrics"] = metrics_data

    # Build clean summary
    summary: dict[str, Any] = {
        "findings_count": findings_count,
        "results_total": total,
        "completed": completed,
        "failed": failed,
        "missing_results": missing_results,
        "empty_ok": bool(total == 0),
        "results": dict(results_payload),
        "validation_ai_required": validation_ai_required,
        "validation_ai_completed": metrics_data["validation_ai_completed"],
        "validation_ai_applied": validation_ai_applied,
    }
    
    stage_payload["summary"] = summary
    
    # Add top-level validation_ai_completed for backward compat
    stage_payload["validation_ai_completed"] = metrics_data["validation_ai_completed"]

    # Clean up any existing merge pollution (top-level, metrics, summary)
    for merge_key in ["merge_zero_packs", "skip_counts", "skip_reason_top", "merge_context"]:
        stage_payload.pop(merge_key, None)
    # Ensure metrics and summary are free of pollution
    for container_key in ["metrics", "summary"]:
        container = stage_payload.get(container_key)
        if isinstance(container, Mapping):
            for merge_key in ["merge_zero_packs", "skip_counts", "skip_reason_top", "merge_context"]:
                if merge_key in container:
                    try:
                        del container[merge_key]
                    except Exception:
                        pass

    stages["validation"] = stage_payload

    promoted = previous_status != "success"
    return (True, promoted, log_context)


def _apply_note_style_stage_promotion(
    data: dict[str, Any],
    run_dir: Path,
    *,
    results_override: tuple[int, int, int] | None = None,
    allow_partial_success: bool = False,
) -> tuple[bool, bool, dict[str, int]]:
    snapshot = _note_style_snapshot_for_run(run_dir)

    from backend.ai.note_style.io import note_style_stage_view

    view = note_style_stage_view(
        run_dir.name,
        runs_root=run_dir.parent,
        snapshot=snapshot,
    )

    total_value = view.total_expected
    built_value = view.built_total
    completed_value = view.completed_total
    failed_value = view.failed_total
    terminal_value = view.terminal_total
    empty_ok = total_value == 0
    if total_value == 0:
        status_value = "empty"
        stage_terminal = True
    elif total_value > 0 and completed_value == 0 and failed_value == 0:
        status_value = "built"
        stage_terminal = False
    elif completed_value + failed_value < total_value:
        status_value = "processing"
        stage_terminal = False
    elif completed_value == total_value:
        status_value = "success"
        stage_terminal = True
    elif failed_value > 0:
        status_value = "error"
        stage_terminal = False
    else:
        status_value = "processing"
        stage_terminal = False

    if results_override is not None:
        override_total, override_completed, override_failed = results_override
        override_total = max(override_total, 0)
        override_completed = max(min(override_completed, override_total), 0)
        override_failed = max(min(override_failed, override_total - override_completed), 0)
        total_value = max(total_value, override_total)
        completed_value = max(completed_value, override_completed)
        failed_value = max(failed_value, override_failed)
        terminal_value = min(completed_value + failed_value, total_value)

    stages = _ensure_stages_dict(data)
    existing = stages.get("note_style") if isinstance(stages, Mapping) else None
    previous_status = _stage_status(stages, "note_style")

    metrics_payload = {"packs_total": total_value}
    results_payload = {
        "results_total": total_value,
        "completed": completed_value,
        "failed": failed_value,
    }
    summary_payload = {
        "packs_total": total_value,
        "results_total": total_value,
        "completed": completed_value,
        "failed": failed_value,
        "empty_ok": empty_ok,
        "metrics": dict(metrics_payload),
        "results": dict(results_payload),
    }

    def _normalize_metrics(mapping: Any) -> dict[str, int]:
        if not isinstance(mapping, Mapping):
            return {}
        value = _coerce_int(mapping.get("packs_total"))
        if value is None:
            return {}
        return {"packs_total": value}

    def _normalize_results(mapping: Any) -> dict[str, int]:
        if not isinstance(mapping, Mapping):
            return {}
        normalized: dict[str, int] = {}
        for key in ("results_total", "completed", "failed"):
            value = _coerce_int(mapping.get(key))
            if value is not None:
                normalized[key] = value
        return normalized

    def _normalize_summary(summary: Any) -> dict[str, Any]:
        if not isinstance(summary, Mapping):
            return {}
        normalized: dict[str, Any] = {}
        empty_value = summary.get("empty_ok")
        if isinstance(empty_value, bool):
            normalized["empty_ok"] = empty_value
        elif empty_value is not None:
            normalized["empty_ok"] = bool(empty_value)
        metrics_component = _normalize_metrics(summary.get("metrics"))
        if metrics_component:
            normalized["metrics"] = metrics_component
        results_component = _normalize_results(summary.get("results"))
        if results_component:
            normalized["results"] = results_component
        for key in ("packs_total", "results_total", "completed", "failed"):
            value = _coerce_int(summary.get(key))
            if value is not None:
                normalized[key] = value
        return normalized

    existing_metrics = _normalize_metrics(existing.get("metrics")) if isinstance(existing, Mapping) else {}
    existing_results = _normalize_results(existing.get("results")) if isinstance(existing, Mapping) else {}
    existing_summary = _normalize_summary(existing.get("summary")) if isinstance(existing, Mapping) else {}
    existing_empty_ok = bool(existing.get("empty_ok")) if isinstance(existing, Mapping) else False
    existing_sent = bool(existing.get("sent")) if isinstance(existing, Mapping) else False
    if isinstance(existing, Mapping):
        completed_raw = existing.get("completed_at")
    else:
        completed_raw = None
    if isinstance(completed_raw, str):
        existing_completed_at = completed_raw.strip() or None
    else:
        existing_completed_at = None
    existing_status = (
        _normalize_stage_status_value(existing.get("status"))
        if isinstance(existing, Mapping)
        else ""
    )

    desired_sent = status_value in {"success", "empty"}
    completed_matches = (
        (stage_terminal and existing_completed_at is not None)
        or (not stage_terminal and existing_completed_at is None)
    )

    update_required = True
    if isinstance(existing, Mapping):
        if (
            existing_status == status_value
            and existing_empty_ok == empty_ok
            and existing_metrics == metrics_payload
            and existing_results == results_payload
            and existing_summary == summary_payload
            and existing_sent == desired_sent
            and completed_matches
        ):
            update_required = False

    promoted = (
        status_value in {"success", "empty", "error"}
        and previous_status != status_value
    )
    log_context = {
        "total": total_value,
        "completed": completed_value,
        "failed": failed_value,
    }

    if not update_required:
        return (False, promoted, log_context)

    timestamp = _now_iso()
    sent_value = desired_sent
    if sent_value:
        completed_at_value = existing_completed_at or timestamp
    else:
        completed_at_value = None
    stage_payload = {
        "status": status_value,
        "last_at": timestamp,
        "empty_ok": empty_ok,
        "metrics": dict(metrics_payload),
        "results": dict(results_payload),
        "summary": summary_payload,
        "sent": sent_value,
        "completed_at": completed_at_value,
    }

    _log_note_style_decision(
        "NOTE_STYLE_STAGE_STATUS_WRITE",
        logger=log,
        sid=run_dir.name,
        runs_root=run_dir.parent,
        reason="apply_note_style_stage_promotion",
        decided_status=status_value,
        packs_expected=total_value,
        packs_built=built_value,
        packs_completed=completed_value,
        packs_failed=failed_value,
    )

    stages["note_style"] = stage_payload
    return (True, promoted, log_context)


def _note_style_result_status_from_payload(payload: Mapping[str, Any]) -> str | None:
    status_value = payload.get("status")
    if isinstance(status_value, str):
        normalized = status_value.strip().lower()
        if normalized in {"completed", "success"}:
            return "completed"
        if normalized in {"failed", "error"}:
            return "failed"

    if payload.get("error") not in (None, "", {}):
        return "failed"

    analysis_payload = payload.get("analysis")
    if isinstance(analysis_payload, Mapping) and analysis_payload:
        return "completed"

    return None


def _parse_note_style_result_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None

    for line in raw.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return payload
    return None


def _infer_note_style_result_status(path: Path) -> str | None:
    name = path.name.lower()
    if name.endswith(".failed.jsonl") or name.endswith(".failed.json"):
        return "failed"
    if name.endswith(".error.jsonl") or name.endswith(".error.json"):
        return "failed"
    if name.endswith(".result.jsonl") or name.endswith(".result.json"):
        payload = _parse_note_style_result_payload(path)
        if payload is None:
            return None
        status = _note_style_result_status_from_payload(payload)
        if status is not None:
            return status
        return None

    payload = _parse_note_style_result_payload(path)
    if payload is None:
        return None
    return _note_style_result_status_from_payload(payload)


def _iter_note_style_result_files(results_dir: Path) -> Iterable[Path]:
    try:
        entries = list(results_dir.iterdir())
    except FileNotFoundError:
        return []
    except NotADirectoryError:
        return []

    return [
        path
        for path in entries
        if path.is_file()
        and not path.name.startswith(".")
        and not path.name.endswith(".tmp")
    ]


def _normalize_note_style_result_account(name: str) -> str:
    text = name
    if text.endswith(".jsonl"):
        text = text[: -len(".jsonl")]
    if text.endswith(".json"):
        text = text[: -len(".json")]
    if text.endswith(".result"):
        text = text[: -len(".result")]
    if text.startswith("acc_"):
        text = text[len("acc_") :]
    return text


def _note_style_index_status_mapping(run_dir: Path) -> dict[str, str]:
    try:
        paths = ensure_note_style_paths(run_dir.parent, run_dir.name, create=False)
        index_path = paths.index_file
    except Exception:
        index_path = (run_dir / config.NOTE_STYLE_STAGE_DIR / "index.json").resolve()
    document = _load_json_mapping(index_path)
    if not isinstance(document, Mapping):
        return {}

    entries: Sequence[Mapping[str, Any]] = ()
    packs_payload = document.get("packs")
    if isinstance(packs_payload, Sequence):
        entries = [entry for entry in packs_payload if isinstance(entry, Mapping)]
    else:
        items_payload = document.get("items")
        if isinstance(items_payload, Sequence):
            entries = [entry for entry in items_payload if isinstance(entry, Mapping)]

    statuses: dict[str, str] = {}
    for entry in entries:
        account_value = entry.get("account_id")
        if not isinstance(account_value, str):
            continue
        statuses[account_value.strip()] = str(entry.get("status") or "")

    return statuses


def _note_style_counts_from_results_dir(run_dir: Path) -> Tuple[int, int, int]:
    snapshot = _note_style_snapshot_for_run(run_dir)

    packs_expected = set(snapshot.packs_expected)
    total = len(packs_expected)
    completed = len(packs_expected & set(snapshot.packs_completed))
    failed = len(packs_expected & set(snapshot.packs_failed))

    return (total, completed, failed)


def refresh_note_style_stage_from_results(
    sid: str, runs_root: Optional[str | Path] = None
) -> Tuple[int, int, int]:
    """Update the note_style stage entry based on stored results."""

    path = _runflow_path(sid, runs_root)
    run_dir = path.parent
    total, completed, failed = _note_style_counts_from_results_dir(run_dir)
    data = _load_runflow(path, sid)

    updated, promoted, log_context = _apply_note_style_stage_promotion(
        data,
        run_dir,
        results_override=(total, completed, failed),
        allow_partial_success=True,
    )
    if not updated:
        return (total, completed, failed)

    latest_snapshot = _load_runflow(path, sid)
    data = _merge_runflow_snapshots(latest_snapshot, data)

    timestamp = _now_iso()
    data["snapshot_version"] = _next_snapshot_version(
        latest_snapshot.get("snapshot_version"), data.get("snapshot_version")
    )
    data["last_writer"] = "refresh_note_style_stage_from_results"
    data["updated_at"] = timestamp

    _atomic_write_json(path, data)

    if _runflow_events_enabled():
        stage_info: Mapping[str, Any] | None = None
        stages_payload = data.get("stages")
        if isinstance(stages_payload, Mapping):
            candidate = stages_payload.get("note_style")
            if isinstance(candidate, Mapping):
                stage_info = candidate

        event_ts = timestamp
        status_text = ""
        empty_ok_flag = False
        metrics_event: dict[str, int] = {}
        results_event: dict[str, int] = {}

        if stage_info is not None:
            last_at_value = stage_info.get("last_at")
            if isinstance(last_at_value, str) and last_at_value:
                event_ts = last_at_value
            status_value = stage_info.get("status")
            if isinstance(status_value, str):
                status_text = status_value
            empty_ok_flag = bool(stage_info.get("empty_ok"))

            metrics_payload = stage_info.get("metrics")
            if isinstance(metrics_payload, Mapping):
                packs_total_value = _coerce_int(metrics_payload.get("packs_total"))
                if packs_total_value is not None:
                    metrics_event["packs_total"] = packs_total_value

            results_payload = stage_info.get("results")
            if isinstance(results_payload, Mapping):
                for key in ("results_total", "completed", "failed"):
                    value = _coerce_int(results_payload.get(key))
                    if value is not None:
                        results_event[key] = value

        events_path = run_dir / "runflow_events.jsonl"
        event_payload: dict[str, Any] = {
            "ts": event_ts,
            "event": "note_style_stage_refresh",
            "sid": sid,
            "stage": "note_style",
            "status": status_text,
            "empty_ok": empty_ok_flag,
        }
        if metrics_event:
            event_payload["metrics"] = dict(metrics_event)
            packs_total_value = metrics_event.get("packs_total")
            if packs_total_value is not None:
                event_payload["packs_total"] = packs_total_value
        if results_event:
            event_payload["results"] = dict(results_event)
            results_total_value = results_event.get("results_total")
            if results_total_value is not None:
                event_payload["results_total"] = results_total_value
            completed_value = results_event.get("completed")
            if completed_value is not None:
                event_payload["results_completed"] = completed_value
            failed_value = results_event.get("failed")
            if failed_value is not None:
                event_payload["results_failed"] = failed_value

        try:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event_payload, ensure_ascii=False))
                handle.write("\n")
        except OSError:
            log.warning(
                "NOTE_STYLE_STAGE_EVENT_WRITE_FAILED sid=%s path=%s",
                sid,
                events_path,
                exc_info=True,
            )

    if promoted:
        log.info(
            "NOTE_STYLE_STAGE_PROMOTED sid=%s total=%s completed=%s failed=%s",
            sid,
            log_context["total"],
            log_context["completed"],
            log_context["failed"],
        )

    runflow_refresh_umbrella_barriers(sid)

    return (total, completed, failed)


def _apply_frontend_stage_promotion(
    data: dict[str, Any], run_dir: Path
) -> tuple[bool, bool, dict[str, int]]:
    required, answered, ready = _frontend_responses_progress(run_dir)
    empty_ok = required == 0
    answers_recorded = 0 if empty_ok else answered
    if empty_ok:
        ready = True

    log_context = {
        "answers_required": required,
        "answers_received": answers_recorded,
    }

    if not ready:
        return (False, False, log_context)

    if not empty_ok and answered != required:
        return (False, False, log_context)

    stages = _ensure_stages_dict(data)
    existing = stages.get("frontend") if isinstance(stages, Mapping) else None
    stage_payload = dict(existing) if isinstance(existing, Mapping) else {}

    previous_status = _stage_status(stages, "frontend")

    stage_payload["status"] = "success"
    stage_payload["last_at"] = _now_iso()
    stage_payload["empty_ok"] = bool(empty_ok)

    metrics_payload = stage_payload.get("metrics")
    if isinstance(metrics_payload, Mapping):
        metrics_data = dict(metrics_payload)
    else:
        metrics_data = {}
    metrics_data["answers_received"] = answers_recorded
    metrics_data["answers_required"] = required
    stage_payload["metrics"] = metrics_data
    stage_payload.pop("answers", None)

    packs_counts = _stage_counts_from_disk("frontend", run_dir)
    packs_count_value = _coerce_int(stage_payload.get("packs_count"))
    disk_packs = _coerce_int(packs_counts.get("packs_count")) if packs_counts else None
    if disk_packs is not None:
        packs_count_value = disk_packs
    if packs_count_value is not None:
        stage_payload["packs_count"] = packs_count_value

    summary_payload = stage_payload.get("summary")
    if isinstance(summary_payload, Mapping):
        summary = dict(summary_payload)
    else:
        summary = {}

    summary.update(
        {
            "answers_received": answers_recorded,
            "answers_required": required,
            "empty_ok": bool(empty_ok),
        }
    )

    if packs_count_value is not None:
        summary["packs_count"] = packs_count_value

    if metrics_data:
        summary["metrics"] = dict(metrics_data)

    stage_payload["summary"] = summary

    stages["frontend"] = stage_payload

    promoted = previous_status != "success"
    return (True, promoted, log_context)


def _resolve_validation_index_path(run_dir: Path) -> Path:
    override = os.getenv("VALIDATION_INDEX_PATH")
    if override:
        candidate = Path(override)
        if not candidate.is_absolute():
            candidate = (run_dir / override).resolve()
        else:
            candidate = candidate.resolve()
        return candidate

    return (run_dir / "ai_packs" / "validation" / "index.json").resolve()


def _resolve_validation_results_dir(run_dir: Path) -> Optional[Path]:
    override = os.getenv("VALIDATION_RESULTS_DIR")
    if not override:
        return None

    candidate = Path(override)
    if not candidate.is_absolute():
        candidate = (run_dir / override).resolve()
    else:
        candidate = candidate.resolve()
    return candidate


def _validation_record_result_paths(
    index: "ValidationIndex", record: Any, *, results_override: Optional[Path] = None
) -> list[Path]:  # pragma: no cover - exercised via higher level tests
    paths: list[Path] = []
    seen: set[str] = set()

    def _append(path: Path) -> None:
        key = str(path)
        if key not in seen:
            seen.add(key)
            paths.append(path)

    result_json_value = getattr(record, "result_json", None)
    if isinstance(result_json_value, str) and result_json_value.strip():
        try:
            _append(index.resolve_result_json_path(record))
        except Exception:
            return []

    result_jsonl_value = getattr(record, "result_jsonl", None)
    if isinstance(result_jsonl_value, str) and result_jsonl_value.strip():
        try:
            _append(index.resolve_result_jsonl_path(record))
        except Exception:
            return []

    if results_override:
        base_dir: Optional[Path]
        try:
            base_dir = index.results_dir_path
        except Exception:  # pragma: no cover - defensive
            base_dir = None

        for candidate in list(paths):
            relative: Optional[Path] = None
            if base_dir is not None:
                try:
                    relative = candidate.resolve().relative_to(base_dir)
                except ValueError:
                    relative = None
            if relative is None:
                relative = Path(candidate.name)

            override_candidate = (results_override / relative).resolve()
            _append(override_candidate)

    return paths


def _validation_record_has_results(
    index: "ValidationIndex", record: Any, *, results_override: Optional[Path] = None
) -> bool:
    paths = _validation_record_result_paths(
        index, record, results_override=results_override
    )
    if not paths:
        return False

    for candidate in paths:
        try:
            if candidate.is_file():
                return True
        except OSError:
            continue
    return False


def _validation_results_progress(run_dir: Path) -> tuple[int, int, int, bool]:
    """Return (total, completed, failed, ready) for validation AI results."""

    index_path = _resolve_validation_index_path(run_dir)
    results_override = _resolve_validation_results_dir(run_dir)
    try:
        index = load_validation_index(index_path)
    except FileNotFoundError:
        return (0, 0, 0, False)
    except Exception:  # pragma: no cover - defensive
        log.warning("RUNFLOW_VALIDATION_INDEX_LOAD_FAILED path=%s", index_path, exc_info=True)
        return (0, 0, 0, False)

    total = 0
    completed = 0
    failed = 0
    ready = True

    for record in getattr(index, "packs", ()):  # type: ignore[attr-defined]
        total += 1
        status_value = getattr(record, "status", None)
        normalized_status = _normalize_terminal_status(
            status_value, stage="validation", run_dir=run_dir
        )

        if normalized_status == "failed":
            failed += 1
            ready = False
            continue

        if normalized_status != "completed":
            ready = False
            continue

        if not _validation_record_has_results(
            index, record, results_override=results_override
        ):
            log.warning(
                "RUNFLOW_VALIDATION_RESULT_MISSING account_id=%s",
                getattr(record, "account_id", "unknown"),
            )
            ready = False
            continue

        completed += 1

    if total == 0:
        return (0, 0, 0, True)

    ready = ready and completed == total
    return (total, completed, failed, ready)


def _note_style_results_progress(run_dir: Path) -> tuple[int, int, int, bool]:
    """Return (total, completed, failed, ready) for note_style results."""
    snapshot = _note_style_snapshot_for_run(run_dir)

    packs_expected = set(snapshot.packs_expected)
    total = len(packs_expected)
    completed = len(packs_expected & set(snapshot.packs_completed))
    failed = len(packs_expected & set(snapshot.packs_failed))
    terminal = completed + failed
    ready = total == 0 or terminal >= total

    return (total, completed, failed, ready)


def _validation_stage_ready(
    run_dir: Path, validation_stage: Mapping[str, Any] | None
) -> bool:
    total, completed, failed, ready = _validation_results_progress(run_dir)

    if total > 0:
        base_ready = ready and completed == total
        
        # V2 invariant: If validation_ai_applied flag exists and is False, block readiness
        # Only enforce this if the flag is explicitly present (V2 runs)
        if base_ready and isinstance(validation_stage, Mapping):
            validation_ai_required = False
            validation_ai_applied = None  # None = not set, not checking
            
            # Check summary first (preferred location)
            summary = validation_stage.get("summary")
            if isinstance(summary, Mapping):
                validation_ai_required = bool(summary.get("validation_ai_required", False))
                if "validation_ai_applied" in summary:
                    validation_ai_applied = bool(summary.get("validation_ai_applied"))
            
            # Fallback to metrics if not in summary
            if validation_ai_applied is None:
                metrics = validation_stage.get("metrics")
                if isinstance(metrics, Mapping):
                    if not validation_ai_required:
                        validation_ai_required = bool(metrics.get("validation_ai_required", False))
                    if "validation_ai_applied" in metrics:
                        validation_ai_applied = bool(metrics.get("validation_ai_applied"))
            
            # Only block if validation_ai_applied is explicitly False (V2 run with apply pending)
            # If the flag is not set (None), allow readiness (legacy or pre-apply state)
            if validation_ai_required and validation_ai_applied is False:
                return False
        
        return base_ready

    if ready:
        return True

    if not isinstance(validation_stage, Mapping):
        return False

    findings_count = _coerce_int(validation_stage.get("findings_count"))
    if findings_count is None or findings_count != 0:
        return False

    packs_total = _stage_metrics_value(validation_stage, "packs_total")
    if packs_total is None:
        return False

    status_raw = validation_stage.get("status")
    status_normalized = str(status_raw).strip().lower() if isinstance(status_raw, str) else ""
    sent_flag = bool(validation_stage.get("sent"))

    if _stage_merge_zero_flag_for_sid(run_dir, validation_stage=validation_stage):
        if status_normalized in {"success", "completed"}:
            return True
        if sent_flag:
            last_sender_started = _last_validation_sender_event_timestamp(
                run_dir, "validation.sender.started"
            )
            if last_sender_started is not None:
                age_seconds = (datetime.now(timezone.utc) - last_sender_started).total_seconds()
                if age_seconds <= _VALIDATION_FASTPATH_WATCHDOG_EVENT_SECONDS:
                    return True

    if not _validation_autosend_enabled():
        return packs_total == 0

    return packs_total == 0


_IDX_ACCOUNT_PATTERN = re.compile(r"idx-(\d+)")


def _response_filename_for_account(account_id: str) -> str:
    trimmed = (account_id or "").strip()
    match = _IDX_ACCOUNT_PATTERN.fullmatch(trimmed)
    number: int | None = None
    if match:
        number = int(match.group(1))
    else:
        try:
            number = int(trimmed)
        except ValueError:
            number = None

    if number is not None:
        return f"idx-{number:03d}.result.json"

    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", trimmed) or "account"
    return f"{sanitized}.result.json"


def _frontend_responses_progress(run_dir: Path) -> tuple[int, int, bool]:
    attachments_required = _review_attachment_required()
    counters = _frontend_answers_counters(
        run_dir, attachments_required=attachments_required
    )

    required = _coerce_int(counters.get("answers_required")) or 0
    answered_accounts_raw = counters.get("answered_accounts")

    answered_accounts: set[str] = set()
    if isinstance(answered_accounts_raw, Sequence) and not isinstance(
        answered_accounts_raw, (str, bytes, bytearray)
    ):
        for entry in answered_accounts_raw:
            if isinstance(entry, (str, int)):
                text = str(entry).strip()
                if text:
                    answered_accounts.add(text)

    config = load_frontend_stage_config(run_dir)
    responses_dir = config.responses_dir

    account_status_map: dict[str, str] = {}

    try:
        response_paths = sorted(
            path
            for path in responses_dir.iterdir()
            if path.is_file() and path.name.endswith(".result.json")
        )
    except OSError:
        response_paths = []

    for path in response_paths:
        payload = _load_json_mapping(path)
        if not isinstance(payload, Mapping):
            continue

        answers = payload.get("answers")
        if not isinstance(answers, Mapping):
            continue

        explanation = answers.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            continue

        if attachments_required and not _frontend_has_review_attachments(answers):
            continue

        received_at = payload.get("received_at")
        if not isinstance(received_at, str) or not received_at.strip():
            continue

        account_id = payload.get("account_id")
        if isinstance(account_id, str) and account_id.strip():
            account_key = account_id.strip()
        else:
            account_key = path.stem

        normalized_status = _normalize_terminal_status(
            payload.get("status"), stage="frontend", run_dir=run_dir
        )

        if normalized_status == "failed":
            account_status_map[account_key] = "failed"
            continue

        if normalized_status == "completed":
            account_status_map[account_key] = "completed"
            continue

    answered = len(account_status_map)
    completed_count = sum(
        1 for status in account_status_map.values() if status == "completed"
    )
    ready = answered >= required and completed_count == answered

    return (required, answered, ready)


def _merge_artifacts_progress(
    run_dir: Path,
) -> tuple[int, int, Optional[int], bool]:
    merge_paths = ensure_merge_paths(run_dir.parent, run_dir.name, create=False)
    results_dir = merge_paths.results_dir
    packs_dir = merge_paths.packs_dir
    result_glob = merge_result_glob_pattern()
    pack_glob = config.MERGE_PACK_GLOB or "pair_*.jsonl"

    try:
        result_files_total = sum(
            1 for path in results_dir.rglob(result_glob) if path.is_file()
        )
    except OSError:
        result_files_total = 0

    if result_files_total == 0:
        fallback_pattern = "pair-*.result.json"
        try:
            result_files_total = sum(
                1 for path in results_dir.glob(fallback_pattern) if path.is_file()
            )
        except OSError:
            pass

    try:
        pack_files_total = sum(
            1 for path in packs_dir.glob(pack_glob) if path.is_file()
        )
    except OSError:
        pack_files_total = 0

    expected_total: Optional[int] = None
    index_payload = _load_json_mapping(merge_paths.index_file)
    if isinstance(index_payload, Mapping):
        totals_payload = index_payload.get("totals")
        if isinstance(totals_payload, Mapping):
            candidates = [
                _coerce_int(totals_payload.get(key))
                for key in ("created_packs", "packs_built", "total_packs")
            ]
            expected_candidates = [value for value in candidates if value is not None]
            if expected_candidates:
                expected_total = max(expected_candidates)

        if expected_total is None:
            fallback_candidates = [
                _coerce_int(index_payload.get(key))
                for key in ("created_packs", "packs_built", "pack_count")
            ]
            fallback_values = [value for value in fallback_candidates if value is not None]
            if fallback_values:
                expected_total = max(fallback_values)

        # NOTE: Do NOT use len(pairs_payload) as fallback for expected_total.
        # pairs array contains bidirectional entries (e.g., [7,10] and [10,7]),
        # so its length is 2x the physical pack count. This causes false failures.
        # If no expected count is available from totals, leave expected_total=None
        # and rely on result_files==pack_files check only.

    ready = False

    if expected_total == 0:
        ready = True
    else:
        ready = result_files_total == pack_files_total
        if expected_total is not None:
            ready = ready and result_files_total == expected_total

    if (
        not ready
        and result_files_total == 0
        and pack_files_total == 0
        and expected_total is None
        and not _merge_required()
    ):
        ready = True

    return (result_files_total, pack_files_total, expected_total, ready)


def _compute_umbrella_barriers(
    run_dir: Path, runflow_payload: Mapping[str, Any] | None = None
) -> dict[str, bool]:
    if runflow_payload is None:
        runflow_path = run_dir / "runflow.json"
        runflow_payload = _load_json_mapping(runflow_path)

    stages_payload = (
        runflow_payload.get("stages")
        if isinstance(runflow_payload, Mapping)
        else None
    )

    if isinstance(stages_payload, Mapping):
        stages = stages_payload
    else:
        stages = {}

    def _flag_from_value(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return bool(value)
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        return None

    def _stage_mapping(stage_name: str) -> Mapping[str, Any] | None:
        candidate = stages.get(stage_name)
        if isinstance(candidate, Mapping):
            return candidate
        return None

    def _stage_status_success(stage_info: Mapping[str, Any] | None) -> bool:
        if not isinstance(stage_info, Mapping):
            return False
        status_value = stage_info.get("status")
        if not isinstance(status_value, str):
            return False
        return status_value.strip().lower() == "success"

    def _summary_value(stage_info: Mapping[str, Any], key: str) -> Optional[int]:
        summary = stage_info.get("summary")
        if isinstance(summary, Mapping):
            value = _coerce_int(summary.get(key))
            if value is not None:
                return value
        return None

    merge_stage = _stage_mapping("merge")
    has_merge_stage = isinstance(merge_stage, Mapping)
    merge_empty_ok = _stage_empty_ok(merge_stage)
    merge_stage_result_files: Optional[int] = None
    merge_ready = False
    if _stage_status_success(merge_stage):
        result_files = _summary_value(merge_stage, "result_files")
        if result_files is None and isinstance(merge_stage, Mapping):
            result_files = _coerce_int(merge_stage.get("result_files"))
        if result_files is None and merge_empty_ok:
            result_files = 0
        merge_stage_result_files = result_files
        if merge_empty_ok:
            merge_ready = True
        elif result_files is not None:
            merge_ready = result_files >= 1

    ( 
        merge_disk_result_files,
        _merge_disk_pack_files,
        _merge_expected,
        merge_ready_disk,
    ) = _merge_artifacts_progress(run_dir)
    if merge_ready_disk:
        merge_ready = True
    elif merge_ready:
        if merge_stage_result_files is None:
            merge_ready = merge_empty_ok and merge_disk_result_files == 0
        else:
            if merge_disk_result_files == 0 and merge_stage_result_files > 0:
                merge_ready = True
            else:
                merge_ready = merge_stage_result_files == merge_disk_result_files

    if not _umbrella_require_merge():
        if merge_stage_result_files is None:
            merge_result_files_for_policy = merge_disk_result_files
        else:
            merge_result_files_for_policy = merge_stage_result_files

        reason: Optional[str] = None
        if not has_merge_stage:
            reason = "no_merge_stage"
        elif merge_result_files_for_policy == 0:
            reason = "empty_merge_results"

        if reason is not None:
            log.info(
                "UMBRELLA_MERGE_OPTIONAL sid=%s reason=%s was_ready=%s merge_files=%s require_merge=0",
                run_dir.name,
                reason,
                merge_ready,
                merge_result_files_for_policy,
            )
            merge_ready = True

    # ── MERGE_AI_APPLIED CHECK ─────────────────────────────────────────────
    # For non-zero-packs cases, require merge_ai_applied flag to be True.
    # This ensures validation cannot start before merge AI results are applied.
    # Zero-packs fast path is unaffected (merge_empty_ok takes precedence).
    if merge_ready and not merge_empty_ok:
        merge_ai_applied = merge_stage.get("merge_ai_applied", False) if isinstance(merge_stage, Mapping) else False
        if not merge_ai_applied:
            log.info(
                "MERGE_NOT_AI_APPLIED sid=%s merge_ready_disk=%s merge_empty_ok=%s",
                run_dir.name,
                merge_ready_disk,
                merge_empty_ok,
            )
            merge_ready = False
    # ───────────────────────────────────────────────────────────────────────

    validation_stage = _stage_mapping("validation")
    validation_total, validation_completed, _validation_failed, validation_ready_disk = (
        _validation_results_progress(run_dir)
    )
    if validation_total > 0:
        validation_ready_disk = validation_ready_disk and (
            validation_completed == validation_total
        )
    validation_ready = validation_ready_disk
    if not validation_ready and _stage_status_success(validation_stage):
        results_payload = validation_stage.get("results")
        if isinstance(results_payload, Mapping):
            completed = _coerce_int(results_payload.get("completed"))
            total = _coerce_int(results_payload.get("results_total"))
            if completed is not None and total is not None and completed == total:
                validation_ready = True
        if not validation_ready and _stage_empty_ok(validation_stage):
            validation_ready = True

    frontend_stage = _stage_mapping("frontend")
    note_style_stage = _stage_mapping("note_style")
    review_required, review_received, review_ready_disk = _frontend_responses_progress(
        run_dir
    )
    has_frontend_stage = isinstance(frontend_stage, Mapping)
    review_disk_evidence = review_required > 0 or review_received > 0
    review_ready = False
    if has_frontend_stage or review_disk_evidence:
        review_ready = review_ready_disk and review_received >= review_required
    if not review_ready and _stage_status_success(frontend_stage):
        metrics_payload = frontend_stage.get("metrics")
        if isinstance(metrics_payload, Mapping):
            required = _coerce_int(metrics_payload.get("answers_required"))
            received = _coerce_int(metrics_payload.get("answers_received"))
            if required is not None and received is not None and received == required:
                review_ready = True
        if not review_ready and _stage_empty_ok(frontend_stage):
            review_ready = True

    style_total, style_completed, style_failed, style_ready_disk = (
        _note_style_results_progress(run_dir)
    )

    # Determine if the note_style stage has actually started (artifacts exist)
    note_style_base = run_dir / "ai_packs" / "note_style"
    index_exists = (note_style_base / "index.json").exists()
    packs_dir = note_style_base / "packs"
    results_dir = note_style_base / "results"
    packs_present = False
    results_present = False
    try:
        packs_present = any(p.is_file() for p in packs_dir.iterdir())
    except Exception:
        packs_present = False
    try:
        results_present = any(p.is_file() for p in results_dir.iterdir())
    except Exception:
        results_present = False
    artifacts_present = (
        index_exists
        or packs_present
        or results_present
    )

    if not artifacts_present:
        # If style hasn't started, treat as ready when frontend stage exists
        style_ready = bool(has_frontend_stage)
    else:
        style_ready = style_ready_disk
        if not style_ready:
            if _stage_status_success(note_style_stage):
                style_ready = True
            elif _stage_empty_ok(note_style_stage):
                style_ready = True

    all_ready_base = merge_ready and validation_ready and review_ready
    if _umbrella_require_style():
        all_ready = all_ready_base and style_ready
    else:
        all_ready = all_ready_base

    merge_zero_packs_flag = False
    if _UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG:
        merge_zero_candidate: Optional[bool] = None
        if isinstance(merge_stage, Mapping):
            containers: list[Mapping[str, Any]] = [merge_stage]
            metrics_payload = merge_stage.get("metrics")
            if isinstance(metrics_payload, Mapping):
                containers.append(metrics_payload)
            summary_payload = merge_stage.get("summary")
            if isinstance(summary_payload, Mapping):
                containers.append(summary_payload)
                summary_metrics_payload = summary_payload.get("metrics")
                if isinstance(summary_metrics_payload, Mapping):
                    containers.append(summary_metrics_payload)
            for container in containers:
                candidate = _flag_from_value(container.get("merge_zero_packs"))
                if candidate is not None:
                    merge_zero_candidate = candidate
                    break

            if merge_zero_candidate is None:
                summary_payload = merge_stage.get("summary")
                scored_val: Optional[int] = None
                created_val: Optional[int] = None
                if isinstance(summary_payload, Mapping):
                    scored_val = _coerce_int(summary_payload.get("pairs_scored"))
                    created_val = _coerce_int(summary_payload.get("packs_created"))
                    if created_val is None:
                        created_val = _coerce_int(summary_payload.get("created_packs"))
                metrics_payload = merge_stage.get("metrics")
                if isinstance(metrics_payload, Mapping):
                    if scored_val is None:
                        scored_val = _coerce_int(metrics_payload.get("pairs_scored"))
                        if scored_val is None:
                            scored_val = _coerce_int(metrics_payload.get("scored_pairs"))
                    if created_val is None:
                        created_val = _coerce_int(metrics_payload.get("created_packs"))
                        if created_val is None:
                            created_val = _coerce_int(metrics_payload.get("packs_created"))
                if (
                    created_val is not None
                    and scored_val is not None
                    and created_val == 0
                    and scored_val > 0
                ):
                    merge_zero_candidate = True

        merge_zero_packs_flag = bool(merge_zero_candidate)

    # Strategy barrier derivation (Phase 3)
    strategy_stage = stages.get("strategy") if isinstance(stages, Mapping) else None
    strategy_status_success = _stage_status_success(strategy_stage)
    strategy_status = str(strategy_stage.get("status")).strip().lower() if isinstance(strategy_stage, Mapping) and isinstance(strategy_stage.get("status"), str) else ""
    # Collect validation flags
    validation_containers: list[Mapping[str, Any]] = []
    if isinstance(validation_stage, Mapping):
        validation_containers.append(validation_stage)
        vs_metrics = validation_stage.get("metrics")
        if isinstance(vs_metrics, Mapping):
            validation_containers.append(vs_metrics)
        vs_summary = validation_stage.get("summary")
        if isinstance(vs_summary, Mapping):
            validation_containers.append(vs_summary)
            summary_metrics = vs_summary.get("metrics")
            if isinstance(summary_metrics, Mapping):
                validation_containers.append(summary_metrics)
    def _v_flag(name: str) -> Optional[bool]:
        for c in validation_containers:
            if not isinstance(c, Mapping):
                continue
            val = c.get(name)
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return bool(val)
        return None
    def _v_int(name: str) -> int:
        for c in validation_containers:
            v = _coerce_int(c.get(name))
            if v is not None:
                return v
        return 0
    v_required = bool(_v_flag("validation_ai_required"))
    v_completed = bool(_v_flag("validation_ai_completed"))
    v_applied = bool(_v_flag("validation_ai_applied"))
    packs_total_val = _v_int("packs_total")
    merge_results_applied = False
    if isinstance(validation_stage, Mapping):
        merge_results_block = validation_stage.get("merge_results")
        if isinstance(merge_results_block, Mapping):
            for key in ("applied","merge_results_applied"):
                flag = merge_results_block.get(key)
                if isinstance(flag, bool):
                    merge_results_applied = flag
                    break
    strategy_required = False
    # Determine if strategy is required: presence of strategy stage or eligible findings/accounts
    findings_total = _v_int("findings_count")
    accounts_eligible = _v_int("accounts_eligible") or _v_int("eligible_accounts")
    if isinstance(strategy_stage, Mapping):
        if strategy_status in {"built","success","error"}:
            strategy_required = True
    if findings_total > 0 or accounts_eligible > 0:
        strategy_required = True
    strategy_ready = False
    if not strategy_required:
        strategy_ready = True
    elif strategy_status == "success":
        strategy_ready = True
    elif strategy_status == "error":
        strategy_ready = False
    else:
        if validation_ready and merge_results_applied:
            if not v_required:
                strategy_ready = True
            else:
                # Tightened semantics: require both completed & applied
                if packs_total_val > 0:
                    strategy_ready = v_completed and v_applied
                else:
                    strategy_ready = v_completed  # zero-pack / legacy path

    readiness: dict[str, bool] = {
        "merge_ready": merge_ready,
        "validation_ready": validation_ready,
        "strategy_ready": strategy_ready,
        "strategy_required": strategy_required,
        "review_ready": review_ready,
        "style_ready": style_ready,
        "all_ready": all_ready and strategy_ready,
    }

    if _UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG:
        readiness["merge_zero_packs"] = merge_zero_packs_flag

    if _document_verifier_enabled():
        readiness["document_ready"] = False

    return readiness


def refresh_validation_stage_from_index(
    sid: str, runs_root: Optional[str | Path] = None
) -> None:
    """Update the validation stage entry when AI results are complete."""

    path = _runflow_path(sid, runs_root)
    run_dir = path.parent
    data = _load_runflow(path, sid)

    updated, promoted, log_context = _apply_validation_stage_promotion(data, run_dir)
    if not updated:
        return

    latest_snapshot = _load_runflow(path, sid)
    data = _merge_runflow_snapshots(latest_snapshot, data)

    timestamp = _now_iso()
    data["snapshot_version"] = _next_snapshot_version(
        latest_snapshot.get("snapshot_version"), data.get("snapshot_version")
    )
    data["last_writer"] = "refresh_validation_stage"
    data["updated_at"] = timestamp

    _atomic_write_json(path, data)

    if promoted:
        log.info(
            "VALIDATION_STAGE_PROMOTED sid=%s total=%s completed=%s failed=%s",
            sid,
            log_context["total"],
            log_context["completed"],
            log_context["failed"],
        )

    runflow_refresh_umbrella_barriers(sid)


def refresh_note_style_stage_from_index(
    sid: str, runs_root: Optional[str | Path] = None
) -> None:
    """Update the note_style stage entry when AI results are complete."""

    path = _runflow_path(sid, runs_root)
    run_dir = path.parent
    data = _load_runflow(path, sid)

    updated, promoted, log_context = _apply_note_style_stage_promotion(data, run_dir)
    if not updated:
        return

    latest_snapshot = _load_runflow(path, sid)
    data = _merge_runflow_snapshots(latest_snapshot, data)

    timestamp = _now_iso()
    data["snapshot_version"] = _next_snapshot_version(
        latest_snapshot.get("snapshot_version"), data.get("snapshot_version")
    )
    data["last_writer"] = "refresh_note_style_stage"
    data["updated_at"] = timestamp

    _atomic_write_json(path, data)

    if _runflow_events_enabled():
        stage_info: Mapping[str, Any] | None = None
        stages_payload = data.get("stages")
        if isinstance(stages_payload, Mapping):
            candidate = stages_payload.get("note_style")
            if isinstance(candidate, Mapping):
                stage_info = candidate

        event_ts = timestamp
        status_text = ""
        empty_ok_flag = False
        metrics_event: dict[str, int] = {}
        results_event: dict[str, int] = {}

        if stage_info is not None:
            last_at_value = stage_info.get("last_at")
            if isinstance(last_at_value, str) and last_at_value:
                event_ts = last_at_value
            status_value = stage_info.get("status")
            if isinstance(status_value, str):
                status_text = status_value
            empty_ok_flag = bool(stage_info.get("empty_ok"))

            metrics_payload = stage_info.get("metrics")
            if isinstance(metrics_payload, Mapping):
                packs_total_value = _coerce_int(metrics_payload.get("packs_total"))
                if packs_total_value is not None:
                    metrics_event["packs_total"] = packs_total_value

            results_payload = stage_info.get("results")
            if isinstance(results_payload, Mapping):
                for key in ("results_total", "completed", "failed"):
                    value = _coerce_int(results_payload.get(key))
                    if value is not None:
                        results_event[key] = value

        events_path = run_dir / "runflow_events.jsonl"
        event_payload: dict[str, Any] = {
            "ts": event_ts,
            "event": "note_style_stage_refresh",
            "sid": sid,
            "stage": "note_style",
            "status": status_text,
            "empty_ok": empty_ok_flag,
        }
        if metrics_event:
            event_payload["metrics"] = dict(metrics_event)
            packs_total_value = metrics_event.get("packs_total")
            if packs_total_value is not None:
                event_payload["packs_total"] = packs_total_value
        if results_event:
            event_payload["results"] = dict(results_event)
            results_total_value = results_event.get("results_total")
            if results_total_value is not None:
                event_payload["results_total"] = results_total_value
            completed_value = results_event.get("completed")
            if completed_value is not None:
                event_payload["results_completed"] = completed_value
            failed_value = results_event.get("failed")
            if failed_value is not None:
                event_payload["results_failed"] = failed_value

        try:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event_payload, ensure_ascii=False))
                handle.write("\n")
        except OSError:
            log.warning(
                "NOTE_STYLE_STAGE_EVENT_WRITE_FAILED sid=%s path=%s",
                sid,
                events_path,
                exc_info=True,
            )

    if promoted:
        log.info(
            "NOTE_STYLE_STAGE_PROMOTED sid=%s total=%s completed=%s failed=%s",
            sid,
            log_context["total"],
            log_context["completed"],
            log_context["failed"],
        )

    runflow_refresh_umbrella_barriers(sid)


def refresh_frontend_stage_from_responses(
    sid: str, runs_root: Optional[str | Path] = None
) -> None:
    """Update the frontend stage entry when customer responses are complete."""

    path = _runflow_path(sid, runs_root)
    run_dir = path.parent
    data = _load_runflow(path, sid)

    updated, promoted, log_context = _apply_frontend_stage_promotion(data, run_dir)
    if not updated:
        return

    latest_snapshot = _load_runflow(path, sid)
    data = _merge_runflow_snapshots(latest_snapshot, data)

    timestamp = _now_iso()
    data["snapshot_version"] = _next_snapshot_version(
        latest_snapshot.get("snapshot_version"), data.get("snapshot_version")
    )
    data["last_writer"] = "refresh_frontend_stage"
    data["updated_at"] = timestamp

    _atomic_write_json(path, data)

    if promoted:
        log.info(
            "FRONTEND_STAGE_PROMOTED sid=%s answers_required=%s answers_received=%s",
            sid,
            log_context["answers_required"],
            log_context["answers_received"],
        )

    runflow_refresh_umbrella_barriers(sid)


def reconcile_umbrella_barriers(
    sid: str, runs_root: Optional[str | Path] = None
) -> dict[str, bool]:
    """Recompute umbrella readiness for ``sid`` and persist the booleans."""

    runflow_path = _runflow_path(sid, runs_root)
    run_dir = runflow_path.parent
    data = _load_runflow(runflow_path, sid)

    _merge_updated, merge_promoted, merge_log = _apply_merge_stage_promotion(data, run_dir)
    if merge_promoted:
        log.info(
            "MERGE_STAGE_PROMOTED sid=%s result_files=%s",
            sid,
            merge_log["result_files"],
        )

    _validation_updated, validation_promoted, validation_log = _apply_validation_stage_promotion(
        data, run_dir
    )
    if validation_promoted:
        log.info(
            "VALIDATION_STAGE_PROMOTED sid=%s total=%s completed=%s failed=%s",
            sid,
            validation_log["total"],
            validation_log["completed"],
            validation_log["failed"],
        )

    _frontend_updated, frontend_promoted, frontend_log = _apply_frontend_stage_promotion(
        data, run_dir
    )
    if frontend_promoted:
        log.info(
            "FRONTEND_STAGE_PROMOTED sid=%s answers_required=%s answers_received=%s",
            sid,
            frontend_log["answers_required"],
            frontend_log["answers_received"],
        )

    _note_style_updated, note_style_promoted, note_style_log = _apply_note_style_stage_promotion(
        data, run_dir
    )
    if note_style_promoted:
        log.info(
            "NOTE_STYLE_STAGE_PROMOTED sid=%s total=%s completed=%s failed=%s",
            sid,
            note_style_log["total"],
            note_style_log["completed"],
            note_style_log["failed"],
        )

    latest_snapshot = _load_runflow(runflow_path, sid)
    data = _merge_runflow_snapshots(latest_snapshot, data)

    if "umbrella_barriers" not in data or not isinstance(data["umbrella_barriers"], Mapping):
        data["umbrella_barriers"] = _default_umbrella_barriers()

    _ensure_merge_zero_pack_metadata(sid, run_dir, data)

    _maybe_enqueue_validation_fastpath(sid, run_dir, data)

    watchdog_context: Optional[dict[str, Any]] = _validation_stuck_fastpath(sid, run_dir, data)
    if watchdog_context is not None:
        _watchdog_trigger_validation_fastpath(sid, run_dir, data, watchdog_context)

    data["snapshot_version"] = _next_snapshot_version(
        latest_snapshot.get("snapshot_version"), data.get("snapshot_version")
    )
    data["last_writer"] = "reconcile_umbrella_barriers"

    statuses = _compute_umbrella_barriers(run_dir, runflow_payload=data)

    existing = data.get("umbrella_barriers")
    if isinstance(existing, Mapping):
        umbrella = dict(existing)
    else:
        umbrella = {}

    timestamp = _now_iso()
    for key, value in statuses.items():
        if isinstance(key, str):
            umbrella[key] = bool(value)
    umbrella["checked_at"] = timestamp

    data["umbrella_barriers"] = umbrella
    data["umbrella_ready"] = bool(statuses.get("all_ready"))
    data["updated_at"] = timestamp

    _atomic_write_json(runflow_path, data)

    # ── VALIDATION CHAIN TRIGGER (EXTERNAL) ────────────────────────────────
    # Trigger validation chain when merge_ready=true and validation not complete.
    # This is the critical external trigger that runs outside the chain itself.
    merge_ready = bool(statuses.get("merge_ready", False))
    validation_ready = bool(statuses.get("validation_ready", False))
    
    if merge_ready and not validation_ready:
        try:
            from backend.pipeline.auto_ai import maybe_trigger_validation_chain_if_merge_ready
            
            trigger_result = maybe_trigger_validation_chain_if_merge_ready(
                sid,
                runs_root=run_dir.parent if run_dir else None,
                flag_env=None,
            )
            
            if trigger_result.get("triggered"):
                log.info(
                    "VALIDATION_CHAIN_EXTERNAL_TRIGGER_SUCCESS sid=%s merge_ready=%s validation_ready=%s",
                    sid, merge_ready, validation_ready
                )
        except Exception:
            log.error(
                "VALIDATION_CHAIN_EXTERNAL_TRIGGER_FAILED sid=%s",
                sid,
                exc_info=True
            )
    # ───────────────────────────────────────────────────────────────────────

    try:
        from backend.runflow.umbrella import schedule_merge_autosend

        schedule_merge_autosend(sid, run_dir=run_dir)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "MERGE_AUTOSEND_STAGE_FAILED sid=%s", sid, exc_info=True
        )

    try:
        from backend.runflow.umbrella import schedule_note_style_after_validation

        schedule_note_style_after_validation(sid, run_dir=run_dir)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "NOTE_STYLE_AUTOSEND_AFTER_VALIDATION_FAILED sid=%s", sid, exc_info=True
        )

    if _barrier_event_logging_enabled():
        events_path = run_dir / "runflow_events.jsonl"
        event_payload = {"ts": timestamp, "event": "barriers_reconciled"}
        for key, value in statuses.items():
            if isinstance(key, str):
                event_payload[key] = bool(value)
        try:
            events_path.parent.mkdir(parents=True, exist_ok=True)
            with events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event_payload, ensure_ascii=False))
                handle.write("\n")
        except OSError:
            log.warning(
                "RUNFLOW_BARRIERS_EVENT_WRITE_FAILED sid=%s path=%s",
                sid,
                events_path,
                exc_info=True,
            )

    return statuses


def evaluate_global_barriers(run_path: str) -> dict[str, bool]:
    """Inspect run artifacts and report readiness for umbrella arguments."""

    if not _umbrella_barriers_enabled():
        return {
            "merge_ready": False,
            "validation_ready": False,
            "review_ready": False,
            "style_ready": False,
            "all_ready": False,
        }

    run_dir = Path(run_path)
    return _compute_umbrella_barriers(run_dir)


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _latest_stage_name(stages: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
    stage_order = {"merge": 0, "validation": 1, "frontend": 2}
    best_name: Optional[str] = None
    best_ts: Optional[datetime] = None

    for name, info in stages.items():
        if not isinstance(info, Mapping):
            continue

        ts = (
            _parse_timestamp(info.get("last_at"))
            or _parse_timestamp(info.get("ended_at"))
            or _parse_timestamp(info.get("started_at"))
        )

        if ts is None:
            continue

        if best_ts is None or ts > best_ts or (
            ts == best_ts
            and stage_order.get(str(name), -1) >= stage_order.get(best_name or "", -1)
        ):
            best_name = str(name)
            best_ts = ts

    if best_name is not None:
        return best_name

    for candidate in ("frontend", "validation", "merge"):
        if candidate in stages:
            return candidate

    for name in stages:
        if isinstance(name, str):
            return name

    return None


def _decision_next_label(next_action: str) -> str:
    mapping = {
        "run_validation": "continue",
        "gen_frontend_packs": "run_frontend",
        "await_input": "await_input",
        "complete_no_action": "done",
        "stop_error": "done",
    }
    return mapping.get(next_action, "continue")


def decide_next(sid: str, runs_root: Optional[str | Path] = None) -> dict[str, str]:
    """Return the next pipeline action for ``sid`` based on recorded stages."""

    path = _runflow_path(sid, runs_root)
    data = _load_runflow(path, sid)
    stages: Mapping[str, Mapping[str, Any]] = data.get("stages", {})  # type: ignore[assignment]
    if not isinstance(stages, dict):
        stages = dict(stages)
        data["stages"] = stages  # type: ignore[assignment]

    run_dir = path.parent

    next_action = "run_validation"
    reason = "validation_pending"
    new_state: RunState = "VALIDATING"
    skip_merge_wait = False
    decision_metadata: dict[str, Any] = {}
    stages_mutated = False

    def _set(next_value: str, reason_value: str, state_value: RunState) -> None:
        nonlocal next_action, reason, new_state
        next_action = next_value
        reason = reason_value
        new_state = state_value

    def _normalize_skip_counts(payload: Optional[Mapping[str, Any]]) -> dict[str, int]:
        normalized: dict[str, int] = {}
        if not isinstance(payload, Mapping):
            return normalized
        for key, value in payload.items():
            if not isinstance(key, str):
                continue
            coerced = _coerce_int(value)
            if coerced is None:
                continue
            normalized[key] = coerced
        return normalized

    merge_stage: Mapping[str, Any] = {}
    has_merge_stage = False

    for stage_name, stage_info in stages.items():
        if not isinstance(stage_info, Mapping):
            continue
        status = str(stage_info.get("status") or "")
        if status == "error":
            _set("stop_error", f"{stage_name}_error", "ERROR")
            break
    else:
        merge_stage_payload = stages.get("merge")
        has_merge_stage = isinstance(merge_stage_payload, Mapping)
        if has_merge_stage:
            merge_stage = merge_stage_payload  # type: ignore[assignment]
        else:
            merge_stage = {}

        accounts_count: Optional[int] = None
        if has_merge_stage:
            for key in ("count", "accounts_count", "total_accounts"):
                accounts_count = _coerce_int(merge_stage.get(key))
                if accounts_count is not None:
                    break

        metrics_payload = merge_stage.get("metrics") if has_merge_stage else None
        skip_counts_metadata: dict[str, int] = {}

        if isinstance(metrics_payload, Mapping):
            skip_reason_top = metrics_payload.get("skip_reason_top")
            if isinstance(skip_reason_top, str) and skip_reason_top.strip():
                decision_metadata["skip_reason_top"] = skip_reason_top.strip()
            skip_counts_metadata = _normalize_skip_counts(metrics_payload.get("skip_counts"))
        else:
            skip_reason_top = None

        summary_payload = merge_stage.get("summary") if has_merge_stage else None
        if skip_reason_top is None and isinstance(summary_payload, Mapping):
            summary_metrics = summary_payload.get("metrics")
            if isinstance(summary_metrics, Mapping):
                candidate_reason = summary_metrics.get("skip_reason_top")
                if isinstance(candidate_reason, str) and candidate_reason.strip():
                    skip_reason_top = candidate_reason.strip()
                    decision_metadata["skip_reason_top"] = skip_reason_top
        if not skip_counts_metadata and isinstance(summary_payload, Mapping):
            summary_metrics = summary_payload.get("metrics")
            if isinstance(summary_metrics, Mapping):
                skip_counts_metadata = _normalize_skip_counts(summary_metrics.get("skip_counts"))
        if skip_counts_metadata:
            decision_metadata.setdefault("skip_counts", skip_counts_metadata)

        merge_zero_packs_flag = False
        if isinstance(metrics_payload, Mapping):
            merge_zero_packs_flag = bool(metrics_payload.get("merge_zero_packs"))
        if not merge_zero_packs_flag and isinstance(summary_payload, Mapping):
            summary_direct_flag = summary_payload.get("merge_zero_packs")
            if isinstance(summary_direct_flag, str):
                lowered = summary_direct_flag.strip().lower()
                if lowered in {"1", "true", "yes", "on"}:
                    merge_zero_packs_flag = True
                elif lowered in {"0", "false", "no", "off"}:
                    merge_zero_packs_flag = False
            elif isinstance(summary_direct_flag, (int, float, bool)) and not isinstance(summary_direct_flag, str):
                merge_zero_packs_flag = bool(summary_direct_flag)
        if not merge_zero_packs_flag and isinstance(summary_payload, Mapping):
            summary_metrics = summary_payload.get("metrics")
            if isinstance(summary_metrics, Mapping):
                merge_zero_packs_flag = bool(summary_metrics.get("merge_zero_packs"))

        merge_status_value = str(merge_stage.get("status", "") or "").strip().lower()
        merge_successful = merge_status_value in {"success", "built"}

        if accounts_count == 0:
            _set("complete_no_action", "no_accounts", "COMPLETE_NO_ACTION")
        else:
            validation_stage = stages.get("validation") if isinstance(stages, Mapping) else None
            validation_status_value = str(
                validation_stage.get("status", "") if isinstance(validation_stage, Mapping) else ""
            ).strip().lower()
            validation_sent_flag = bool(validation_stage.get("sent")) if isinstance(validation_stage, Mapping) else False
            validation_fastpath_locked = _fastpath_has_lock(run_dir)
            zero_packs_fastpath = (
                _RUNFLOW_MERGE_ZERO_PACKS_FASTPATH
                and merge_successful
                and merge_zero_packs_flag
                and (
                    not isinstance(validation_stage, Mapping)
                    or validation_status_value in {"empty", "built", "pending"}
                )
            )

            fastpath_enqueued = False
            if zero_packs_fastpath and not validation_sent_flag and not validation_fastpath_locked:
                fastpath_enqueued = _enqueue_validation_fastpath(
                    sid,
                    run_dir,
                    merge_zero_packs=True,
                    payload={"merge_zero_fastpath": True},
                )
                validation_fastpath_locked = validation_fastpath_locked or fastpath_enqueued

            if zero_packs_fastpath and (validation_sent_flag or validation_fastpath_locked or fastpath_enqueued):
                skip_merge_wait = True
                _set("run_validation", "merge_zero_packs", "VALIDATING")
                decision_metadata["validation_fastpath"] = True

                if not isinstance(validation_stage, Mapping):
                    validation_stage = {}
                else:
                    validation_stage = dict(validation_stage)

                validation_stage["sent"] = True
                stage_status_normalized = validation_status_value
                if stage_status_normalized in {"empty", "built", "", "pending"}:
                    validation_stage["status"] = "in_progress"

                validation_stage["last_at"] = _now_iso()

                metrics_payload = validation_stage.get("metrics")
                if isinstance(metrics_payload, Mapping):
                    metrics_payload = dict(metrics_payload)
                else:
                    metrics_payload = {}
                metrics_payload["merge_zero_packs"] = True
                validation_stage["metrics"] = metrics_payload

                summary_payload = validation_stage.get("summary")
                if isinstance(summary_payload, Mapping):
                    summary_payload = dict(summary_payload)
                else:
                    summary_payload = {}
                summary_payload["merge_zero_packs"] = True

                summary_metrics_payload = summary_payload.get("metrics")
                if isinstance(summary_metrics_payload, Mapping):
                    summary_metrics_payload = dict(summary_metrics_payload)
                else:
                    summary_metrics_payload = {}
                summary_metrics_payload["merge_zero_packs"] = True
                summary_payload["metrics"] = summary_metrics_payload
                validation_stage["summary"] = summary_payload

                stages_mutated = True
                stages["validation"] = validation_stage  # type: ignore[index]
            else:
                if not isinstance(validation_stage, Mapping):
                    validation_stage = {}
                validation_status = str(validation_stage.get("status") or "")
                normalized_validation_status = validation_status.strip().lower()
                findings_count = _coerce_int(validation_stage.get("findings_count"))
                if findings_count is None:
                    findings_count = 0

                if normalized_validation_status == "error":
                    _set("stop_error", "validation_error", "ERROR")
                elif normalized_validation_status not in {"success", "built"}:
                    _set("run_validation", "validation_pending", "VALIDATING")
                elif findings_count <= 0:
                    _set("complete_no_action", "validation_no_findings", "COMPLETE_NO_ACTION")
                else:
                    frontend_stage = stages.get("frontend")
                    if not isinstance(frontend_stage, Mapping):
                        _set("gen_frontend_packs", "validation_has_findings", "VALIDATING")
                    else:
                        frontend_status = str(frontend_stage.get("status") or "")
                        normalized_frontend_status = frontend_status.strip().lower()
                        if normalized_frontend_status == "error":
                            _set("stop_error", "frontend_error", "ERROR")
                        elif normalized_frontend_status in {"published", "success"}:
                            packs_count = _coerce_int(frontend_stage.get("packs_count")) or 0
                            if packs_count <= 0:
                                _set(
                                    "complete_no_action",
                                    "frontend_no_packs",
                                    "COMPLETE_NO_ACTION",
                                )
                            else:
                                reason_label = (
                                    "frontend_completed"
                                    if normalized_frontend_status == "success"
                                    else "frontend_published"
                                )
                                _set(
                                    "await_input",
                                    reason_label,
                                    "AWAITING_CUSTOMER_INPUT",
                                )
                        else:
                            _set("gen_frontend_packs", "validation_has_findings", "VALIDATING")

    run_state_changed = data.get("run_state") != new_state
    if run_state_changed:
        data["run_state"] = new_state

    if run_state_changed or stages_mutated:
        data = record_stage_force(
            sid,
            data,
            runs_root=runs_root,
            last_writer="decide_next",
            refresh_barriers=False,
        )
        try:
            reconcile_umbrella_barriers(sid, runs_root=runs_root)
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "RUNFLOW_RECONCILE_AT_STATE_CHANGE_FAILED sid=%s state=%s",
                sid,
                new_state,
                exc_info=True,
            )

    log.info(
        "RUNFLOW_DECIDE sid=%s next=%s reason=%s state=%s",
        sid,
        next_action,
        reason,
        new_state,
    )

    stage_for_step = _latest_stage_name(stages)
    if stage_for_step and isinstance(stages.get(stage_for_step), Mapping):
        try:
            runflow_decide_step(
                sid,
                stage_for_step,
                next_action=_decision_next_label(next_action),
                reason=reason,
            )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "RUNFLOW_DECIDE_STEP_FAILED sid=%s stage=%s next=%s reason=%s",
                sid,
                stage_for_step,
                next_action,
                reason,
                exc_info=True,
            )

    decision_payload: dict[str, Any] = {"next": next_action, "reason": reason}
    if skip_merge_wait:
        decision_payload["skip_merge_wait"] = True
    decision_payload.update(decision_metadata)
    if (
        _MERGE_ZERO_PACKS_SIGNAL_ENABLED
        and "merge_zero_packs" not in decision_payload
        and merge_stage
    ):
        merge_metrics_payload = merge_stage.get("metrics") if isinstance(merge_stage, Mapping) else None
        merge_zero_flag = False
        if isinstance(merge_metrics_payload, Mapping):
            merge_zero_flag = bool(merge_metrics_payload.get("merge_zero_packs"))
        if not merge_zero_flag and isinstance(merge_stage, Mapping):
            summary_payload = merge_stage.get("summary")
            if isinstance(summary_payload, Mapping):
                summary_metrics_payload = summary_payload.get("metrics")
                if isinstance(summary_metrics_payload, Mapping):
                    merge_zero_flag = bool(summary_metrics_payload.get("merge_zero_packs"))
        if merge_zero_flag:
            decision_payload["merge_zero_packs"] = True

    return decision_payload


__all__ = [
    "record_stage",
    "decide_next",
    "StageStatus",
    "RunState",
    "get_runflow_snapshot",
    "refresh_validation_stage_from_index",
    "refresh_note_style_stage_from_index",
    "refresh_frontend_stage_from_responses",
    "reconcile_umbrella_barriers",
]
if TYPE_CHECKING:
    from backend.ai.note_style.io import NoteStyleSnapshot
    from backend.validation.index_schema import ValidationIndex


def _note_style_snapshot_for_run(run_dir: Path) -> "NoteStyleSnapshot":
    from backend.ai.note_style.io import note_style_snapshot

    sid_text = run_dir.name
    runs_root_path = run_dir.parent
    return note_style_snapshot(sid_text, runs_root=runs_root_path)

