from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import asyncio
import functools
import json
import logging
import os
import threading
from concurrent.futures import Future

_LOG = logging.getLogger(__name__)

from backend.core.ai.paths import (
    validation_result_json_filename_for_account,
    validation_result_jsonl_filename_for_account,
)
from backend.core.io.json_io import update_json_in_place
from backend.core.runflow_steps import (
    RUNS_ROOT,
    steps_append,
    steps_init,
    steps_stage_finish,
    steps_stage_start,
    steps_update_aggregate,
)
from backend.runflow.counters import frontend_answers_counters as _frontend_answers_counters


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    mode = 0o644
    fd = os.open(path, flags, mode)
    try:
        os.write(fd, payload)
    finally:
        os.close(fd)


def _events_path(sid: str) -> Path:
    return RUNS_ROOT / sid / "runflow_events.jsonl"


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw.strip())
    except (TypeError, ValueError):
        return default
    return value


def _review_attachment_required() -> bool:
    return _env_enabled("UMBRELLA_BARRIERS_REVIEW_REQUIRE_FILE", False)


def _review_explanation_required() -> bool:
    return _env_enabled("UMBRELLA_BARRIERS_REVIEW_REQUIRE_EXPLANATION", True)


def _umbrella_require_style() -> bool:
    return _env_enabled("UMBRELLA_REQUIRE_STYLE", False)


def _document_verifier_enabled() -> bool:
    return _env_enabled("DOCUMENT_VERIFIER_ENABLED", False)


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, Mapping):
        return payload

    return None


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _has_review_attachments(payload: Mapping[str, Any]) -> bool:
    attachments = payload.get("attachments")
    if isinstance(attachments, Mapping):
        for value in attachments.values():
            if isinstance(value, str) and value.strip():
                return True
            if isinstance(value, Iterable) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for entry in value:
                    if isinstance(entry, str) and entry.strip():
                        return True

    legacy = payload.get("evidence")
    if isinstance(legacy, Iterable) and not isinstance(legacy, (str, bytes, bytearray)):
        for item in legacy:
            if not isinstance(item, Mapping):
                continue
            docs = item.get("docs")
            if isinstance(docs, Iterable) and not isinstance(docs, (str, bytes, bytearray)):
                for doc in docs:
                    if not isinstance(doc, Mapping):
                        continue
                    doc_ids = doc.get("doc_ids")
                    if isinstance(doc_ids, Iterable) and not isinstance(
                        doc_ids, (str, bytes, bytearray)
                    ):
                        for doc_id in doc_ids:
                            if isinstance(doc_id, str) and doc_id.strip():
                                return True
    return False


def _resolve_review_path(
    run_dir: Path,
    env_name: str,
    canonical: Path,
    *,
    review_dir: Path,
    require_descendant: bool = False,
) -> Path:
    override = os.getenv(env_name)
    if not override:
        return canonical

    candidate = Path(override)
    if not candidate.is_absolute():
        candidate = (run_dir / candidate).resolve()
    else:
        candidate = candidate.resolve()

    frontend_base = review_dir.parent
    try:
        candidate.relative_to(frontend_base)
        within_frontend = True
    except ValueError:
        within_frontend = False

    try:
        candidate.relative_to(review_dir)
        within_review = True
    except ValueError:
        within_review = False

    if within_frontend and not within_review:
        return canonical

    if require_descendant and within_review and candidate == review_dir:
        return canonical

    return candidate


def _load_response_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None

    text = raw.strip()
    if not text:
        return None

    if path.suffix == ".jsonl":
        for line in reversed(text.splitlines()):
            stripped = line.strip()
            if stripped:
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, Mapping) else None
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None

    return payload if isinstance(payload, Mapping) else None


_ENABLE_STEPS = _env_enabled("RUNFLOW_VERBOSE")
_ENABLE_EVENTS = _env_enabled("RUNFLOW_EVENTS")
_STEP_SAMPLE_EVERY = max(_env_int("RUNFLOW_STEP_LOG_EVERY", 1), 1)
_PAIR_TOPN = max(_env_int("RUNFLOW_STEPS_PAIR_TOPN", 5), 0)
_ENABLE_SPANS = _env_enabled("RUNFLOW_STEPS_ENABLE_SPANS", True)
_ENABLE_ACCOUNT_STEPS = _env_enabled("RUNFLOW_ACCOUNT_STEPS", True)
_SUPPRESS_ACCOUNT_STEPS = _env_enabled("RUNFLOW_STEPS_SUPPRESS_PER_ACCOUNT")
_ONLY_AGGREGATES = _env_enabled("RUNFLOW_STEPS_ONLY_AGGREGATES")
_UMBRELLA_BARRIERS_ENABLED = _env_enabled("UMBRELLA_BARRIERS_ENABLED", True)
_UMBRELLA_BARRIERS_LOG = _env_enabled("UMBRELLA_BARRIERS_LOG", True)
_UMBRELLA_BARRIERS_WATCHDOG_INTERVAL_MS = max(
    _env_int("UMBRELLA_BARRIERS_WATCHDOG_INTERVAL_MS", 5000), 0
)
_UMBRELLA_BARRIERS_DEBOUNCE_MS = max(_env_int("UMBRELLA_BARRIERS_DEBOUNCE_MS", 300), 0)
_UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG = _env_enabled(
    "UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG", True
)


_STEP_CALL_COUNTS: dict[tuple[str, str, str, str], int] = defaultdict(int)
_STARTED_STAGES: set[tuple[str, str]] = set()
_STAGE_COUNTERS: dict[str, dict[str, dict[str, int]]] = {}
_STAGE_AGGREGATES: dict[str, dict[str, dict[str, int]]] = {}
_WATCHDOG_LOOP: Optional[asyncio.AbstractEventLoop] = None
_WATCHDOG_THREAD: Optional[threading.Thread] = None
_WATCHDOG_FUTURES: dict[str, Future[Any]] = {}
_WATCHDOG_LOCK = threading.Lock()
_UMBRELLA_REFRESH_TIMERS: dict[str, tuple[threading.Timer, object]] = {}
_UMBRELLA_REFRESH_LOCK = threading.Lock()

_VALIDATION_AGGREGATE_KEYS = frozenset({"packs_total", "packs_completed", "packs_pending"})
_REVIEW_AGGREGATE_KEYS = frozenset({"answers_received", "answers_required"})


def _append_event(sid: str, row: Mapping[str, Any]) -> None:
    if not _ENABLE_EVENTS:
        return
    _append_jsonl(_events_path(sid), row)


def steps_pair_topn() -> int:
    """Return the configured Top-N threshold for merge pair steps."""

    return _PAIR_TOPN


def runflow_account_steps_enabled() -> bool:
    """Return ``True`` when per-account runflow step logging is enabled."""

    return _ENABLE_ACCOUNT_STEPS


def _coerce_summary_counts(summary: Mapping[str, Any]) -> dict[str, int]:
    normalized: dict[str, int] = {}
    for key, value in summary.items():
        try:
            normalized[str(key)] = int(value)
        except (TypeError, ValueError):
            normalized[str(key)] = 0
    return normalized


def _store_stage_counter(sid: str, bucket: str, summary: Mapping[str, Any]) -> dict[str, int]:
    counters = _STAGE_COUNTERS.setdefault(sid, {})
    payload = _coerce_summary_counts(summary)
    counters[bucket] = payload
    return payload


def _aggregates_enabled() -> bool:
    return _SUPPRESS_ACCOUNT_STEPS and _ONLY_AGGREGATES


def _aggregate_state(sid: str, stage: str) -> dict[str, int]:
    state = _STAGE_AGGREGATES.setdefault(sid, {})
    return state.setdefault(stage, {})


def _aggregate_prune(stage_state: dict[str, int], allowed_keys: Iterable[str]) -> None:
    allowed = set(allowed_keys)
    for key in list(stage_state):
        if key not in allowed:
            stage_state.pop(key, None)


def _aggregate_set_nonnegative(stage_state: dict[str, int], key: str, value: Any) -> bool:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return False
    if number < 0:
        number = 0
    stage_state[key] = number
    return True


def _aggregate_value(stage_state: Mapping[str, Any], key: str) -> Optional[int]:
    value = stage_state.get(key)
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_stage_aggregate(sid: str, stage: str) -> None:
    if not _aggregates_enabled():
        return
    state = _STAGE_AGGREGATES.get(sid, {})
    stage_state = state.get(stage)
    if not stage_state:
        return
    summary_payload: dict[str, int]
    if stage == "validation":
        _aggregate_prune(stage_state, _VALIDATION_AGGREGATE_KEYS)
        total_value = _aggregate_value(stage_state, "packs_total")
        completed_value = _aggregate_value(stage_state, "packs_completed")
        pending_value = _aggregate_value(stage_state, "packs_pending")
        if total_value is None and completed_value is None and pending_value is None:
            return
        total = max(total_value or 0, 0)
        completed = max(completed_value or 0, 0)
        if total and completed > total:
            completed = total
        if pending_value is None:
            pending = max(total - completed, 0)
        else:
            pending = max(pending_value, 0)
            if total and pending > total:
                pending = total
            if total and pending < total - completed:
                pending = total - completed
        stage_state["packs_total"] = total
        stage_state["packs_completed"] = completed
        stage_state["packs_pending"] = pending
        summary_payload = {
            "packs_total": total,
            "packs_completed": completed,
            "packs_pending": pending,
        }
    elif stage == "review":
        _aggregate_prune(stage_state, _REVIEW_AGGREGATE_KEYS)
        received_value = _aggregate_value(stage_state, "answers_received")
        required_value = _aggregate_value(stage_state, "answers_required")
        if received_value is None and required_value is None:
            return
        received = max(received_value or 0, 0)
        required = max(required_value or 0, 0)
        if required and received > required:
            received = required
        stage_state["answers_received"] = received
        stage_state["answers_required"] = required
        summary_payload = {
            "answers_received": received,
            "answers_required": required,
        }
    else:
        summary_payload = {key: stage_state[key] for key in sorted(stage_state)}
    steps_update_aggregate(sid, stage, summary_payload)


def _clear_stage_aggregate(sid: str, stage: str) -> None:
    state = _STAGE_AGGREGATES.get(sid)
    if not state:
        return
    state.pop(stage, None)
    if not state:
        _STAGE_AGGREGATES.pop(sid, None)


def _emit_summary_step(
    sid: str, stage: str, step: str, *, summary: Mapping[str, int]
) -> None:
    metrics = {str(key): value for key, value in summary.items()}
    out_payload = {"summary": dict(metrics)} if metrics else None
    runflow_step(
        sid,
        stage,
        step,
        status="success",
        metrics=metrics or None,
        out=out_payload,
    )


def _clear_stage_counters(sid: str, *buckets: str) -> None:
    state = _STAGE_COUNTERS.get(sid)
    if not state:
        return
    for bucket in buckets:
        state.pop(bucket, None)
    if not state:
        _STAGE_COUNTERS.pop(sid, None)


def _stage_status_success(stage_info: Mapping[str, Any] | None) -> bool:
    if not isinstance(stage_info, Mapping):
        return False
    status_value = stage_info.get("status")
    if not isinstance(status_value, str):
        return False
    return status_value.strip().lower() == "success"


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None or isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _ensure_stage_dict(stages: dict[str, Any], stage_name: str) -> Optional[dict[str, Any]]:
    candidate = stages.get(stage_name)
    if not isinstance(candidate, Mapping):
        return None
    if not isinstance(candidate, dict):
        candidate = dict(candidate)
        stages[stage_name] = candidate
    return candidate


def _ensure_summary(stage: dict[str, Any]) -> dict[str, Any]:
    summary_payload = stage.get("summary")
    if isinstance(summary_payload, dict):
        summary = summary_payload
    elif isinstance(summary_payload, Mapping):
        summary = dict(summary_payload)
    else:
        summary = {}
    stage["summary"] = summary
    return summary


def _ensure_results(stage: dict[str, Any]) -> dict[str, Any]:
    results_payload = stage.get("results")
    if isinstance(results_payload, dict):
        results = results_payload
    elif isinstance(results_payload, Mapping):
        results = dict(results_payload)
    else:
        results = {}
    stage["results"] = results
    return results


def _ensure_metrics(stage: dict[str, Any]) -> dict[str, Any]:
    metrics_payload = stage.get("metrics")
    if isinstance(metrics_payload, dict):
        metrics = metrics_payload
    elif isinstance(metrics_payload, Mapping):
        metrics = dict(metrics_payload)
    else:
        metrics = {}
    stage["metrics"] = metrics
    return metrics


def _apply_umbrella_barriers(
    payload_dict: dict[str, Any],
    *,
    sid: str,
    timestamp: Optional[str] = None,
) -> Optional[tuple[dict[str, Any], bool, bool, bool, bool, str]]:
    if not _UMBRELLA_BARRIERS_ENABLED:
        return None

    normalized_sid = str(sid or "").strip()
    if not normalized_sid:
        existing_sid = payload_dict.get("sid")
        if isinstance(existing_sid, str):
            normalized_sid = existing_sid.strip()
    if not normalized_sid:
        return None

    ts = timestamp or _utcnow_iso()

    stages_payload = payload_dict.get("stages")
    if isinstance(stages_payload, dict):
        stages = stages_payload
    elif isinstance(stages_payload, Mapping):
        stages = dict(stages_payload)
        payload_dict["stages"] = stages
    else:
        stages = {}
        payload_dict["stages"] = stages

    merge_stage = _ensure_stage_dict(stages, "merge")
    validation_stage = _ensure_stage_dict(stages, "validation")
    frontend_stage = _ensure_stage_dict(stages, "frontend")
    note_style_stage = _ensure_stage_dict(stages, "note_style")
    strategy_stage = _ensure_stage_dict(stages, "strategy")

    def _boolish(value: Any) -> Optional[bool]:
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

    # ------------------------------------------------------------------
    # Fallback readiness based on the existing runflow snapshot.  This is
    # retained so that we still populate summaries/metrics when disk
    # inspection fails for any reason.  The booleans are overridden with
    # authoritative values computed from disk further below.
    # ------------------------------------------------------------------
    merge_ready = False
    if merge_stage and _stage_status_success(merge_stage):
        summary = _ensure_summary(merge_stage)
        result_files_value = _coerce_int(summary.get("result_files"))
        if result_files_value is None:
            fallback_result_files = _coerce_int(merge_stage.get("result_files"))
            if fallback_result_files is not None:
                summary["result_files"] = fallback_result_files
                result_files_value = fallback_result_files
        empty_ok = bool(merge_stage.get("empty_ok")) or bool(summary.get("empty_ok"))
        if empty_ok and result_files_value is None:
            result_files_value = 0
        if empty_ok:
            merge_ready = True
        else:
            merge_ready = (result_files_value or 0) >= 1

    validation_ready = False
    validation_ready_latched = False

    def _validation_flag(stage: Mapping[str, Any], key: str) -> Optional[bool]:
        containers: list[Mapping[str, Any]] = [stage]
        metrics_payload = stage.get("metrics")
        if isinstance(metrics_payload, Mapping):
            containers.append(metrics_payload)
        summary_payload = stage.get("summary")
        if isinstance(summary_payload, Mapping):
            containers.append(summary_payload)
            summary_metrics = summary_payload.get("metrics")
            if isinstance(summary_metrics, Mapping):
                containers.append(summary_metrics)
        for container in containers:
            value = container.get(key)
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return bool(value)
        return None

    if validation_stage:
        flag = _validation_flag(validation_stage, "ready_latched")
        if flag:
            validation_ready_latched = True

    validation_metrics_payload: dict[str, Any] = {}
    validation_summary_payload: dict[str, Any] = {}

    if validation_stage:
        results_data = _ensure_results(validation_stage)
        summary = _ensure_summary(validation_stage)
        validation_summary_payload = summary
        summary_results_payload = summary.get("results")
        if isinstance(summary_results_payload, dict):
            summary_results = summary_results_payload
        elif isinstance(summary_results_payload, Mapping):
            summary_results = dict(summary_results_payload)
        else:
            summary_results = {}
        results_missing = False
        for key in ("results_total", "completed", "failed"):
            value = _coerce_int(results_data.get(key))
            if value is None:
                fallback_value = _coerce_int(summary_results.get(key))
                if fallback_value is not None:
                    results_data[key] = fallback_value
                    value = fallback_value
                else:
                    default_value = 0
                    results_data[key] = default_value
                    value = default_value
                    if key in {"results_total", "completed"}:
                        results_missing = True
            summary_results[key] = value
        if summary_results:
            summary["results"] = summary_results

        metrics_payload = _ensure_metrics(validation_stage)
        validation_metrics_payload = metrics_payload
        required_flag = _validation_flag(validation_stage, "validation_ai_required")
        if required_flag is None:
            required_flag = False
        metrics_payload["validation_ai_required"] = bool(required_flag)
        summary["validation_ai_required"] = bool(required_flag)
        completed_flag = _validation_flag(validation_stage, "validation_ai_completed")
        if completed_flag:
            metrics_payload["validation_ai_completed"] = True
            summary["validation_ai_completed"] = True

        empty_ok = bool(validation_stage.get("empty_ok")) or bool(summary.get("empty_ok"))
        status_value = validation_stage.get("status")
        status_normalized = (
            status_value.strip().lower() if isinstance(status_value, str) else ""
        )

        if validation_ready_latched:
            validation_ready = True
            metrics_payload["validation_ai_completed"] = True
            summary["validation_ai_completed"] = True
        elif required_flag:
            if not results_missing:
                completed_value = _coerce_int(results_data.get("completed"))
                total_value = _coerce_int(results_data.get("results_total"))
                if (
                    completed_value is not None
                    and total_value is not None
                    and total_value > 0
                ):
                    validation_ready = completed_value == total_value
                if not validation_ready:
                    metrics_payload.setdefault("validation_ai_completed", False)
                    summary.setdefault("validation_ai_completed", False)
        else:
            if status_normalized not in {"error", "failed"}:
                validation_ready = True
                metrics_payload["validation_ai_completed"] = True
                summary["validation_ai_completed"] = True
            else:
                metrics_payload.setdefault("validation_ai_completed", False)
                summary.setdefault("validation_ai_completed", False)

        summary.setdefault(
            "validation_ai_completed",
            bool(metrics_payload.get("validation_ai_completed", False)),
        )

    review_ready = False
    if frontend_stage and _stage_status_success(frontend_stage):
        metrics_data = _ensure_metrics(frontend_stage)
        summary = _ensure_summary(frontend_stage)
        summary_metrics_payload = summary.get("metrics")
        if isinstance(summary_metrics_payload, dict):
            summary_metrics = summary_metrics_payload
        elif isinstance(summary_metrics_payload, Mapping):
            summary_metrics = dict(summary_metrics_payload)
        else:
            summary_metrics = {}
        metrics_missing = False
        required_value = _coerce_int(metrics_data.get("answers_required"))
        if required_value is None:
            fallback_required = _coerce_int(summary_metrics.get("answers_required"))
            if fallback_required is not None:
                metrics_data["answers_required"] = fallback_required
                required_value = fallback_required
            else:
                metrics_data["answers_required"] = 0
                required_value = 0
                metrics_missing = True
        summary_metrics["answers_required"] = required_value
        received_value = _coerce_int(metrics_data.get("answers_received"))
        if received_value is None:
            fallback_received = _coerce_int(summary_metrics.get("answers_received"))
            if fallback_received is not None:
                metrics_data["answers_received"] = fallback_received
                received_value = fallback_received
            else:
                metrics_data["answers_received"] = 0
                received_value = 0
                metrics_missing = True
        summary_metrics["answers_received"] = received_value
        if summary_metrics:
            summary["metrics"] = summary_metrics
        empty_ok = bool(frontend_stage.get("empty_ok")) or bool(summary.get("empty_ok"))
        if (
            required_value is not None
            and received_value is not None
            and not metrics_missing
        ):
            review_ready = received_value == required_value
        if empty_ok:
            review_ready = True

    style_ready = False
    style_required = False
    style_waiting_for_review = False
    if note_style_stage:
        summary = _ensure_summary(note_style_stage)
        metrics_payload = _ensure_metrics(note_style_stage)
        summary_metrics = summary.get("metrics")
        if isinstance(summary_metrics, Mapping) and not isinstance(summary_metrics, dict):
            summary_metrics = dict(summary_metrics)
        elif not isinstance(summary_metrics, Mapping):
            summary_metrics = {}
        summary["metrics"] = summary_metrics

        waiting_candidates = [
            note_style_stage.get("waiting_for_review"),
            summary.get("waiting_for_review"),
        ]
        for candidate in waiting_candidates:
            flag = _boolish(candidate)
            if flag is not None:
                style_waiting_for_review = flag
                break
        summary["waiting_for_review"] = style_waiting_for_review

        packs_total_candidates = [
            _coerce_int(note_style_stage.get("packs_total")),
            _coerce_int(metrics_payload.get("packs_total")),
            _coerce_int(summary.get("packs_total")),
            _coerce_int(summary_metrics.get("packs_total")),
        ]
        packs_total = max((value for value in packs_total_candidates if value is not None), default=0)
        metrics_payload["packs_total"] = packs_total
        summary["packs_total"] = packs_total

        summary_results_payload = summary.get("results")
        if isinstance(summary_results_payload, Mapping) and not isinstance(summary_results_payload, dict):
            summary_results = dict(summary_results_payload)
        elif isinstance(summary_results_payload, Mapping):
            summary_results = dict(summary_results_payload)
        else:
            summary_results = {}
        summary["results"] = summary_results

        results_total_candidates = [
            _coerce_int(summary_results.get("results_total")),
            _coerce_int(summary.get("results_total")),
        ]
        results_total = max((value for value in results_total_candidates if value is not None), default=0)
        summary_results["results_total"] = results_total
        summary["results_total"] = results_total

        completed_candidates = [
            _coerce_int(summary_results.get("completed")),
            _coerce_int(summary.get("completed")),
        ]
        completed_total = max((value for value in completed_candidates if value is not None), default=0)
        summary_results["completed"] = completed_total
        summary["completed"] = completed_total

        failed_candidates = [
            _coerce_int(summary_results.get("failed")),
            _coerce_int(summary.get("failed")),
        ]
        failed_total = max((value for value in failed_candidates if value is not None), default=0)
        summary_results["failed"] = failed_total
        summary["failed"] = failed_total

        status_raw = note_style_stage.get("status")
        status_normalized = status_raw.strip().lower() if isinstance(status_raw, str) else ""
        empty_ok = bool(note_style_stage.get("empty_ok")) or bool(summary.get("empty_ok"))

        style_required = bool(style_waiting_for_review)
        if not style_required:
            if packs_total > 0 or results_total > 0 or completed_total > 0 or failed_total > 0:
                style_required = True
            elif status_normalized in {
                "built",
                "processing",
                "in_progress",
                "waiting_for_review",
                "error",
            }:
                style_required = True
            elif status_normalized in {"success", "empty"} and not empty_ok:
                style_required = True

        if style_waiting_for_review:
            style_required = True

        if not style_required:
            style_ready = True
        else:
            if style_waiting_for_review:
                style_ready = False
            elif status_normalized in {"success", "empty"}:
                if packs_total == 0 and results_total == 0 and failed_total == 0 and empty_ok:
                    style_ready = True
                else:
                    style_ready = (
                        packs_total > 0
                        and completed_total == packs_total
                        and failed_total == 0
                    )
            else:
                style_ready = (
                    packs_total > 0
                    and completed_total == packs_total
                    and failed_total == 0
                )
    else:
        style_required = False
        style_ready = True

    if _umbrella_require_style():
        style_required = True
        if not note_style_stage:
            style_ready = False
        elif style_waiting_for_review:
            style_ready = False

    strategy_ready = False
    strategy_required = False
    strategy_status_normalized = ""
    if strategy_stage:
        status_value = strategy_stage.get("status")
        if isinstance(status_value, str):
            strategy_status_normalized = status_value.strip().lower()
        if strategy_status_normalized in {"built", "success", "error"}:
            strategy_required = True

    validation_containers: list[Mapping[str, Any]] = []
    if validation_stage:
        validation_containers.append(validation_stage)
    if validation_metrics_payload:
        validation_containers.append(validation_metrics_payload)
    if validation_summary_payload:
        validation_containers.append(validation_summary_payload)
        summary_metrics = validation_summary_payload.get("metrics")
        if isinstance(summary_metrics, Mapping):
            validation_containers.append(summary_metrics)
        summary_results = validation_summary_payload.get("results")
        if isinstance(summary_results, Mapping):
            validation_containers.append(summary_results)

    def _validation_bool(key: str) -> Optional[bool]:
        for container in validation_containers:
            if not isinstance(container, Mapping):
                continue
            flag = _boolish(container.get(key))
            if flag is not None:
                return flag
        return None

    def _validation_int(*keys: str) -> int:
        for container in validation_containers:
            if not isinstance(container, Mapping):
                continue
            for key in keys:
                value = _coerce_int(container.get(key))
                if value is not None:
                    return value
        return 0

    validation_ai_required = bool(_validation_bool("validation_ai_required"))
    validation_ai_completed = bool(_validation_bool("validation_ai_completed"))

    validation_findings = _validation_int("findings_count", "findings_total")
    validation_accounts_eligible = _validation_int(
        "accounts_eligible",
        "eligible_accounts",
        "packs_total",
    )

    merge_results_applied = False
    if validation_stage:
        merge_block = validation_stage.get("merge_results")
        if isinstance(merge_block, Mapping):
            for key in ("applied", "merge_results_applied"):
                flag = _boolish(merge_block.get(key))
                if flag is not None:
                    merge_results_applied = flag
                    break
    if not merge_results_applied:
        for merge_key in ("merge_results_applied", "merge_applied"):
            flag = _validation_bool(merge_key)
            if flag is not None:
                merge_results_applied = flag
                break

    if validation_findings > 0 or validation_accounts_eligible > 0:
        strategy_required = True

    if not strategy_required:
        strategy_ready = True
    elif strategy_status_normalized == "success":
        strategy_ready = True
    elif strategy_status_normalized == "error":
        strategy_ready = False
    else:
        if validation_ready and merge_results_applied:
            if not validation_ai_required:
                strategy_ready = True
            elif validation_ai_completed:
                strategy_ready = True

    if not strategy_ready and not strategy_required and strategy_stage and _stage_status_success(strategy_stage):
        strategy_ready = True

    all_ready = (
        merge_ready
        and validation_ready
        and review_ready
        and style_ready
        and strategy_ready
    )

    # ------------------------------------------------------------------
    # Prefer authoritative readiness derived from disk artifacts.
    # ------------------------------------------------------------------
    disk_statuses: Optional[Mapping[str, Any]] = None
    run_dir = RUNS_ROOT / normalized_sid
    try:  # pragma: no cover - exercised via higher-level tests
        from backend.runflow.decider import evaluate_global_barriers
    except Exception:  # pragma: no cover - defensive import fallback
        evaluate_global_barriers = None  # type: ignore[assignment]

    if evaluate_global_barriers is not None:
        try:
            disk_statuses = evaluate_global_barriers(str(run_dir))
        except Exception:  # pragma: no cover - defensive
            disk_statuses = None

    merge_zero_packs_flag = False
    if _UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG and merge_stage:
        merge_metrics = _ensure_metrics(merge_stage)
        merge_zero_candidate: Any = merge_metrics.get("merge_zero_packs")
        if merge_zero_candidate is None:
            summary_payload = _ensure_summary(merge_stage)
            direct_merge_zero = summary_payload.get("merge_zero_packs")
            if direct_merge_zero is not None:
                merge_zero_candidate = direct_merge_zero
        if merge_zero_candidate is None:
            summary_payload = _ensure_summary(merge_stage)
            summary_metrics = summary_payload.get("metrics")
            if isinstance(summary_metrics, Mapping):
                merge_zero_candidate = summary_metrics.get("merge_zero_packs")
        if merge_zero_candidate is None:
            summary_payload = _ensure_summary(merge_stage)
            scored_val = _coerce_int(summary_payload.get("pairs_scored"))
            created_val = _coerce_int(summary_payload.get("packs_created"))
            if scored_val is not None and created_val is not None and scored_val > 0 and created_val == 0:
                merge_zero_candidate = True
        merge_zero_packs_flag = bool(merge_zero_candidate)

    if isinstance(disk_statuses, Mapping) and disk_statuses:
        merge_ready = bool(disk_statuses.get("merge_ready", merge_ready))
        validation_ready = bool(
            disk_statuses.get("validation_ready", validation_ready)
        )
        review_ready = bool(disk_statuses.get("review_ready", review_ready))
        style_ready = bool(disk_statuses.get("style_ready", style_ready))
        style_required = bool(disk_statuses.get("style_required", style_required))
        strategy_ready = bool(disk_statuses.get("strategy_ready", strategy_ready))
        strategy_required = bool(
            disk_statuses.get("strategy_required", strategy_required)
        )
        if "style_waiting_for_review" in disk_statuses:
            style_waiting_for_review = bool(
                disk_statuses.get("style_waiting_for_review", style_waiting_for_review)
            )
        if style_waiting_for_review:
            style_ready = False
        default_all_ready = (
            merge_ready
            and validation_ready
            and review_ready
            and style_ready
            and strategy_ready
        )
        all_ready = bool(disk_statuses.get("all_ready", default_all_ready))
        if _UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG:
            merge_zero_packs_flag = bool(
                disk_statuses.get("merge_zero_packs", merge_zero_packs_flag)
            )
    else:
        default_all_ready = (
            merge_ready
            and validation_ready
            and review_ready
            and style_ready
            and strategy_ready
        )
        all_ready = default_all_ready
        if style_waiting_for_review:
            style_ready = False

    normalized_barriers: dict[str, Any] = {
        "merge_ready": merge_ready,
        "validation_ready": validation_ready,
        "strategy_ready": strategy_ready,
        "review_ready": review_ready,
        "style_ready": style_ready,
        "all_ready": all_ready,
        "checked_at": ts,
    }
    normalized_barriers["style_waiting_for_review"] = style_waiting_for_review
    normalized_barriers["style_required"] = style_required
    normalized_barriers["strategy_required"] = strategy_required
    if _UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG:
        normalized_barriers["merge_zero_packs"] = merge_zero_packs_flag
    if _document_verifier_enabled():
        normalized_barriers.setdefault("document_ready", False)
    if isinstance(disk_statuses, Mapping):
        if "document_ready" in disk_statuses:
            normalized_barriers["document_ready"] = bool(
                disk_statuses.get("document_ready")
            )

    existing_barriers = payload_dict.get("umbrella_barriers")
    if isinstance(existing_barriers, Mapping):
        barriers_payload = dict(existing_barriers)
    else:
        barriers_payload = {}

    barriers_payload.update(normalized_barriers)
    payload_dict["umbrella_barriers"] = barriers_payload
    payload_dict["umbrella_ready"] = bool(all_ready)
    payload_dict.setdefault("sid", normalized_sid)
    payload_dict["updated_at"] = ts
    payload_dict["stages"] = stages

    return barriers_payload, merge_ready, validation_ready, review_ready, all_ready, ts


def _reset_step_counters(sid: str, stage: str) -> None:
    if not _STEP_CALL_COUNTS:
        return
    keys_to_delete = [key for key in _STEP_CALL_COUNTS if key[0] == sid and key[1] == stage]
    for key in keys_to_delete:
        del _STEP_CALL_COUNTS[key]


def _watchdog_interval_seconds() -> float:
    if _UMBRELLA_BARRIERS_WATCHDOG_INTERVAL_MS <= 0:
        return 0.0
    return _UMBRELLA_BARRIERS_WATCHDOG_INTERVAL_MS / 1000.0


def _ensure_watchdog_loop() -> Optional[asyncio.AbstractEventLoop]:
    interval = _watchdog_interval_seconds()
    if interval <= 0 or not _UMBRELLA_BARRIERS_ENABLED:
        return None

    global _WATCHDOG_LOOP, _WATCHDOG_THREAD

    loop = _WATCHDOG_LOOP
    if loop is not None and loop.is_running():
        return loop

    loop = asyncio.new_event_loop()
    _WATCHDOG_LOOP = loop

    def _runner() -> None:
        global _WATCHDOG_LOOP, _WATCHDOG_THREAD
        asyncio.set_event_loop(loop)
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            with _WATCHDOG_LOCK:
                if _WATCHDOG_LOOP is loop:
                    _WATCHDOG_LOOP = None
                if _WATCHDOG_THREAD is threading.current_thread():
                    _WATCHDOG_THREAD = None

    thread = threading.Thread(
        target=_runner,
        name="runflow-barriers-watchdog",
        daemon=True,
    )
    _WATCHDOG_THREAD = thread
    thread.start()
    return loop


def _launch_watchdog_if_needed(sid: str) -> None:
    if not sid or not _UMBRELLA_BARRIERS_ENABLED:
        return

    interval = _watchdog_interval_seconds()
    if interval <= 0:
        return

    with _WATCHDOG_LOCK:
        existing = _WATCHDOG_FUTURES.get(sid)
        if existing is not None and not existing.done():
            return

        loop = _ensure_watchdog_loop()
        if loop is None:
            return

        try:
            future = asyncio.run_coroutine_threadsafe(runflow_barriers_watchdog(sid), loop)
        except RuntimeError:
            loop = _ensure_watchdog_loop()
            if loop is None:
                return
            future = asyncio.run_coroutine_threadsafe(runflow_barriers_watchdog(sid), loop)

        _WATCHDOG_FUTURES[sid] = future

        def _cleanup(done: Future[Any]) -> None:
            with _WATCHDOG_LOCK:
                current = _WATCHDOG_FUTURES.get(sid)
                if current is done:
                    _WATCHDOG_FUTURES.pop(sid, None)

        future.add_done_callback(_cleanup)


async def runflow_barriers_watchdog(sid: str) -> None:
    """Periodically reconcile umbrella readiness until ``sid`` is ready."""

    if not sid or not _UMBRELLA_BARRIERS_ENABLED:
        return

    normalized_sid = str(sid).strip()
    if not normalized_sid:
        return

    interval = _watchdog_interval_seconds()
    if interval <= 0:
        return

    base_dir = RUNS_ROOT / normalized_sid
    runflow_path = base_dir / "runflow.json"

    while True:
        try:
            runflow_barriers_refresh(normalized_sid)
        except Exception:  # pragma: no cover - defensive logging
            _LOG.debug(
                "[Runflow] Watchdog refresh failed sid=%s", normalized_sid, exc_info=True
            )

        runflow_payload = _load_json_mapping(runflow_path)
        ready = False
        if isinstance(runflow_payload, Mapping):
            barriers_payload = runflow_payload.get("umbrella_barriers")
            if isinstance(barriers_payload, Mapping):
                ready = bool(barriers_payload.get("all_ready"))

        if ready:
            break

        try:
            await asyncio.sleep(interval)
        except asyncio.CancelledError:
            raise


def runflow_barriers_refresh(sid: str) -> Optional[dict[str, Any]]:
    """Recalculate umbrella readiness flags for ``sid``."""

    if not _UMBRELLA_BARRIERS_ENABLED:
        return None

    normalized_sid = str(sid or "").strip()
    if not normalized_sid:
        return None

    base_dir = RUNS_ROOT / normalized_sid

    try:
        if not base_dir.exists() or not base_dir.is_dir():
            return None
    except OSError:
        return None

    runflow_path = base_dir / "runflow.json"
    try:
        from backend.runflow.decider import reconcile_umbrella_barriers
    except Exception:  # pragma: no cover - defensive import
        reconcile_umbrella_barriers = None  # type: ignore[assignment]

    if reconcile_umbrella_barriers is None:
        # Fall back to the legacy in-place computation.
        normalized_barriers = _apply_umbrella_barriers(
            {}, sid=normalized_sid
        )  # type: ignore[arg-type]
        if normalized_barriers is None:
            return None
        barriers_payload, *_rest = normalized_barriers
        legacy_result: dict[str, Any] = {
            "merge_ready": bool(barriers_payload.get("merge_ready")),
            "validation_ready": bool(barriers_payload.get("validation_ready")),
            "review_ready": bool(barriers_payload.get("review_ready")),
            "all_ready": bool(barriers_payload.get("all_ready")),
            "checked_at": barriers_payload.get("checked_at"),
        }
        if _document_verifier_enabled():
            legacy_result.setdefault(
                "document_ready", bool(barriers_payload.get("document_ready", False))
            )
        return legacy_result

    statuses: Optional[Mapping[str, Any]]
    try:
        statuses = reconcile_umbrella_barriers(normalized_sid)
    except Exception:
        return None

    try:
        payload = json.loads(runflow_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        payload = {}

    if _UMBRELLA_BARRIERS_LOG and isinstance(statuses, Mapping):
        _LOG.info(
            "[Runflow] Umbrella barriers: merge=%s validation=%s review=%s all_ready=%s",
            bool(statuses.get("merge_ready")),
            bool(statuses.get("validation_ready")),
            bool(statuses.get("review_ready")),
            bool(statuses.get("all_ready")),
        )

    barriers_payload = payload.get("umbrella_barriers")
    normalized: dict[str, Any] = {
        "merge_ready": bool(statuses.get("merge_ready")) if statuses else False,
        "validation_ready": bool(statuses.get("validation_ready")) if statuses else False,
        "review_ready": bool(statuses.get("review_ready")) if statuses else False,
        "all_ready": bool(statuses.get("all_ready")) if statuses else False,
        "checked_at": None,
    }

    if isinstance(barriers_payload, Mapping):
        normalized["merge_ready"] = bool(barriers_payload.get("merge_ready"))
        normalized["validation_ready"] = bool(barriers_payload.get("validation_ready"))
        normalized["review_ready"] = bool(barriers_payload.get("review_ready"))
        normalized["all_ready"] = bool(barriers_payload.get("all_ready"))
        normalized["checked_at"] = barriers_payload.get("checked_at")
        if _document_verifier_enabled():
            normalized["document_ready"] = bool(
                barriers_payload.get("document_ready", False)
            )
    elif _document_verifier_enabled():
        normalized["document_ready"] = bool(
            statuses.get("document_ready") if statuses else False
        )

    if normalized.get("checked_at") is None and isinstance(barriers_payload, Mapping):
        normalized["checked_at"] = barriers_payload.get("checked_at")

    return normalized


def _debounced_barriers_refresh_worker(sid: str, token: object) -> None:
    try:
        runflow_barriers_refresh(sid)
    except Exception:  # pragma: no cover - defensive logging
        _LOG.debug(
            "[Runflow] Debounced barriers refresh failed sid=%s", sid, exc_info=True
        )
    finally:
        with _UMBRELLA_REFRESH_LOCK:
            existing = _UMBRELLA_REFRESH_TIMERS.get(sid)
            if existing and existing[1] is token:
                _UMBRELLA_REFRESH_TIMERS.pop(sid, None)


def _schedule_umbrella_barriers_refresh(sid: str) -> None:
    if not _UMBRELLA_BARRIERS_ENABLED:
        return

    normalized_sid = str(sid or "").strip()
    if not normalized_sid:
        return

    should_run_now = False
    with _UMBRELLA_REFRESH_LOCK:
        debounce_ms = _UMBRELLA_BARRIERS_DEBOUNCE_MS
        if debounce_ms <= 0:
            should_run_now = True
            existing = _UMBRELLA_REFRESH_TIMERS.pop(normalized_sid, None)
            if existing:
                existing_timer, _ = existing
                existing_timer.cancel()
        else:
            delay = debounce_ms / 1000.0
            token = object()
            timer = threading.Timer(
                delay, _debounced_barriers_refresh_worker, args=(normalized_sid, token)
            )
            timer.daemon = True
            existing = _UMBRELLA_REFRESH_TIMERS.get(normalized_sid)
            if existing:
                existing_timer, _ = existing
                existing_timer.cancel()
            _UMBRELLA_REFRESH_TIMERS[normalized_sid] = (timer, token)
            timer.start()

    if should_run_now:
        runflow_barriers_refresh(normalized_sid)


def _update_umbrella_barriers(sid: str) -> None:
    _schedule_umbrella_barriers_refresh(sid)


def runflow_refresh_umbrella_barriers(sid: str) -> None:
    """Re-evaluate umbrella readiness for ``sid``."""

    _schedule_umbrella_barriers_refresh(sid)


def _should_record_step(
    sid: str, stage: str, substage: str, step: str, status: str
) -> bool:
    if status != "success":
        return True
    if _STEP_SAMPLE_EVERY <= 1:
        key = (sid, stage, substage, step)
        _STEP_CALL_COUNTS[key] += 1
        return True

    key = (sid, stage, substage, step)
    _STEP_CALL_COUNTS[key] += 1
    count = _STEP_CALL_COUNTS[key]
    return count == 1 or count % _STEP_SAMPLE_EVERY == 0


def _shorten_message(message: str, *, limit: int = 200) -> str:
    compact = " ".join(message.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "\u2026"


def _format_error_payload(exc: Exception, where: str) -> dict[str, str]:
    message = str(exc) or exc.__class__.__name__
    short = _shorten_message(message)
    return {
        "type": exc.__class__.__name__,
        "message": short,
        "where": where,
        "hint": "see runflow_events.jsonl",
    }


def runflow_start_stage(
    sid: str, stage: str, extra: Optional[Mapping[str, Any]] = None
) -> None:
    steps_enabled = _ENABLE_STEPS
    events_enabled = _ENABLE_EVENTS
    _launch_watchdog_if_needed(sid)
    if not (steps_enabled or events_enabled):
        return

    ts = _utcnow_iso()
    key = (sid, stage)
    created = False
    _reset_step_counters(sid, stage)
    if steps_enabled:
        steps_init(sid)
        created = steps_stage_start(sid, stage, started_at=ts, extra=extra)
    else:
        created = key not in _STARTED_STAGES

    if events_enabled and created:
        _append_event(sid, {"ts": ts, "stage": stage, "event": "start"})

    if created:
        _STARTED_STAGES.add(key)


def _first_int_from_candidates(key: str, *candidates: Any) -> Optional[int]:
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        value = _coerce_int(candidate.get(key))
        if value is not None:
            return value
    return None


def _stage_end_event_counters(
    stage: str, summary: Optional[Mapping[str, Any]]
) -> dict[str, Mapping[str, int]]:
    counters: dict[str, Mapping[str, int]] = {}
    if not isinstance(summary, Mapping):
        return counters

    stage_key = str(stage)
    if stage_key == "merge":
        result_files = _first_int_from_candidates(
            "result_files",
            summary,
            summary.get("results"),
        )
        if result_files is not None:
            counters["merge"] = {"result_files": result_files}
        return counters

    if stage_key == "validation":
        results_candidate = summary.get("results")
        results_payload = results_candidate if isinstance(results_candidate, Mapping) else None
        payload: dict[str, int] = {}
        for key in ("results_total", "completed", "failed"):
            value = _first_int_from_candidates(key, summary, results_payload)
            if value is not None:
                payload[key] = value
        if payload:
            counters["validation"] = payload
        return counters

    if stage_key == "frontend":
        metrics_candidate = summary.get("metrics")
        metrics_payload = metrics_candidate if isinstance(metrics_candidate, Mapping) else None
        payload: dict[str, int] = {}
        for key in ("answers_required", "answers_received"):
            value = _first_int_from_candidates(key, summary, metrics_payload)
            if value is not None:
                payload[key] = value
        if payload:
            counters["frontend"] = payload
        return counters

    return counters


def _normalize_barrier_flags(barriers: Optional[Mapping[str, Any]]) -> dict[str, bool]:
    keys = (
        "merge_ready",
        "validation_ready",
        "review_ready",
        "style_ready",
        "all_ready",
        "merge_zero_packs",
    )
    result: dict[str, bool] = {}
    for key in keys:
        value = False
        if isinstance(barriers, Mapping) and key in barriers:
            raw = barriers.get(key)
            value = bool(raw)
        result[key] = value
    return result


def runflow_end_stage(
    sid: str,
    stage: str,
    *,
    status: str = "success",
    summary: Optional[Mapping[str, Any]] = None,
    stage_status: Optional[str] = None,
    empty_ok: bool = False,
    barriers: Optional[Mapping[str, Any]] = None,
    umbrella_ready: Optional[bool] = None,
    refresh_barriers: bool = True,
) -> None:
    steps_enabled = _ENABLE_STEPS
    events_enabled = _ENABLE_EVENTS

    if steps_enabled or events_enabled:
        ts = _utcnow_iso()
        if steps_enabled:
            status_for_steps = stage_status or status
            steps_stage_finish(
                sid,
                stage,
                status_for_steps,
                summary,
                ended_at=ts,
                empty_ok=empty_ok,
            )

        if events_enabled:
            counters_payload = _stage_end_event_counters(stage, summary)
            barrier_flags = _normalize_barrier_flags(barriers)
            event: dict[str, Any] = {
                "ts": ts,
                "t": ts,
                "sid": str(sid),
                "stage": stage,
                "event": "end",
                "status": status,
                "counters": counters_payload,
                "umbrella_barriers": barrier_flags,
            }
            if summary:
                event["summary"] = {str(k): v for k, v in summary.items()}
            if barriers:
                event["barriers"] = {str(k): v for k, v in barriers.items()}
            if umbrella_ready is not None:
                event["umbrella_ready"] = bool(umbrella_ready)
            _append_event(sid, event)

    _STARTED_STAGES.discard((sid, stage))
    _reset_step_counters(sid, stage)

    if refresh_barriers:
        _update_umbrella_barriers(sid)

    if stage == "validation":
        _clear_stage_counters(sid, "validation_build", "validation_results")
        if _aggregates_enabled():
            _clear_stage_aggregate(sid, "validation")
    elif stage == "frontend":
        _clear_stage_counters(sid, "frontend_review")
        if _aggregates_enabled():
            _clear_stage_aggregate(sid, "review")


def runflow_event(
    sid: str,
    stage: str,
    step: str,
    *,
    status: str = "success",
    account: Optional[str] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    out: Optional[Mapping[str, Any]] = None,
    substage: Optional[str] = None,
    reason: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    error: Optional[Mapping[str, Any]] = None,
) -> None:
    """Emit a runflow event without recording a step entry."""

    if not _ENABLE_EVENTS:
        return

    ts = _utcnow_iso()
    substage_name = substage or "default"
    event: dict[str, Any] = {
        "ts": ts,
        "stage": stage,
        "step": step,
        "status": status,
        "substage": substage_name,
    }
    if account is not None:
        event["account"] = account
    if metrics:
        event["metrics"] = {str(k): v for k, v in metrics.items()}
    if out:
        event["out"] = {str(k): v for k, v in out.items()}
    if reason is not None:
        event["reason"] = reason
    if span_id is not None:
        event["span_id"] = span_id
    if parent_span_id is not None:
        event["parent_span_id"] = parent_span_id
    if error:
        event["error"] = {str(k): v for k, v in error.items()}
    _append_event(sid, event)


def runflow_decide_step(
    sid: str,
    stage: str,
    *,
    next_action: str,
    reason: str,
) -> None:
    """Record a compact decision step for ``stage`` if runflow is enabled."""

    steps_enabled = _ENABLE_STEPS
    events_enabled = _ENABLE_EVENTS

    if not (steps_enabled or events_enabled):
        return

    ts = _utcnow_iso()
    stage_name = str(stage)

    decision_out = {"next": str(next_action), "reason": str(reason)}

    if steps_enabled:
        steps_append(
            sid,
            stage_name,
            "runflow_decide",
            "success",
            t=ts,
            out=decision_out,
        )

    if events_enabled:
        event: dict[str, Any] = {
            "ts": ts,
            "stage": stage_name,
            "event": "decide",
            "next": str(next_action),
            "reason": str(reason),
        }
        _append_event(sid, event)


def runflow_step(
    sid: str,
    stage: str,
    step: str,
    *,
    status: str = "success",
    account: Optional[str] = None,
    metrics: Optional[Mapping[str, Any]] = None,
    out: Optional[Mapping[str, Any]] = None,
    substage: Optional[str] = None,
    reason: Optional[str] = None,
    span_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
    error: Optional[Mapping[str, Any]] = None,
) -> None:
    steps_enabled = _ENABLE_STEPS
    events_enabled = _ENABLE_EVENTS

    if not (steps_enabled or events_enabled):
        return

    ts = _utcnow_iso()
    substage_name = substage or "default"

    event_step = step
    event_status = status

    step_for_steps: Optional[str] = step
    status_for_steps = status

    if stage == "merge":
        if event_step == "merge_scoring":
            if event_status == "start":
                step_for_steps = "merge_scoring_start"
                status_for_steps = "success"
            else:
                step_for_steps = "merge_scoring_finish"
                status_for_steps = event_status
        elif event_step == "pack_skip" and event_status == "success":
            step_for_steps = None

    record_step_success = True
    if step_for_steps is not None and status_for_steps == "success":
        record_step_success = _should_record_step(
            sid, stage, substage_name, step_for_steps, status_for_steps
        )
        if not record_step_success:
            return
    elif step_for_steps is not None and status_for_steps != "success":
        _STEP_CALL_COUNTS[(sid, stage, substage_name, step_for_steps)] += 1

    if steps_enabled and step_for_steps is not None and record_step_success:
        step_span_id = span_id if _ENABLE_SPANS else None
        step_parent_span_id = parent_span_id if _ENABLE_SPANS else None
        should_write_step = True
        if stage == "merge":
            allowed_success_steps = {
                "merge_scoring_start",
                "acctnum_normalize",
                "acctnum_match_level",
                "acctnum_pairs_summary",
                "no_merge_candidates",
                "pack_create",
                "merge_zero_packs",
                "merge_scoring_finish",
            }
            if status_for_steps == "success":
                should_write_step = step_for_steps in allowed_success_steps
            else:
                should_write_step = step_for_steps == "merge_scoring_finish"
        if (
            _SUPPRESS_ACCOUNT_STEPS
            and account is not None
            and status_for_steps == "success"
        ):
            should_write_step = False
        if (
            _aggregates_enabled()
            and stage in {"validation", "frontend"}
            and status_for_steps == "success"
            and error is None
        ):
            should_write_step = False
        if should_write_step:
            steps_append(
                sid,
                stage,
                step_for_steps,
                status_for_steps,
                t=ts,
                account=account,
                metrics=metrics,
                out=out,
                reason=reason,
                span_id=step_span_id,
                parent_span_id=step_parent_span_id,
                error=error,
            )
        elif stage == "merge":
            key = (sid, stage)
            if key not in _STARTED_STAGES:
                steps_stage_start(sid, stage, started_at=ts)
                _STARTED_STAGES.add(key)

    if events_enabled:
        event: dict[str, Any] = {
            "ts": ts,
            "stage": stage,
            "step": event_step,
            "status": event_status,
            "substage": substage_name,
        }
        if account is not None:
            event["account"] = account
        if metrics:
            event["metrics"] = {str(k): v for k, v in metrics.items()}
        if out:
            event["out"] = {str(k): v for k, v in out.items()}
        if reason is not None:
            event["reason"] = reason
        if span_id is not None:
            event["span_id"] = span_id
        if parent_span_id is not None:
            event["parent_span_id"] = parent_span_id
        if error:
            event["error"] = {str(k): v for k, v in error.items()}
        _append_event(sid, event)


def record_validation_build_summary(
    sid: str,
    *,
    eligible_accounts: Any,
    packs_built: Any,
    packs_skipped: Any,
) -> None:
    summary = {
        "eligible_accounts": eligible_accounts,
        "packs_built": packs_built,
        "packs_skipped": packs_skipped,
    }
    normalized = _store_stage_counter(sid, "validation_build", summary)
    if _aggregates_enabled():
        stage_state = _aggregate_state(sid, "validation")
        _aggregate_prune(stage_state, _VALIDATION_AGGREGATE_KEYS)
        total_updated = _aggregate_set_nonnegative(
            stage_state, "packs_total", normalized.get("eligible_accounts")
        )
        if not total_updated and "packs_total" not in stage_state:
            built_value = normalized.get("packs_built")
            skipped_value = normalized.get("packs_skipped")
            candidate_total: Optional[int]
            try:
                built_int = int(built_value)
                skipped_int = int(skipped_value)
            except (TypeError, ValueError):
                candidate_total = None
            else:
                candidate_total = built_int + skipped_int
            if candidate_total is not None:
                _aggregate_set_nonnegative(stage_state, "packs_total", candidate_total)
        _aggregate_set_nonnegative(stage_state, "packs_completed", normalized.get("packs_built"))
        stage_state.pop("packs_pending", None)
        _write_stage_aggregate(sid, "validation")
    _emit_summary_step(sid, "validation", "build_packs", summary=normalized)


def record_validation_results_summary(
    sid: str,
    *,
    results_total: Any,
    completed: Any,
    failed: Any,
    pending: Any,
) -> None:
    summary = {
        "results_total": results_total,
        "completed": completed,
        "failed": failed,
        "pending": pending,
    }
    normalized = _store_stage_counter(sid, "validation_results", summary)
    if normalized.get("pending", 0) < 0:
        normalized["pending"] = 0
    if _aggregates_enabled():
        stage_state = _aggregate_state(sid, "validation")
        _aggregate_prune(stage_state, _VALIDATION_AGGREGATE_KEYS)
        _aggregate_set_nonnegative(stage_state, "packs_total", normalized.get("results_total"))
        _aggregate_set_nonnegative(stage_state, "packs_completed", normalized.get("completed"))
        pending_provided = _aggregate_set_nonnegative(
            stage_state, "packs_pending", normalized.get("pending")
        )
        if not pending_provided:
            stage_state.pop("packs_pending", None)
        _write_stage_aggregate(sid, "validation")
    _emit_summary_step(sid, "validation", "collect_results", summary=normalized)


def record_frontend_responses_progress(
    sid: str,
    *,
    accounts_published: Any,
    answers_received: Any,
    answers_required: Any,
) -> None:
    base_dir = RUNS_ROOT / sid
    counters = _frontend_answers_counters(
        base_dir, attachments_required=_review_attachment_required()
    )

    answers_required_disk = counters.get("answers_required")
    answers_received_disk = counters.get("answers_received")

    if isinstance(answers_required_disk, int):
        accounts_value = answers_required_disk
    else:
        accounts_value = accounts_published

    summary = {
        "accounts_published": accounts_value,
        "answers_received": answers_received_disk
        if isinstance(answers_received_disk, int)
        else answers_received,
        "answers_required": answers_required_disk
        if isinstance(answers_required_disk, int)
        else answers_required,
    }
    normalized = _store_stage_counter(sid, "frontend_review", summary)
    if _aggregates_enabled():
        stage_state = _aggregate_state(sid, "review")
        _aggregate_prune(stage_state, _REVIEW_AGGREGATE_KEYS)
        _aggregate_set_nonnegative(
            stage_state, "answers_received", normalized.get("answers_received")
        )
        _aggregate_set_nonnegative(
            stage_state, "answers_required", normalized.get("answers_required")
        )
        _write_stage_aggregate(sid, "review")
    _emit_summary_step(sid, "frontend", "responses_progress", summary=normalized)


def runflow_step_dec(stage: str, step: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def inner(*args: Any, **kwargs: Any) -> Any:
            if not (_ENABLE_STEPS or _ENABLE_EVENTS):
                return fn(*args, **kwargs)

            sid: Optional[str] = None
            if "sid" in kwargs and isinstance(kwargs["sid"], str):
                sid = kwargs["sid"]
            elif args:
                candidate = args[0]
                if hasattr(candidate, "sid"):
                    sid_value = getattr(candidate, "sid")
                    if isinstance(sid_value, str):
                        sid = sid_value
                elif isinstance(candidate, Mapping):
                    sid_value = candidate.get("sid")
                    if isinstance(sid_value, str):
                        sid = sid_value

            where = f"{fn.__module__}:{getattr(fn, '__qualname__', fn.__name__)}"

            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                if sid:
                    error_payload = _format_error_payload(exc, where)
                    runflow_step(
                        sid,
                        stage,
                        step,
                        status="error",
                        error=error_payload,
                    )
                raise

            if sid:
                runflow_step(sid, stage, step, status="success")
            return result

        return inner

    return _wrap


__all__ = [
    "runflow_start_stage",
    "runflow_end_stage",
    "runflow_barriers_refresh",
    "runflow_barriers_watchdog",
    "runflow_refresh_umbrella_barriers",
    "runflow_decide_step",
    "runflow_event",
    "runflow_step",
    "runflow_step_dec",
    "steps_pair_topn",
    "record_validation_build_summary",
    "record_validation_results_summary",
    "record_frontend_responses_progress",
    "runflow_account_steps_enabled",
]
