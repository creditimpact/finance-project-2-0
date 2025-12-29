"""Celery task chain used by the automatic AI adjudication pipeline."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
import time
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from celery import chain, shared_task

from backend.ai.manifest import ensure_validation_section
from backend.ai.validation_builder import build_validation_packs_for_run
from backend.core.ai.paths import (
    ensure_merge_paths,
    merge_result_glob_pattern,
    parse_pair_result_filename,
    probe_legacy_ai_packs,
    validation_index_path,
    validation_results_dir,
)
from backend.core.logic import polarity
from backend.pipeline.auto_ai import (
    INFLIGHT_LOCK_FILENAME,
    LAST_OK_FILENAME,
    _build_ai_packs,
    _compact_accounts,
    _indices_from_index,
    _load_ai_index,
    _normalize_indices,
    _send_ai_packs,
    has_ai_merge_best_pairs,
    run_consistency_writeback_for_all_accounts,
    run_strategy_planner_for_all_accounts,
    run_validation_requirements_for_all_accounts,
)
from backend.core.logic.validation_ai_merge import (
    apply_validation_ai_decisions_for_all_accounts,
    summarize_validation_ai_state,
)
from backend.core.runflow import runflow_barriers_refresh, runflow_step
from backend.core.runflow.io import (
    compose_hint,
    format_exception_tail,
    runflow_stage_error,
    runflow_stage_start,
)
from backend.runflow.decider import (
    StageStatus,
    decide_next,
    finalize_merge_stage,
    get_runflow_snapshot,
    record_stage,
    record_stage_force,
    reconcile_umbrella_barriers,
    release_validation_fastpath_lock,
    _validation_results_progress,
)
from backend.frontend.packs.generator import generate_frontend_packs_for_run
from backend.api.tasks import enqueue_generate_frontend_packs
from backend.runflow.manifest import (
    update_manifest_frontend,
    update_manifest_state,
)
from backend.prevalidation.tasks import (
    detect_and_persist_date_convention,
    run_date_convention_detector,
)
from backend.core.ai.validators import validate_ai_result
from backend.core.io.tags import read_tags, upsert_tag
from backend.validation.manifest import rewrite_index_to_canonical_layout
from backend.validation.send_packs import send_validation_packs
from backend.ai.validation_builder import run_validation_send_for_sid
from scripts.score_bureau_pairs import score_accounts
from backend.pipeline.runs import RunManifest

LEGACY_PIPELINE_DIRNAME = "ai_packs"
LEGACY_MARKER_FILENAME = "auto_ai_pipeline_in_progress.json"

logger = logging.getLogger(__name__)

# Strict Validation Pipeline controls (Phase 1)
def _strict_validation_pipeline_enabled() -> bool:
    raw = os.getenv("VALIDATION_STRICT_PIPELINE")
    if raw is None:
        return True
    lowered = str(raw).strip().lower()
    return lowered not in {"", "0", "false", "no", "off"}

# Phase 2 Orchestrator mode
def _orchestrator_mode_enabled() -> bool:
    raw = os.getenv("VALIDATION_ORCHESTRATOR_MODE")
    if raw is None:
        return True
    lowered = str(raw).strip().lower()
    return lowered not in {"", "0", "false", "no", "off"}

def _delivery_lock_path(runs_root: Path, sid: str) -> Path:
    return runs_root / sid / ".locks" / "validation_delivery.lock"

def _wait_lock_path(runs_root: Path, sid: str) -> Path:
    return runs_root / sid / ".locks" / "validation_results_wait.lock"

def _lock_set(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    except Exception:
        logger.debug("VALIDATION_LOCK_SET_FAILED path=%s", path, exc_info=True)

def _lock_clear(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.debug("VALIDATION_LOCK_CLEAR_FAILED path=%s", path, exc_info=True)

def _lock_exists(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def _maybe_autobuild_review(
    sid: str,
    *,
    already_triggered: bool = False,
    source: str = "validation",
) -> bool:
    """Kick off Frontend/Review packs build after validation has run.

    Returns ``True`` when a build was enqueued during this invocation.
    """

    if already_triggered:
        logger.info(
            "REVIEW_AUTO: skip enqueue sid=%s source=%s reason=already_triggered",
            sid,
            source,
        )
        return False

    if os.getenv("GENERATE_FRONTEND_ON_VALIDATION", "1") != "1":
        logger.info(
            "REVIEW_AUTO: skip enqueue sid=%s source=%s reason=env_disabled",
            sid,
            source,
        )
        return False

    enqueue_generate_frontend_packs(sid)
    logger.info(
        "REVIEW_AUTO: queued_generate_frontend_packs sid=%s source=%s",
        sid,
        source,
    )
    # Discover actual validation result files
    try:
        results_dir = validation_results_dir(sid, runs_root=runs_root, create=True)
        result_files = list(results_dir.glob("*.json*"))
        logger.info(
            "VALIDATION_SEND_RESULTS_DISCOVERED sid=%s dir=%s count=%d",
            sid,
            results_dir,
            len(result_files),
        )
        if len(result_files) == 0:
            logger.error(
                "VALIDATION_SEND_NO_RESULTS_FOUND sid=%s dir=%s (empty)",
                sid,
                results_dir,
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "VALIDATION_SEND_RESULTS_DISCOVERY_FAILED sid=%s error=%s",
            sid,
            exc.__class__.__name__,
            exc_info=True,
        )
    return True


_PAIR_TAG_BY_DECISION: dict[str, str] = {
    "same_account_same_debt": "same_account_pair",
    "same_account_diff_debt": "same_account_pair",
    "same_account_debt_unknown": "same_account_pair",
    "same_debt_diff_account": "same_debt_pair",
    "same_debt_account_unknown": "same_debt_pair",
}


def _bool_env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    return lowered not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value >= 0 else default


def _validation_merge_wait_settings() -> tuple[int, int]:
    return (
        _env_int("VALIDATION_MERGE_WAIT_SECONDS", 90),
        _env_int("VALIDATION_MERGE_POLL_SECONDS", 5),
    )


def _await_validation_results(sid: str, runs_root_path: Path):
    run_dir = runs_root_path / sid
    progress = _validation_results_progress(run_dir)
    max_wait, poll_interval = _validation_merge_wait_settings()
    try:
        index_exists = validation_index_path(sid, runs_root=runs_root_path, create=False).exists()
        existing_result_files = list(results_dir.glob("*.json*")) if results_dir.is_dir() else []
        if index_exists and not existing_result_files:
            logger.warning(
                "VALIDATION_MERGE_NO_RESULTS_YET sid=%s dir=%s",
                sid,
                results_dir,
            )
    except Exception:  # pragma: no cover - defensive
        logger.debug("VALIDATION_MERGE_PRECHECK_FAILED sid=%s", sid, exc_info=True)

    if (
        progress.total == 0
        or progress.ready
        or progress.failed > 0
        or max_wait <= 0
        or poll_interval < 0
    ):
        return progress

    deadline = time.monotonic() + max_wait
    attempt = 0

    while progress.failed == 0 and not progress.ready and time.monotonic() < deadline:
        attempt += 1
        pending_accounts = getattr(progress, "missing_accounts", ()) or getattr(progress, "pack_accounts", ())
        pending_display = ",".join(pending_accounts) if pending_accounts else "<none>"
        logger.info(
            "VALIDATION_MERGE_WAIT sid=%s pending=%s completed=%s total=%s attempt=%d",
            sid,
            pending_display,
            progress.completed,
            progress.total,
            attempt,
        )

        remaining = max(0.0, deadline - time.monotonic())
        sleep_time = min(poll_interval, remaining)
        if sleep_time > 0:
            time.sleep(sleep_time)

        progress = _validation_results_progress(run_dir)

    return progress


def _validation_zero_fastpath_enabled() -> bool:
    return _bool_env_flag("VALIDATION_ZERO_PACKS_FASTPATH", True)


def _ensure_zero_pack_validation_index(
    sid: str,
    runs_root: Path,
    *,
    merge_zero_packs: bool,
) -> Path:
    index_path = validation_index_path(sid, runs_root=runs_root, create=True)
    if index_path.exists():
        return index_path

    index_path.parent.mkdir(parents=True, exist_ok=True)
    validation_results_dir(sid, runs_root=runs_root).mkdir(parents=True, exist_ok=True)

    payload = {
        "sid": sid,
        "generated_at": _isoformat_timestamp(),
        "packs": [],
        "totals": {
            "packs_total": 0,
            "merge_zero_packs": bool(merge_zero_packs),
        },
    }

    try:
        index_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError:
        logger.warning(
            "VALIDATION_FASTPATH_INDEX_WRITE_FAILED sid=%s path=%s",
            sid,
            index_path,
            exc_info=True,
        )

    return index_path


def _append_runflow_event(run_dir: Path, payload: Mapping[str, object]) -> None:
    events_path = run_dir / "runflow_events.jsonl"
    try:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), ensure_ascii=False))
            handle.write("\n")
    except OSError:
        logger.warning(
            "RUNFLOW_EVENT_WRITE_FAILED sid=%s path=%s",
            run_dir.name,
            events_path,
            exc_info=True,
        )


def _isoformat_timestamp(now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    return current.isoformat(timespec="seconds").replace("+00:00", "Z")


def _serialize_match_flag(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false", "unknown"}:
            return lowered
    return "unknown"


def _load_ai_results(results_dir: Path) -> list[tuple[int, int, dict[str, object]]]:
    if not results_dir.exists():
        return []

    pairs: list[tuple[int, int, dict[str, object]]] = []
    pattern = merge_result_glob_pattern()
    for path in sorted(results_dir.glob(pattern)):
        parsed = parse_pair_result_filename(path.name)
        if not parsed:
            logger.debug("AUTO_AI_RESULT_SKIP_UNMATCHED path=%s", path)
            continue

        a_idx, b_idx = parsed

        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("AUTO_AI_RESULT_INVALID_JSON path=%s", path, exc_info=True)
            continue

        if not isinstance(loaded, Mapping):
            logger.warning("AUTO_AI_RESULT_INVALID_TYPE path=%s", path)
            continue

        payload = dict(loaded)
        flags_obj = payload.get("flags")
        if not isinstance(flags_obj, Mapping):
            logger.warning("AUTO_AI_RESULT_MISSING_FLAGS path=%s", path)
            continue

        try:
            valid, error = validate_ai_result(
                {"decision": payload.get("decision"), "reason": payload.get("reason"), "flags": dict(flags_obj)}
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.warning("AUTO_AI_RESULT_VALIDATION_ERROR path=%s", path, exc_info=True)
            continue

        if not valid:
            logger.warning(
                "AUTO_AI_RESULT_INVALID sid_pair=%s_%s path=%s error=%s",
                a_idx,
                b_idx,
                path,
                error or "unknown",
            )
            continue

        payload["flags"] = dict(flags_obj)
        pairs.append((a_idx, b_idx, payload))

    return pairs


def _prune_pair_tags(tag_path: Path, other_idx: int, *, keep_kind: str | None) -> None:
    existing_tags = read_tags(tag_path)
    if not existing_tags:
        return

    filtered: list[dict[str, object]] = []
    modified = False
    for entry in existing_tags:
        kind = str(entry.get("kind", "")).strip().lower()
        if kind not in {"same_account_pair", "same_debt_pair"}:
            filtered.append(dict(entry))
            continue

        source = str(entry.get("source", ""))
        if source != "ai_adjudicator":
            filtered.append(dict(entry))
            continue

        partner_raw = entry.get("with")
        try:
            partner_val = int(partner_raw) if partner_raw is not None else None
        except (TypeError, ValueError):
            partner_val = None

        if partner_val != other_idx:
            filtered.append(dict(entry))
            continue

        if keep_kind is not None and kind == keep_kind:
            filtered.append(dict(entry))
            continue

        modified = True

    if not modified:
        return

    tag_path.write_text(json.dumps(filtered, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _apply_ai_result_to_accounts(
    accounts_dir: Path, a_idx: int, b_idx: int, payload: Mapping[str, object]
) -> None:
    decision_raw = payload.get("decision")
    decision = str(decision_raw).strip().lower() if isinstance(decision_raw, str) else ""
    reason_raw = payload.get("reason")
    if isinstance(reason_raw, str):
        reason = reason_raw.strip()
    elif reason_raw is None:
        reason = ""
    else:
        reason = str(reason_raw)

    flags_raw = payload.get("flags")
    flags_serialized: dict[str, str] | None = None
    if isinstance(flags_raw, Mapping):
        flags_serialized = {
            "account_match": _serialize_match_flag(flags_raw.get("account_match")),
            "debt_match": _serialize_match_flag(flags_raw.get("debt_match")),
        }

    timestamp = _isoformat_timestamp()
    pair_tag_kind = _PAIR_TAG_BY_DECISION.get(decision)

    normalized_raw = payload.get("normalized")
    normalized_flag = normalized_raw if isinstance(normalized_raw, bool) else None
    raw_response = payload.get("raw_response") if isinstance(payload.get("raw_response"), Mapping) else None

    for source_idx, other_idx in ((a_idx, b_idx), (b_idx, a_idx)):
        account_dir = accounts_dir / f"{source_idx}"
        account_dir.mkdir(parents=True, exist_ok=True)
        tag_path = account_dir / "tags.json"

        decision_tag: dict[str, object] = {
            "kind": "ai_decision",
            "tag": "ai_decision",
            "source": "ai_adjudicator",
            "with": other_idx,
            "decision": decision,
            "reason": reason,
            "at": timestamp,
        }
        if flags_serialized is not None:
            decision_tag["flags"] = dict(flags_serialized)
        if normalized_flag is not None:
            decision_tag["normalized"] = normalized_flag
        if raw_response is not None:
            decision_tag["raw_response"] = dict(raw_response)

        upsert_tag(tag_path, decision_tag, unique_keys=("kind", "with", "source"))

        if pair_tag_kind is not None:
            pair_tag = {
                "kind": pair_tag_kind,
                "with": other_idx,
                "source": "ai_adjudicator",
                "reason": reason,
                "at": timestamp,
            }
            upsert_tag(tag_path, pair_tag, unique_keys=("kind", "with", "source"))
            _prune_pair_tags(tag_path, other_idx, keep_kind=pair_tag_kind)
        else:
            _prune_pair_tags(tag_path, other_idx, keep_kind=None)


def _append_run_log_entry(
    *,
    runs_root: Path,
    sid: str,
    packs: int,
    pairs: int,
    reason: str | None = None,
) -> None:
    """Append a compact JSON line describing the AI run outcome."""

    merge_paths = ensure_merge_paths(runs_root, sid, create=True)
    logs_path = merge_paths.log_file
    entry = {
        "sid": sid,
        "at": datetime.now(timezone.utc).isoformat(),
        "packs": int(packs),
        "pairs": int(pairs),
        "keywords": [
            "CANDIDATE_LOOP_START",
            "CANDIDATE_CONSIDERED",
            "CANDIDATE_SKIPPED",
            "CANDIDATE_LOOP_END",
            "MERGE_V2_ACCT_BEST",
        ],
        "verify": [
            f"rg \"CANDIDATE_(CONSIDERED|SKIPPED)\" {logs_path}",
            f"rg \"MERGE_V2_ACCT_BEST\" {logs_path}",
        ],
    }
    if reason:
        entry["reason"] = reason

    serialized_entry = json.dumps(entry, ensure_ascii=False) + "\n"

    try:
        logs_path.parent.mkdir(parents=True, exist_ok=True)
        with logs_path.open("a", encoding="utf-8") as handle:
            handle.write(serialized_entry)
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "AUTO_AI_LOG_APPEND_FAILED sid=%s path=%s", sid, logs_path, exc_info=True
        )

    legacy_logs_path = runs_root / sid / "ai_packs" / "logs.txt"
    if legacy_logs_path != logs_path:
        try:
            legacy_logs_path.parent.mkdir(parents=True, exist_ok=True)
            with legacy_logs_path.open("a", encoding="utf-8") as handle:
                handle.write(serialized_entry)
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "AUTO_AI_LOG_LEGACY_WRITE_FAILED sid=%s path=%s",
                sid,
                legacy_logs_path,
                exc_info=True,
            )



def _ensure_payload(prev: Mapping[str, object] | None) -> dict[str, object]:
    if isinstance(prev, Mapping):
        return dict(prev)
    return {}


def _resolve_runs_root(payload: Mapping[str, object], sid: str) -> Path:
    runs_root_value = payload.get("runs_root")
    if isinstance(runs_root_value, (str, os.PathLike)):
        return Path(runs_root_value)

    env_root = os.environ.get("RUNS_ROOT")
    if env_root:
        return Path(env_root)

    default_root = Path("runs")

    pipeline_dir = ensure_merge_paths(default_root, sid, create=False).base
    lock_path = pipeline_dir / INFLIGHT_LOCK_FILENAME
    if lock_path.exists():
        try:
            data = json.loads(lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.debug(
                "AUTO_AI_LOCK_READ_FAILED sid=%s path=%s", sid, lock_path, exc_info=True
            )
        else:
            marker_root = data.get("runs_root")
            if isinstance(marker_root, str) and marker_root:
                return Path(marker_root)
        return pipeline_dir.parent.parent

    legacy_dir = default_root / sid / LEGACY_PIPELINE_DIRNAME
    legacy_marker = legacy_dir / LEGACY_MARKER_FILENAME
    if legacy_marker.exists():
        try:
            data = json.loads(legacy_marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.debug(
                "AUTO_AI_LEGACY_MARKER_READ_FAILED sid=%s path=%s",
                sid,
                legacy_marker,
                exc_info=True,
            )
        else:
            marker_root = data.get("runs_root")
            if isinstance(marker_root, str) and marker_root:
                return Path(marker_root)
        return legacy_dir.parent.parent

    return default_root


def _populate_common_paths(payload: MutableMapping[str, object]) -> None:
    sid = str(payload.get("sid") or "")
    if not sid:
        return

    runs_root = _resolve_runs_root(payload, sid)
    accounts_dir = runs_root / sid / "cases" / "accounts"
    merge_paths = ensure_merge_paths(runs_root, sid, create=True)
    pipeline_dir = merge_paths.base
    lock_path = pipeline_dir / INFLIGHT_LOCK_FILENAME
    last_ok_path = pipeline_dir / LAST_OK_FILENAME

    payload["runs_root"] = str(runs_root)
    payload["accounts_dir"] = str(accounts_dir)
    payload["pipeline_dir"] = str(pipeline_dir)
    payload["lock_path"] = str(lock_path)
    payload["marker_path"] = str(lock_path)
    payload["last_ok_path"] = str(last_ok_path)


def _cleanup_lock(payload: Mapping[str, object], *, reason: str) -> bool:
    sid = str(payload.get("sid") or "")
    lock_value = payload.get("lock_path") or payload.get("marker_path")
    if not lock_value:
        return False

    lock_path = Path(str(lock_value))
    try:
        if lock_path.exists():
            lock_path.unlink()
            logger.info(
                "AUTO_AI_LOCK_REMOVED sid=%s reason=%s lock=%s",
                sid,
                reason,
                lock_path,
            )
            return True
        logger.info(
            "AUTO_AI_LOCK_MISSING sid=%s reason=%s lock=%s",
            sid,
            reason,
            lock_path,
        )
        return False
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "AUTO_AI_LOCK_CLEANUP_FAILED sid=%s reason=%s lock=%s",
            sid,
            reason,
            lock_path,
            exc_info=True,
        )
        return False


def _discover_strategy_accounts(accounts_dir: Path) -> list[str]:
    if not accounts_dir.is_dir():
        return []

    planned: list[str] = []
    for account_dir in accounts_dir.iterdir():
        if not account_dir.is_dir():
            continue
        strategy_dir = account_dir / "strategy"
        if not strategy_dir.is_dir():
            continue
        has_plans = False
        try:
            for bureau_dir in strategy_dir.iterdir():
                if not bureau_dir.is_dir():
                    continue
                plan_file = bureau_dir / "plan.json"
                if plan_file.exists():
                    has_plans = True
                    break
                for weekday_plan in bureau_dir.glob("plan_wd*.json"):
                    if weekday_plan.is_file():
                        has_plans = True
                        break
                if has_plans:
                    break
        except OSError:  # pragma: no cover - defensive directory guard
            continue
        if has_plans:
            planned.append(account_dir.name)

    return sorted(planned)


def _load_strategy_manifest_or_raise(
    sid: str,
    runs_root_path: Path | None,
) -> RunManifest:
    """Return a manifest instance for ``sid`` or raise if unavailable."""

    last_error: Exception | None = None
    attempts: list[str] = []

    if runs_root_path is not None:
        manifest_path = runs_root_path / sid / "manifest.json"
        attempts.append(f"load:{manifest_path}")
        if manifest_path.exists():
            try:
                return RunManifest(manifest_path).load()
            except Exception as exc:  # pragma: no cover - defensive manifest load guard
                last_error = exc
                logger.debug(
                    "AUTO_AI_STRATEGY_MANIFEST_LOAD_FAILED sid=%s manifest=%s",
                    sid,
                    manifest_path,
                    exc_info=True,
                )
        else:
            last_error = FileNotFoundError(f"manifest missing at {manifest_path}")

    attempts.append("for_sid")
    try:
        return RunManifest.for_sid(sid, allow_create=False)
    except Exception as exc:  # pragma: no cover - defensive manifest load guard
        last_error = exc
        logger.debug(
            "AUTO_AI_STRATEGY_MANIFEST_FOR_SID_FAILED sid=%s",
            sid,
            exc_info=True,
        )

    attempts_display = ", ".join(attempts) if attempts else "none"
    message = (
        f"Unable to load strategy manifest for sid={sid}; attempts={attempts_display}"
    )
    raise RuntimeError(message) from last_error


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_score_step(self, sid: str, runs_root: str | None = None) -> dict[str, object]:
    """Recompute merge scores and persist merge tags for ``sid``."""

    logger.info("AI_SCORE_START sid=%s", sid)

    payload: dict[str, object] = {
        "sid": sid,
        "_chain_context": True,  # Mark as chain execution
    }
    if runs_root is not None:
        payload["runs_root"] = runs_root
    _populate_common_paths(payload)

    runs_root = Path(payload["runs_root"])

    try:
        result = score_accounts(sid, runs_root=runs_root, write_tags=True)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_SCORE_FAILED sid=%s", sid, exc_info=True)
        _cleanup_lock(payload, reason="score_failed")
        raise

    touched_accounts = sorted(_normalize_indices(result.indices))
    payload["touched_accounts"] = touched_accounts

    logger.info("AI_SCORE_END sid=%s touched=%d", sid, len(touched_accounts))
    return payload


def _merge_build_stage(payload: dict[str, object]) -> dict[str, object]:
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_BUILD_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])

    # Idempotency check: skip if merge already complete
    try:
        from backend.runflow.decider import get_runflow_snapshot
        snapshot = get_runflow_snapshot(sid, runs_root=runs_root)
        merge_stage = snapshot.get("stages", {}).get("merge", {})
        
        # Check all completion indicators
        status = merge_stage.get("status")
        merge_applied = merge_stage.get("merge_ai_applied", False)
        result_files = merge_stage.get("result_files", 0)
        expected = merge_stage.get("expected_packs", 0)
        
        if status == "success" and merge_applied and result_files > 0 and result_files >= expected:
            logger.info(
                "MERGE_BUILD_IDEMPOTENT_SKIP sid=%s reason=already_complete status=%s applied=%s results=%s/%s",
                sid, status, merge_applied, result_files, expected
            )
            payload["merge_packs"] = 0
            payload["merge_skipped"] = True
            payload["skip_reason"] = "already_complete"
            payload["ai_index"] = []
            return payload
    except Exception:
        logger.debug("MERGE_BUILD_IDEMPOTENT_CHECK_FAILED sid=%s", sid, exc_info=True)
        # Continue with normal flow on check failure

    logger.info("AI_BUILD_START sid=%s", sid)

    runflow_stage_start("merge", sid=sid)

    if not has_ai_merge_best_pairs(sid, runs_root):
        logger.info("AUTO_AI_SKIP_NO_CANDIDATES sid=%s", sid)
        logger.info("AUTO_AI_BUILDER_BYPASSED_ZERO_DEBT sid=%s", sid)
        logger.info("AUTO_AI_SKIPPED sid=%s reason=no_candidates", sid)
        payload["ai_index"] = []
        payload["skip_reason"] = "no_candidates"
        runflow_step(
            sid,
            "merge",
            "build",
            status="skipped",
            metrics={"packs": 0},
            out={"reason": "no_candidates"},
        )
        return payload

    try:
        _build_ai_packs(sid, runs_root)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_BUILD_FAILED sid=%s", sid, exc_info=True)
        _cleanup_lock(payload, reason="build_failed")
        runflow_step(
            sid,
            "merge",
            "build",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        runflow_stage_error(
            "merge",
            sid=sid,
            error_type=exc.__class__.__name__,
            message=str(exc),
            traceback_tail=format_exception_tail(exc),
            hint=compose_hint("merge build", exc),
            summary={"phase": "build"},
        )
        raise

    merge_paths = ensure_merge_paths(runs_root, sid, create=True)
    index_path = merge_paths.index_file
    if not index_path.exists():
        legacy_dir = probe_legacy_ai_packs(runs_root, sid)
        if legacy_dir is not None:
            legacy_index = legacy_dir / "index.json"
            if legacy_index.exists():
                index_path = legacy_index
    try:
        index_entries = _load_ai_index(index_path)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "AUTO_AI_BUILD_INVALID_INDEX sid=%s path=%s", sid, index_path, exc_info=True
        )
        _cleanup_lock(payload, reason="build_invalid_index")
        runflow_step(
            sid,
            "merge",
            "build",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        runflow_stage_error(
            "merge",
            sid=sid,
            error_type=exc.__class__.__name__,
            message=str(exc),
            traceback_tail=format_exception_tail(exc),
            hint=compose_hint("merge build", exc),
            summary={"phase": "build"},
        )
        raise

    payload["ai_index"] = index_entries

    touched: set[int] = set(_normalize_indices(payload.get("touched_accounts", [])))
    touched.update(_indices_from_index(index_entries))
    payload["touched_accounts"] = sorted(touched)

    logger.info("AI_PACKS_INDEX sid=%s path=%s count=%d", sid, index_path, len(index_entries))
    logger.info("AI_BUILD_END sid=%s packs=%d", sid, len(index_entries))

    runflow_step(
        sid,
        "merge",
        "build",
        metrics={
            "packs": len(index_entries),
            "touched_accounts": len(touched),
        },
    )
    return payload


def _merge_send_stage(payload: dict[str, object]) -> dict[str, object]:
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_SEND_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])

    # Idempotency check: skip if merge already sent/applied
    try:
        from backend.runflow.decider import get_runflow_snapshot
        snapshot = get_runflow_snapshot(sid, runs_root=runs_root)
        merge_stage = snapshot.get("stages", {}).get("merge", {})
        
        merge_applied = merge_stage.get("merge_ai_applied", False)
        result_files = merge_stage.get("result_files", 0)
        
        if merge_applied and result_files > 0:
            logger.info(
                "MERGE_SEND_IDEMPOTENT_SKIP sid=%s reason=already_sent applied=%s results=%s",
                sid, merge_applied, result_files
            )
            payload["merge_sent"] = True
            payload["merge_skipped"] = True
            return payload
    except Exception:
        logger.debug("MERGE_SEND_IDEMPOTENT_CHECK_FAILED sid=%s", sid, exc_info=True)

    logger.info("AI_SEND_START sid=%s", sid)

    index_entries = payload.get("ai_index")
    if not index_entries:
        reason = str(payload.get("skip_reason") or "no_packs")
        logger.info("AI_SEND_SKIP sid=%s reason=%s", sid, reason)
        runflow_step(
            sid,
            "merge",
            "send",
            status="skipped",
            out={"reason": reason},
        )
        return payload

    try:
        _send_ai_packs(sid, runs_root=runs_root)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_SEND_FAILED sid=%s", sid, exc_info=True)
        _cleanup_lock(payload, reason="send_failed")
        runflow_step(
            sid,
            "merge",
            "send",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        runflow_stage_error(
            "merge",
            sid=sid,
            error_type=exc.__class__.__name__,
            message=str(exc),
            traceback_tail=format_exception_tail(exc),
            hint=compose_hint("merge send", exc),
            summary={"phase": "send"},
        )
        raise

    logger.info("AI_SEND_END sid=%s", sid)
    runflow_step(
        sid,
        "merge",
        "send",
        metrics={"packs": len(index_entries)},
    )
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_validation_requirements_step(
    self, prev: Mapping[str, object] | None
) -> dict[str, object]:
    """Populate validation requirements after AI adjudication results."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_VALIDATION_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)

    runs_root_value = payload.get("runs_root")
    runs_root_path = Path(str(runs_root_value)) if runs_root_value else None

    logger.info("AI_VALIDATION_REQUIREMENTS_START sid=%s", sid)

    # ── T0 INITIALIZATION: Validation Paths ───────────────────────────────
    # Initialize validation manifest paths BEFORE any pack building.
    # This ensures the manifest contains all validation directories.
    try:
        from backend.ai.manifest import ensure_validation_section
        ensure_validation_section(sid, runs_root=runs_root_path)
        logger.info("VALIDATION_PATHS_T0_INITIALIZED sid=%s", sid)
    except Exception:
        logger.error("VALIDATION_PATHS_T0_INIT_FAILED sid=%s", sid, exc_info=True)
    # ───────────────────────────────────────────────────────────────────────

    detection_block: dict[str, object] | None = None
    try:
        detection_block = detect_and_persist_date_convention(sid, runs_root=runs_root_path)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AI_DATE_CONVENTION_FAILED sid=%s", sid, exc_info=True)

    try:
        stats = run_validation_requirements_for_all_accounts(
            sid, runs_root=runs_root_path
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "AI_VALIDATION_REQUIREMENTS_FAILED sid=%s", sid, exc_info=True
        )
        _cleanup_lock(payload, reason="validation_requirements_failed")
        runflow_step(
            sid,
            "validation",
            "requirements",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        raise

    payload["validation_requirements"] = stats
    if isinstance(detection_block, dict):
        payload["date_convention"] = dict(detection_block)

    logger.info(
        "AI_VALIDATION_REQUIREMENTS_END sid=%s processed=%d findings=%d",
        sid,
        stats.get("processed_accounts", 0),
        stats.get("findings", 0),
    )

    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def validation_merge_ai_results_step(
    self, prev: Mapping[str, object] | None
) -> dict[str, object]:
    """Merge validation AI decisions into summaries after results are written."""

    payload = _ensure_payload(prev)
    if _orchestrator_mode_enabled():
        logger.info("VALIDATION_ORCHESTRATOR_MODE_SKIP validation_merge sid=%s", payload.get("sid"))
        return payload
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_VALIDATION_MERGE_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)

    runs_root_value = payload.get("runs_root")
    if not runs_root_value:
        logger.info(
            "AUTO_AI_VALIDATION_MERGE_SKIP sid=%s reason=no_runs_root", sid
        )
        return payload

    runs_root_path = Path(str(runs_root_value))
    # Guard: consult runflow terminal flag AND manifest 'sent' to decide whether to skip
    try:
        from backend.runflow.decider import get_runflow_snapshot  # deferred import
        snapshot = get_runflow_snapshot(sid, runs_root=runs_root_path)
    except Exception:
        snapshot = {}
    stages_snapshot = snapshot.get("stages") if isinstance(snapshot, Mapping) else None
    validation_stage = stages_snapshot.get("validation") if isinstance(stages_snapshot, Mapping) else None
    def _any_true(container: Mapping[str, object] | None, key: str) -> bool:
        if not isinstance(container, Mapping):
            return False
        val = container.get(key)
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return bool(val)
        return False
    terminal_flag = (
        _any_true(validation_stage, "validation_send_terminal")
        or _any_true(validation_stage.get("metrics") if isinstance(validation_stage, Mapping) else None, "validation_send_terminal")
        or _any_true(validation_stage.get("summary") if isinstance(validation_stage, Mapping) else None, "validation_send_terminal")
    )
    # Read manifest validation.sent to distinguish real send attempts from preemptive terminal marking
    manifest_sent = False
    try:
        manifest = RunManifest.for_sid(sid, allow_create=True, runs_root=runs_root_path)
        validation_status = manifest.get_ai_stage_status("validation")
        if isinstance(validation_status, Mapping):
            manifest_sent = bool(validation_status.get("sent"))
    except Exception:
        manifest_sent = False

    if terminal_flag and manifest_sent:
        logger.info("VALIDATION_MERGE_TERMINAL sid=%s reason=terminal_and_manifest_sent", sid)
        return payload
    if terminal_flag and not manifest_sent:
        logger.info("VALIDATION_MERGE_TERMINAL_IGNORED sid=%s reason=manifest_not_sent", sid)
        # proceed to attempt an inline send below
    # Ensure the expected results directory exists (create=True)
    results_dir = validation_results_dir(sid, runs_root=runs_root_path, create=True)
    max_wait, poll_interval = _validation_merge_wait_settings()

    logger.info(
        "VALIDATION_AI_MERGE_STEP_ENTER sid=%s results_dir=%s poll_interval=%ds max_wait=%ds",
        sid,
        results_dir,
        poll_interval,
        max_wait,
    )

    # Inline fallback: if no validation results exist yet, run the legacy sender synchronously
    try:
        existing_results = list(results_dir.glob("*.json*"))
    except Exception:
        existing_results = []
    if not existing_results:
        logger.warning(
            "VALIDATION_RESULTS_MISSING_BEFORE_SEND sid=%s dir=%s – invoking run_validation_send_for_sid inline",
            sid,
            results_dir,
        )
        try:
            stats = run_validation_send_for_sid(sid, runs_root_path)
            logger.info(
                "VALIDATION_SEND_INLINE_DONE sid=%s result_files=%s accounts_total=%s stats=%s",
                sid,
                stats.get("result_files") if isinstance(stats, Mapping) else None,
                stats.get("accounts_total") if isinstance(stats, Mapping) else None,
                stats,
            )
        except Exception:
            logger.exception("VALIDATION_SEND_INLINE_FAILED sid=%s", sid)

        # Re-check after inline send
        try:
            existing_results = list(results_dir.glob("*.json*"))
        except Exception:
            existing_results = []
        logger.info(
            "VALIDATION_RESULTS_AFTER_INLINE_SEND sid=%s dir=%s result_files=%d",
            sid,
            results_dir,
            len(existing_results),
        )

    # Strict pipeline: during results-wait phase, do not perform promotions or errors.
    # Only poll until deadline; if incomplete, log and return early with no side-effects.
    progress = _await_validation_results(sid, runs_root_path)
    results_required = progress.total > 0

    if results_required and (progress.failed > 0 or progress.completed < progress.total):
        # Strict behavior: do nothing except minimal log and keep wait lock
        missing = max(progress.missing, 0)
        logger.info(
            "VALIDATION_PARTIAL sid=%s expected=%d completed=%d missing=%d",
            sid,
            progress.total,
            progress.completed,
            missing,
        )
        return payload

    merge_stats: Mapping[str, Any] | None = None

    try:
        from backend.pipeline.validation_merge_helpers import (
            apply_validation_merge_and_update_state,
        )
        
        merge_stats = apply_validation_merge_and_update_state(
            sid,
            runs_root=runs_root_path,
            source="auto_ai_tasks",
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "AUTO_AI_VALIDATION_MERGE_FAILED sid=%s", sid, exc_info=True
        )
        _cleanup_lock(payload, reason="validation_merge_failed")
        runflow_step(
            sid,
            "validation",
            "merge_results",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        raise
    else:
        if isinstance(merge_stats, Mapping):
            logger.info(
                "VALIDATION_AI_APPLIED sid=%s accounts=%s accounts_updated=%d fields_total=%d fields_updated=%d",
                sid,
                merge_stats.get("accounts"),
                merge_stats.get("accounts_updated", 0),
                merge_stats.get("fields_total", 0),
                merge_stats.get("fields_updated", 0),
            )
            payload["validation_ai_merge"] = dict(merge_stats)
            # Mark terminal success to prevent further retries
            try:
                record_stage_force(
                    sid,
                    {"stages": {"validation": {"validation_send_terminal": True, "sent": True}}},
                    runs_root=runs_root_path,
                    last_writer="validation_merge_results_success",
                    refresh_barriers=True,
                )
            except Exception:
                logger.debug("VALIDATION_TERMINAL_SUCCESS_FLAG_WRITE_FAILED sid=%s", sid, exc_info=True)

    # Log final merge results
    logger.info(
        "VALIDATION_MERGE_DONE sid=%s results_total=%s missing_results=%s",
        sid,
        merge_stats.get("results_total") if merge_stats else 0,
        merge_stats.get("missing_results") if merge_stats else 0,
    )

    logger.info("AUTO_AI_VALIDATION_MERGE_DONE sid=%s", sid)
    # Finalization: mark validation success and latch readiness, then drop wait lock
    try:
        record_stage_force(
            sid,
            {"stages": {"validation": {
                "status": "success",
                "sent": True,
                "ready_latched": True,
                "ready_latched_at": _isoformat_timestamp(),
                "last_at": _isoformat_timestamp(),
                "validation_finalized": True,
            }}},
            runs_root=runs_root_path,
            last_writer="validation_finalize_success",
            refresh_barriers=True,
        )
    except Exception:
        logger.debug("VALIDATION_FINALIZE_STAGE_WRITE_FAILED sid=%s", sid, exc_info=True)

    # Clear wait lock after successful finalization
    try:
        _lock_clear(_wait_lock_path(runs_root_path, sid))
    except Exception:
        logger.debug("VALIDATION_WAIT_LOCK_CLEAR_FAILED sid=%s", sid, exc_info=True)

    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def strategy_planner_step(
    self, prev: Mapping[str, object] | None
) -> dict[str, object]:
    """Run the strategy planner after validation AI decisions have been merged."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_STRATEGY_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)

    runs_root_value = payload.get("runs_root")
    runs_root_path = Path(str(runs_root_value)) if runs_root_value else None

    stats_source = payload.get("validation_requirements")
    stats_input = dict(stats_source) if isinstance(stats_source, Mapping) else None

    try:
        manifest = _load_strategy_manifest_or_raise(sid, runs_root_path)
    except Exception as exc:
        logger.error(
            "AUTO_AI_STRATEGY_MANIFEST_UNAVAILABLE sid=%s runs_root=%s",
            sid,
            runs_root_path,
            exc_info=True,
        )
        runflow_step(
            sid,
            "validation",
            "planner",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        _cleanup_lock(payload, reason="strategy_manifest_unavailable")
        raise

    logger.info(
        "STRATEGY_TASK_MANIFEST_START sid=%s runs_root=%s", sid, runs_root_path
    )
    
    # Short-circuit if strategy already completed (check BEFORE marking started)
    strategy_status = manifest.get_ai_stage_status("strategy")
    strategy_state = strategy_status.get("state")
    if strategy_state == "success":
        logger.info(
            "STRATEGY_TASK_SHORT_CIRCUIT sid=%s reason=already_success state=%s",
            sid,
            strategy_state,
        )
        # Return cached stats from manifest
        payload["validation_requirements"] = {
            "sid": sid,
            "planner_stage_status": "success",
            "planner_plans_written": strategy_status.get("plans_written", 0),
            "planner_errors": strategy_status.get("planner_errors", 0),
            "planner_accounts_seen": strategy_status.get("accounts_seen", 0),
            "planner_accounts_with_openers": strategy_status.get("accounts_with_openers", 0),
            "planner_accounts_planned": strategy_status.get("planner_accounts_planned", []),
            "strategy_short_circuit": True,
        }
        return payload
    
    # Not already done, mark as started and proceed
    manifest.mark_strategy_started()
    manifest.save()

    try:
        stats = run_strategy_planner_for_all_accounts(
            sid,
            runs_root=runs_root_path,
            stats=stats_input,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_STRATEGY_FAILED sid=%s", sid, exc_info=True)
        _cleanup_lock(payload, reason="strategy_failed")
        runflow_step(
            sid,
            "validation",
            "planner",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        manifest.mark_strategy_completed({}, failed=True, state="error").save()
        raise

    payload["validation_requirements"] = stats

    logger.info(
        "AUTO_AI_STRATEGY_END sid=%s status=%s plans=%d planner_errors=%d",
        sid,
        stats.get("planner_stage_status"),
        stats.get("planner_plans_written", 0),
        stats.get("planner_errors", 0),
    )

    planner_stats_accounts = [
        str(acc_id)
        for acc_id in stats.get("planner_accounts_planned", []) or []
    ]
    accounts_dir_value = payload.get("accounts_dir")
    accounts_dir = Path(str(accounts_dir_value)) if accounts_dir_value else None
    filesystem_accounts: list[str] = []
    if accounts_dir is not None:
        filesystem_accounts = _discover_strategy_accounts(accounts_dir)

    planned_accounts = sorted({*planner_stats_accounts, *filesystem_accounts})
    runs_root_for_registration = (
        runs_root_path if runs_root_path is not None else manifest.path.parent.parent
    )
    logger.info(
        "STRATEGY_TASK_MANIFEST_ACCOUNTS sid=%s planned=%s", sid, planned_accounts
    )
    for account_id in planned_accounts:
        manifest.register_strategy_artifacts_for_account(
            account_id,
            runs_root=runs_root_for_registration,
        )

    stage_state = str(stats.get("planner_stage_status") or "")
    strategy_status_payload = {
        "plans_written": stats.get("planner_plans_written"),
        "planner_errors": stats.get("planner_errors"),
        "accounts_seen": stats.get("planner_accounts_seen"),
        "accounts_with_openers": stats.get("planner_accounts_with_openers"),
    }
    manifest.mark_strategy_completed(
        strategy_status_payload,
        failed=stage_state.lower() == "error",
        state=stage_state or None,
    ).save()
    logger.info("STRATEGY_TASK_MANIFEST_DONE sid=%s state=%s", sid, stage_state)

    validation_ok = bool(stats.get("ok", True))
    stage_status: StageStatus = "built" if validation_ok else "error"
    findings_count = int(stats.get("findings_count", stats.get("findings", 0)) or 0)
    empty_ok = findings_count == 0
    notes_value = stats.get("notes")

    packs_total = int(stats.get("processed_accounts", 0) or 0)
    accounts_eligible = packs_total
    total_accounts = int(stats.get("total_accounts", 0) or 0)
    packs_skipped = max(0, total_accounts - accounts_eligible)
    validation_metrics = {
        "packs_total": packs_total,
        "accounts_eligible": accounts_eligible,
        "packs_skipped": packs_skipped,
        "plans_written": stats.get("planner_plans_written", 0),
        "planner_errors": stats.get("planner_errors", 0),
    }

    record_stage(
        sid,
        "validation",
        status=stage_status,
        counts={"findings_count": findings_count},
        empty_ok=empty_ok,
        notes=notes_value,
        metrics=validation_metrics,
        runs_root=runs_root_path,
    )

    try:
        fe_result = generate_frontend_packs_for_run(
            sid, runs_root=runs_root_path
        )
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_FRONTEND_PACKS_FAILED sid=%s", sid, exc_info=True)
        record_stage(
            sid,
            "frontend",
            status="error",
            counts={"packs_count": 0},
            empty_ok=True,
            notes="generation_failed",
            runs_root=runs_root_path,
        )
        try:
            reconcile_umbrella_barriers(sid, runs_root=runs_root_path)
        except Exception:  # pragma: no cover - defensive logging
            logger.warning(
                "AUTO_AI_FRONTEND_BARRIERS_RECONCILE_FAILED sid=%s", sid, exc_info=True
            )
    else:
        if fe_result.get("autorun_disabled"):
            logger.info(
                "AUTO_AI_FRONTEND_AUTORUN_DISABLED sid=%s status=%s",
                sid,
                fe_result.get("status"),
            )
        else:
            packs_count = int(fe_result.get("packs_count", 0) or 0)
            frontend_status_value = str(fe_result.get("status") or "success")
            frontend_stage_status: StageStatus = (
                "error" if frontend_status_value == "error" else "published"
            )
            notes_override = (
                frontend_status_value
                if frontend_status_value not in {"", "success"}
                else None
            )
            empty_ok_frontend = bool(fe_result.get("empty_ok", packs_count == 0))

            record_stage(
                sid,
                "frontend",
                status=frontend_stage_status,
                counts={"packs_count": packs_count},
                empty_ok=empty_ok_frontend,
                notes=notes_override,
                runs_root=runs_root_path,
            )

            try:
                reconcile_umbrella_barriers(sid, runs_root=runs_root_path)
            except Exception:  # pragma: no cover - defensive logging
                logger.warning(
                    "AUTO_AI_FRONTEND_BARRIERS_RECONCILE_FAILED sid=%s", sid, exc_info=True
                )

            if frontend_stage_status == "success":
                update_manifest_frontend(
                    sid,
                    packs_dir=fe_result.get("packs_dir"),
                    packs_count=packs_count,
                    built=bool(fe_result.get("built", False)),
                    last_built_at=fe_result.get("last_built_at"),
                    runs_root=runs_root_path,
                )

    decision = decide_next(sid, runs_root=runs_root_path)
    next_action = decision.get("next")

    final_next = decision.get("next") if next_action is None else next_action
    if final_next == "await_input":
        update_manifest_state(
            sid,
            "AWAITING_CUSTOMER_INPUT",
            runs_root=runs_root_path,
        )
    elif final_next == "complete_no_action":
        update_manifest_state(
            sid,
            "COMPLETE_NO_ACTION",
            runs_root=runs_root_path,
        )
    elif final_next == "stop_error":
        update_manifest_state(
            sid,
            "ERROR",
            runs_root=runs_root_path,
        )

    return payload


def _merge_compact_stage(payload: dict[str, object]) -> dict[str, object]:
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_COMPACT_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)

    accounts_dir_value = payload.get("accounts_dir")
    accounts_dir = Path(str(accounts_dir_value)) if accounts_dir_value else None
    runs_root_value = payload.get("runs_root")
    runs_root = Path(str(runs_root_value)) if runs_root_value else None

    indices_set: set[int] = set(_normalize_indices(payload.get("touched_accounts", [])))

    result_pairs: list[tuple[int, int, dict[str, object]]] = []
    merge_paths = None
    if runs_root is not None:
        try:
            merge_paths = ensure_merge_paths(runs_root, sid, create=True)
        except Exception:  # pragma: no cover - defensive logging
            logger.error(
                "AUTO_AI_RESULTS_PATH_FAILED sid=%s runs_root=%s", sid, runs_root, exc_info=True
            )
        else:
            result_pairs = _load_ai_results(merge_paths.results_dir)

    if result_pairs and accounts_dir is None:
        logger.warning("AUTO_AI_RESULTS_NO_ACCOUNTS_DIR sid=%s", sid)

    if result_pairs and accounts_dir is not None:
        for a_idx, b_idx, result_payload in result_pairs:
            _apply_ai_result_to_accounts(accounts_dir, a_idx, b_idx, result_payload)
            indices_set.add(int(a_idx))
            indices_set.add(int(b_idx))
        logger.info("AI_RESULTS_APPLIED sid=%s count=%d", sid, len(result_pairs))

    indices = sorted(indices_set)
    payload["touched_accounts"] = indices

    logger.info("AI_COMPACT_START sid=%s accounts=%d", sid, len(indices))

    if accounts_dir and accounts_dir.exists() and indices:
        try:
            _compact_accounts(accounts_dir, indices)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "AUTO_AI_COMPACT_FAILED sid=%s dir=%s", sid, accounts_dir, exc_info=True
            )
            _cleanup_lock(payload, reason="compact_failed")
            runflow_step(
                sid,
                "merge",
                "compact",
                status="error",
                metrics={"accounts": len(indices)},
                out={"error": exc.__class__.__name__, "msg": str(exc)},
            )
            runflow_stage_error(
                "merge",
                sid=sid,
                error_type=exc.__class__.__name__,
                message=str(exc),
                traceback_tail=format_exception_tail(exc),
                hint=compose_hint("merge compact", exc),
                summary={"phase": "compact"},
            )
            raise

    logger.info("AI_COMPACT_END sid=%s", sid)

    packs_count = len(payload.get("ai_index", []) or [])
    pairs_count = len(indices)
    payload["packs"] = packs_count
    payload["created_packs"] = packs_count
    payload["pairs"] = pairs_count

    logger.info("MERGE_STAGE_DONE sid=%s", sid)
    runflow_step(
        sid,
        "merge",
        "compact",
        metrics={"packs": packs_count, "pairs": pairs_count},
    )

    try:
        runflow_barriers_refresh(sid)
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "AUTO_AI_MERGE_BARRIERS_REFRESH_FAILED sid=%s", sid, exc_info=True
        )

    scored_pairs_value = 0
    if merge_paths is not None:
        try:
            pairs_index_payload = json.loads(
                merge_paths.index_file.read_text(encoding="utf-8")
            )
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            scored_pairs_value = 0
        else:
            totals = pairs_index_payload.get("totals") if isinstance(pairs_index_payload, Mapping) else None
            if isinstance(totals, Mapping):
                try:
                    scored_pairs_value = int(totals.get("scored_pairs", 0) or 0)
                except (TypeError, ValueError):
                    scored_pairs_value = 0

    payload["merge_scored_pairs"] = scored_pairs_value
    return payload


def _finalize_stage(payload: dict[str, object]) -> dict[str, object]:
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_FINALIZE_SKIP payload=%s", payload)
        return payload

    last_ok_value = payload.get("last_ok_path")
    if last_ok_value:
        last_ok_path = Path(str(last_ok_value))
        last_ok_payload = {
            "sid": sid,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "packs": payload.get("packs", 0),
            "pairs": payload.get("pairs", 0),
        }
        try:
            last_ok_path.write_text(
                json.dumps(last_ok_payload, ensure_ascii=False), encoding="utf-8"
            )
            logger.info("AUTO_AI_LAST_OK sid=%s path=%s", sid, last_ok_path)
        except Exception:  # pragma: no cover - defensive logging
            logger.warning(
                "AUTO_AI_LAST_OK_WRITE_FAILED sid=%s path=%s",
                sid,
                last_ok_path,
                exc_info=True,
            )

    runs_root_path: Path | None = None
    runs_root_value = payload.get("runs_root")
    if runs_root_value:
        try:
            runs_root_path = Path(str(runs_root_value))
        except Exception:  # pragma: no cover - defensive logging
            logger.debug(
                "AUTO_AI_LOG_ROOT_INVALID sid=%s runs_root=%r",
                sid,
                runs_root_value,
                exc_info=True,
            )
        else:
            skip_reason = payload.get("skip_reason")
            reason_text = str(skip_reason) if isinstance(skip_reason, str) else None
            _append_run_log_entry(
                runs_root=runs_root_path,
                sid=sid,
                packs=int(payload.get("packs", 0)),
                pairs=int(payload.get("pairs", 0)),
                reason=reason_text,
            )

    removed = _cleanup_lock(payload, reason="chain_complete")
    logger.info(
        "AUTO_AI_CHAIN_END sid=%s lock_removed=%s packs=%s pairs=%s polarity=%s",
        sid,
        1 if removed else 0,
        payload.get("packs"),
        payload.get("pairs"),
        payload.get("polarity_processed"),
    )

    skip_reason = payload.get("skip_reason")
    if isinstance(skip_reason, str):
        skip_reason = skip_reason.strip()
    else:
        skip_reason = None

    snapshot = finalize_merge_stage(
        sid,
        runs_root=runs_root_path,
        notes=skip_reason if skip_reason else None,
    )

    counts = snapshot.get("counts", {})
    metrics = snapshot.get("metrics", {})

    if "packs_created" in counts:
        payload["packs"] = counts["packs_created"]
        payload["created_packs"] = counts["packs_created"]
    if "pairs_scored" in counts:
        payload["merge_scored_pairs"] = counts["pairs_scored"]
    elif "scored_pairs" in metrics:
        payload["merge_scored_pairs"] = metrics["scored_pairs"]
    if "result_files" in counts:
        payload["merge_result_files"] = counts["result_files"]
    payload["merge_empty_ok"] = bool(snapshot.get("empty_ok"))

    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def merge_build_packs(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    return _merge_build_stage(payload)


ai_build_packs_step = merge_build_packs


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def merge_send(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    return _merge_send_stage(payload)


ai_send_packs_step = merge_send


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def merge_compact(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    return _merge_compact_stage(payload)


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def validation_build_packs(
    self, prev: Mapping[str, object] | None
) -> dict[str, object]:
    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("VALIDATION_BUILD_SKIP payload=%s", payload)
        return payload

    # ── CHAIN EXECUTION GUARD ──────────────────────────────────────────────
    # This task MUST only run as part of the Auto-AI chain.
    # If called standalone (legacy path), log warning and defer to chain.
    if not payload.get("_chain_context"):
        logger.warning(
            "VALIDATION_BUILD_STANDALONE_CALL_BLOCKED sid=%s - must run via chain",
            sid
        )
        # Return minimal payload to prevent further execution
        return {"sid": sid, "validation_deferred": True}
    # ───────────────────────────────────────────────────────────────────────

    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])

    ensure_validation_section(sid, runs_root=runs_root)

    # Strict: establish delivery lock at the start of pack build (disabled in orchestrator mode)
    if _strict_validation_pipeline_enabled() and not _orchestrator_mode_enabled():
        _lock_set(_delivery_lock_path(runs_root, sid), "build:start")
    
    # Runflow-first short-circuit: if runflow says validation succeeded, skip work
    try:
        from backend.pipeline.runs import get_stage_status
        status = get_stage_status(sid, stage="validation", runs_root=runs_root)
        if isinstance(status, str) and status.strip().lower() == "success":
            logger.info(
                "VALIDATION_BUILD_SHORT_CIRCUIT sid=%s reason=runflow_success status=%s",
                sid,
                status,
            )
            payload["validation_packs"] = 0
            payload["validation_short_circuit"] = True
            return payload
    except Exception:  # pragma: no cover - defensive, don't fail on check
        logger.debug("VALIDATION_RUNFLOW_SHORT_CIRCUIT_CHECK_FAILED sid=%s", sid, exc_info=True)
    
    # Short-circuit if validation already completed
    try:
        from backend.pipeline.runs import RunManifest
        manifest = RunManifest.for_sid(sid, runs_root=runs_root, allow_create=False)
        validation_status = manifest.get_ai_stage_status("validation")
        state = validation_status.get("state")
        if state == "success":
            logger.info(
                "VALIDATION_BUILD_SHORT_CIRCUIT sid=%s reason=already_success state=%s",
                sid,
                state,
            )
            payload["validation_packs"] = 0
            payload["validation_short_circuit"] = True
            return payload
    except Exception:  # pragma: no cover - defensive, don't fail on check
        logger.debug("VALIDATION_SHORT_CIRCUIT_CHECK_FAILED sid=%s", sid, exc_info=True)
    
    # Additional idempotency: check if packs already built (not just validation complete)
    try:
        from backend.core.ai.paths import validation_index_path
        index_path = validation_index_path(sid, runs_root=runs_root, create=False)
        
        if index_path.exists():
            # Index exists — check if packs are already built
            import json
            try:
                index_data = json.loads(index_path.read_text(encoding="utf-8"))
                packs_list = index_data.get("packs", [])
                if len(packs_list) > 0:
                    logger.info(
                        "VALIDATION_BUILD_IDEMPOTENT_SKIP sid=%s reason=packs_already_built count=%d",
                        sid, len(packs_list)
                    )
                    payload["validation_packs"] = len(packs_list)
                    payload["validation_short_circuit"] = True
                    return payload
            except (json.JSONDecodeError, OSError):
                pass  # Continue if index unreadable
    except Exception:
        logger.debug("VALIDATION_BUILD_IDEMPOTENT_PACKS_CHECK_FAILED sid=%s", sid, exc_info=True)
    
    logger.info("VALIDATION_STAGE_STARTED sid=%s", sid)

    try:
        results = build_validation_packs_for_run(sid, runs_root=runs_root)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error("VALIDATION_BUILD_FAILED sid=%s", sid, exc_info=True)
        runflow_step(
            sid,
            "validation",
            "build",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        runflow_stage_error(
            "validation",
            sid=sid,
            error_type=exc.__class__.__name__,
            message=str(exc),
            traceback_tail=format_exception_tail(exc),
            hint=compose_hint("validation build", exc),
            summary={"phase": "build"},
        )
        raise

    packs_written = sum(len(entries or []) for entries in results.values())
    payload["validation_packs"] = packs_written
    logger.info("VALIDATION_BUILD_DONE sid=%s packs=%d", sid, packs_written)
    if not _strict_validation_pipeline_enabled() and not _orchestrator_mode_enabled():
        runflow_step(
            sid,
            "validation",
            "build",
            metrics={"packs": packs_written},
        )
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def validation_send(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    """Send validation packs to AI and collect results.
    
    In orchestrator mode, uses the new validation_sender_v2 (clean sender).
    In legacy mode, uses the old validation sender path.
    """
    payload = _ensure_payload(prev)
    
    sid = str(payload.get("sid") or "")
    
    if _orchestrator_mode_enabled():
        logger.info("VALIDATION_ORCHESTRATOR_SEND_V2 sid=%s", sid)
        
        if not sid:
            logger.warning("VALIDATION_SEND_SKIP_NO_SID payload=%s", payload)
            return payload
        
        _populate_common_paths(payload)
        runs_root = Path(payload["runs_root"])
        
        # Idempotency check: skip if validation already sent
        try:
            from backend.runflow.decider import get_runflow_snapshot
            snapshot = get_runflow_snapshot(sid, runs_root=runs_root)
            validation_stage = snapshot.get("stages", {}).get("validation", {})
            
            status = validation_stage.get("status")
            results = validation_stage.get("results", {})
            results_total = results.get("results_total", 0)
            missing_results = results.get("missing_results", 1)
            
            if status == "success" and results_total > 0 and missing_results == 0:
                logger.info(
                    "VALIDATION_SEND_IDEMPOTENT_SKIP sid=%s reason=already_sent status=%s results=%s missing=%s",
                    sid, status, results_total, missing_results
                )
                payload["validation_sent"] = True
                payload["validation_skipped"] = True
                return payload
        except Exception:
            logger.debug("VALIDATION_SEND_IDEMPOTENT_CHECK_FAILED sid=%s", sid, exc_info=True)
        
        # Use new clean sender (validation_sender_v2)
        try:
            from backend.ai.validation_sender_v2 import run_validation_send_for_sid_v2
            stats = run_validation_send_for_sid_v2(sid, runs_root)
            logger.info(
                "VALIDATION_ORCHESTRATOR_SEND_V2_DONE sid=%s expected=%s sent=%s written=%s failed=%s",
                sid,
                stats.get("expected"),
                stats.get("sent"),
                stats.get("written"),
                stats.get("failed"),
            )
            payload["validation_sent"] = True
            payload["validation_v2_stats"] = stats
        except Exception as exc:
            logger.exception("VALIDATION_ORCHESTRATOR_SEND_V2_FAILED sid=%s", sid)
            payload["validation_sent"] = False
            raise
        
        return payload
    
    # Legacy mode below (non-orchestrator)
    logger.info("VALIDATION_SEND_ENTRY sid=%s payload_keys=%s", 
                sid, list((prev or {}).keys()))
    
    logger.info("VALIDATION_SEND_ENTRY sid=%s", sid)
    if not sid:
        logger.warning("VALIDATION_SEND_SKIP_NO_SID payload=%s", payload)
        return payload

    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])
    
    # Get the validation index path
    index_path = validation_index_path(sid, runs_root=runs_root, create=True)
    
    # Check if index exists before attempting to send
    if not index_path.exists():
        # Inspect runflow to determine if packs exist; if yes, treat as hard error
        packs_hint = 0
        try:
            from backend.runflow.decider import get_runflow_snapshot
            snapshot = get_runflow_snapshot(sid, runs_root=runs_root)
        except Exception:
            snapshot = {}
        stages_snapshot = snapshot.get("stages") if isinstance(snapshot, Mapping) else None
        validation_stage = stages_snapshot.get("validation") if isinstance(stages_snapshot, Mapping) else None
        def _extract_packs(container: Mapping[str, object] | None) -> int:
            if not isinstance(container, Mapping):
                return 0
            for key in ("packs", "packs_total", "expected_packs"):
                val = container.get(key)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    if int(val) > 0:
                        return int(val)
            return 0
        packs_hint = max(
            _extract_packs(validation_stage),
            _extract_packs(validation_stage.get("metrics") if isinstance(validation_stage, Mapping) else None),
            _extract_packs(validation_stage.get("summary") if isinstance(validation_stage, Mapping) else None),
            _extract_packs((validation_stage.get("summary") or {}).get("metrics") if isinstance(validation_stage, Mapping) else None),
        )
        if packs_hint > 0:
            logger.warning(
                "VALIDATION_SEND_INDEX_MISSING_BUT_PACKS_EXIST sid=%s index=%s packs=%s",
                sid,
                index_path,
                packs_hint,
            )
            payload["validation_sent"] = False
            # Do not raise to keep chain alive; merge step will inline-send
            return payload
        logger.warning("VALIDATION_SEND_SKIP_NO_INDEX sid=%s index=%s", sid, index_path)
        payload["validation_sent"] = False
        return payload
    
    logger.info("VALIDATION_SEND_START sid=%s index=%s", sid, index_path)
    
    # Use the legacy send behavior via helper to ensure results are written
    try:
        stats = run_validation_send_for_sid(sid, runs_root)
    except Exception as exc:
        logger.exception("VALIDATION_SEND_FAILED sid=%s index=%s", sid, index_path)
        payload["validation_sent"] = False
        raise
    
    # Log completion with stats from send_validation_packs
    discovered = None
    try:
        if isinstance(stats, Mapping):
            discovered = stats.get("results_discovered") or stats.get("result_files")
    except Exception:
        discovered = None
    logger.info(
        "VALIDATION_SEND_DONE sid=%s result_files=%s stats=%s",
        sid,
        discovered,
        stats,
    )
    
    payload["validation_sent"] = True

    # Strict: move from delivery phase to results-wait phase
    if _strict_validation_pipeline_enabled():
        _lock_clear(_delivery_lock_path(runs_root, sid))
        _lock_set(_wait_lock_path(runs_root, sid), "wait:start")
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def validation_compact(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("VALIDATION_COMPACT_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])
    index_path = validation_index_path(sid, runs_root=runs_root, create=True)

    if not index_path.exists():
        logger.info(
            "VALIDATION_COMPACT_SKIP sid=%s reason=index_missing path=%s", sid, index_path
        )
        runflow_step(
            sid,
            "validation",
            "compact",
            status="skipped",
            out={"reason": "index_missing"},
        )
        return payload

    # Idempotency check: skip if already compacted
    try:
        import json
        index_data = json.loads(index_path.read_text(encoding="utf-8"))
        compacted_flag = index_data.get("compacted", False)
        canonical_layout = index_data.get("canonical_layout", False)
        
        if compacted_flag or canonical_layout:
            logger.info(
                "VALIDATION_COMPACT_IDEMPOTENT_SKIP sid=%s reason=already_compacted flag=%s canonical=%s",
                sid, compacted_flag, canonical_layout
            )
            payload["validation_compacted"] = True
            payload["validation_compact_skipped"] = True
            return payload
    except (json.JSONDecodeError, OSError, KeyError):
        pass  # Continue if check fails

    # Strict: avoid compaction while delivery/wait is in-flight
    if _strict_validation_pipeline_enabled():
        if _lock_exists(_delivery_lock_path(runs_root, sid)) or _lock_exists(_wait_lock_path(runs_root, sid)):
            logger.info("VALIDATION_COMPACT_SKIP sid=%s reason=strict_pipeline_inflight", sid)
            return payload

    logger.info("VALIDATION_COMPACT_START sid=%s", sid)

    try:
        rewrite_index_to_canonical_layout(index_path, runs_root=runs_root)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.error(
            "VALIDATION_COMPACT_FAILED sid=%s path=%s", sid, index_path, exc_info=True
        )
        runflow_step(
            sid,
            "validation",
            "compact",
            status="error",
            out={"error": exc.__class__.__name__, "msg": str(exc)},
        )
        runflow_stage_error(
            "validation",
            sid=sid,
            error_type=exc.__class__.__name__,
            message=str(exc),
            traceback_tail=format_exception_tail(exc),
            hint=compose_hint("validation compact", exc),
            summary={"phase": "compact"},
        )
        raise

    payload["validation_compacted"] = True
    logger.info("VALIDATION_COMPACT_DONE sid=%s", sid)
    runflow_step(
        sid,
        "validation",
        "compact",
        metrics={"compacted": True},
    )
    if _maybe_autobuild_review(
        sid,
        already_triggered=bool(payload.get("review_autobuild_queued")),
        source="validation_compact",
    ):
        payload["review_autobuild_queued"] = True
    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def pipeline_finalize(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    return _finalize_stage(payload)


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_compact_tags_step(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    """Backwards-compatible task that compacts merge data then finalizes the run."""

    payload = _ensure_payload(prev)
    payload = _merge_compact_stage(payload)
    return _finalize_stage(payload)


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_polarity_check_step(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    """Run polarity classification after AI adjudication outputs are compacted."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_POLARITY_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)

    accounts_dir_value = payload.get("accounts_dir")
    accounts_dir = Path(str(accounts_dir_value)) if accounts_dir_value else None
    if accounts_dir is None:
        logger.info("AUTO_AI_POLARITY_SKIP sid=%s reason=no_accounts_dir", sid)
        return payload

    indices = sorted(_normalize_indices(payload.get("touched_accounts", [])))
    if not indices:
        logger.info("AUTO_AI_POLARITY_SKIP sid=%s reason=no_indices", sid)
        return payload

    logger.info("AI_POLARITY_START sid=%s accounts=%d", sid, len(indices))

    try:
        result = polarity.apply_polarity_checks(accounts_dir, indices, sid=sid)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_POLARITY_FAILED sid=%s dir=%s", sid, accounts_dir, exc_info=True)
        _cleanup_lock(payload, reason="polarity_failed")
        raise

    payload["polarity_processed"] = result.processed_accounts
    payload["polarity_updated"] = result.updated_accounts
    if result.config_digest:
        payload["polarity_config_digest"] = result.config_digest

    logger.info(
        "AI_POLARITY_END sid=%s processed=%d updated=%d",
        sid,
        result.processed_accounts,
        len(result.updated_accounts),
    )

    return payload


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def ai_consistency_step(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    """Persist field consistency snapshots for accounts touched by the AI flow."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("AUTO_AI_CONSISTENCY_SKIP payload=%s", payload)
        return payload

    _populate_common_paths(payload)

    runs_root_value = payload.get("runs_root")
    runs_root_path = Path(str(runs_root_value)) if runs_root_value else None

    logger.info("AI_CONSISTENCY_START sid=%s", sid)

    try:
        stats = run_consistency_writeback_for_all_accounts(
            sid, runs_root=runs_root_path
        )
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AI_CONSISTENCY_FAILED sid=%s", sid, exc_info=True)
        _cleanup_lock(payload, reason="consistency_failed")
        raise

    payload["consistency"] = stats

    logger.info(
        "AI_CONSISTENCY_END sid=%s processed=%d fields=%d",
        sid,
        stats.get("processed_accounts", 0),
        stats.get("fields", 0),
    )

    return payload


def enqueue_auto_ai_chain(sid: str, runs_root: Path | str | None = None) -> str:
    """Queue the AI adjudication Celery chain and return the root task id."""

    runs_root_value = str(runs_root) if runs_root is not None else None

    logger.info("AUTO_AI_CHAIN_START sid=%s runs_root=%s", sid, runs_root_value)
    logger.info("VALIDATION_PIPELINE_ENTRY sid=%s path=%s mode=%s", sid, "AUTO_AI_CHAIN", "full")
    logger.info("STAGE_CHAIN_STARTED sid=%s", sid)

    # Mark this execution as coming from the chain
    # This prevents standalone task execution outside the chain
    workflow = chain(
        ai_score_step.s(sid, runs_root_value),
        merge_build_packs.s(),
        merge_send.s(),
        merge_compact.s(),
        run_date_convention_detector.s(),
        ai_validation_requirements_step.s(),
        validation_build_packs.s(),
        validation_send.s(),
        validation_compact.s(),
        validation_merge_ai_results_step.s(),
        strategy_planner_step.s(),
        ai_polarity_check_step.s(),
        ai_consistency_step.s(),
        pipeline_finalize.s(),
    )

    result = workflow.apply_async()
    task_id = str(result.id)

    logger.info(
        "AUTO_AI_CHAIN_ENQUEUED sid=%s task_id=%s runs_root=%s",
        sid,
        task_id,
        runs_root_value,
    )
    return task_id


# --- Strategy recovery helpers -------------------------------------------------

@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def strategy_seed_payload(self, sid: str, runs_root_value: str | None) -> dict[str, object]:
    """Seed a minimal payload for starting the strategy tail of the pipeline.

    This task is used by the recovery path to kick off strategy planning and
    subsequent stages when the main chain aborted earlier.
    """
    payload: dict[str, object] = {"sid": str(sid)}
    if runs_root_value:
        payload["runs_root"] = runs_root_value
    return payload


def _resolve_default_queue_name() -> str:
    """Return the default Celery queue name safely.

    Prefer the application-level helper when available, otherwise fall back
    to the environment variable or the conventional "celery" queue.
    """
    try:  # Deferred import to avoid hard dependency at module import time
        from backend.api.tasks import _default_queue_name as _dq  # type: ignore

        name = _dq()
        if isinstance(name, str) and name.strip():
            return name.strip()
    except Exception:  # pragma: no cover - defensive import fallback
        pass

    env_name = (os.getenv("CELERY_DEFAULT_QUEUE") or "").strip()
    return env_name or "celery"


def enqueue_strategy_recovery_chain(sid: str, runs_root: Path | str | None = None) -> str:
    """Queue the strategy tail (planner → polarity → consistency → finalize).

    Returns the Celery task id of the enqueued chain. Safe to call multiple
    times if callers guard with runflow/manifest checks to avoid duplicates.
    """
    runs_root_value = str(runs_root) if runs_root is not None else None

    logger.info(
        "STRATEGY_RECOVERY_CHAIN_START sid=%s runs_root=%s", sid, runs_root_value
    )

    workflow = chain(
        strategy_seed_payload.s(sid, runs_root_value),
        strategy_planner_step.s(),
        ai_polarity_check_step.s(),
        ai_consistency_step.s(),
        pipeline_finalize.s(),
    )

    # Route onto the same default queue used for the main pipeline.
    queue_name = _resolve_default_queue_name()
    logger.info(
        "STRATEGY_RECOVERY_CHAIN_QUEUING sid=%s queue=%s runs_root=%s",
        sid,
        queue_name,
        runs_root_value,
    )

    result = workflow.apply_async(queue=queue_name)
    task_id = str(result.id)
    logger.info(
        "STRATEGY_RECOVERY_CHAIN_ENQUEUED sid=%s task_id=%s runs_root=%s queue=%s",
        sid,
        task_id,
        runs_root_value,
        queue_name,
    )
    return task_id

