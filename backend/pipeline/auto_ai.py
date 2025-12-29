"""Automatic AI adjudication hooks for the case-build pipeline."""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone, date
from pathlib import Path
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

from celery import shared_task

from backend.ai.validation_builder import (
    _maybe_send_validation_packs,
    build_validation_packs_for_run,
)
from backend.core.ai.paths import ensure_merge_paths, probe_legacy_ai_packs
from backend.core.config import ENABLE_VALIDATION_REQUIREMENTS
from backend.core.io.json_io import update_json_in_place
from backend.core.logic.consistency import (
    compute_field_consistency,
    _AMOUNT_TOL_ABS,
    _AMOUNT_TOL_RATIO,
    _DATE_TOLERANCE_DAYS,
    _DATE_TOLERANT_FIELDS,
    _TOLERANT_AMOUNT_FIELDS,
)
from backend.core.logic.summary_compact import compact_merge_sections
from backend.core.logic.validation_requirements import (
    build_validation_requirements_for_account,
)
from backend.core.logic.tags.compact import (
    compact_account_tags,
    compact_tags_for_sid,
)
from backend.pipeline.runs import RUNS_ROOT, RunManifest, persist_manifest
from backend.core.runflow import runflow_step
from backend.core.runflow.io import (
    compose_hint,
    format_exception_tail,
    runflow_stage_end,
    runflow_stage_error,
    runflow_stage_start,
)
from backend.prevalidation import detect_and_persist_date_convention
from backend.pipeline.steps.validation_requirements_step import (
    run as validation_requirements_step,
)
from backend.strategy.config import PlannerEnv
from backend.strategy.runflow import record_strategy_stage
from backend.validation.build_packs import load_manifest_from_source
from backend.validation.pipeline import (
    PlannerHookResult,
    iterate_accounts,
    maybe_run_planner_for_account,
)
from scripts.build_ai_merge_packs import main as build_ai_merge_packs_main
from scripts.send_ai_merge_packs import main as send_ai_merge_packs_main
from backend.runflow.decider import get_runflow_snapshot, record_stage_force

logger = logging.getLogger(__name__)

AUTO_AI_PIPELINE_DIRNAME = Path("ai_packs") / "merge"
INFLIGHT_LOCK_FILENAME = "inflight.lock"
LAST_OK_FILENAME = "last_ok.json"
DEFAULT_INFLIGHT_TTL_SECONDS = 30 * 60


def _maybe_slice(iterable: Iterable[object]) -> Iterable[object]:
    """Return ``iterable`` unchanged to ensure all accounts are processed."""

    debug_first_n = os.getenv("DEBUG_FIRST_N", "").strip()
    if debug_first_n:
        logger.debug(
            "DEBUG_FIRST_N=%s ignored; processing all items", debug_first_n
        )
    return iterable


def _clone_without_raw(value: Any) -> Any:
    """Return a JSON-serializable clone of ``value`` without ``"raw"`` keys."""

    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if key_str == "raw":
                continue
            result[key_str] = _clone_without_raw(item)
        return result

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_clone_without_raw(item) for item in value]

    return value


def _count_tolerance_hits(details: Mapping[str, Any]) -> tuple[int, int]:
    date_hits = 0
    amount_hits = 0

    for field, info in details.items():
        if not isinstance(info, Mapping):
            continue

        normalized_map = info.get("normalized")
        if not isinstance(normalized_map, Mapping):
            continue

        values = [value for value in normalized_map.values() if value not in (None, "")]
        if not values:
            continue

        if field in _DATE_TOLERANT_FIELDS:
            parsed_dates: list[date] = []
            unique_values: set[str] = set()
            for candidate in values:
                if not isinstance(candidate, str):
                    continue
                try:
                    parsed = datetime.strptime(candidate, "%Y-%m-%d").date()
                except ValueError:
                    continue
                parsed_dates.append(parsed)
                unique_values.add(candidate)
            if len(unique_values) > 1 and parsed_dates:
                span = (max(parsed_dates) - min(parsed_dates)).days
                if span <= _DATE_TOLERANCE_DAYS:
                    date_hits += 1

        if field in _TOLERANT_AMOUNT_FIELDS:
            numeric_values: list[float] = []
            for candidate in values:
                if isinstance(candidate, (int, float)):
                    numeric_values.append(float(candidate))
                elif isinstance(candidate, str):
                    try:
                        numeric_values.append(float(candidate))
                    except ValueError:
                        continue
            if len(numeric_values) > 1:
                maximum = max(numeric_values)
                minimum = min(numeric_values)
                if maximum != minimum:
                    diff = abs(maximum - minimum)
                    scale = max(abs(maximum), abs(minimum))
                    tolerance = max(_AMOUNT_TOL_ABS, _AMOUNT_TOL_RATIO * scale)
                    if diff <= tolerance:
                        amount_hits += 1

    return date_hits, amount_hits


def _normalize_consistency_without_raw(
    consistency: Mapping[str, Any]
) -> Dict[str, Any]:
    """Normalize ``consistency`` for summary.json without embedding bureau raw values."""

    return {
        str(field): _clone_without_raw(details)
        for field, details in consistency.items()
    }


def packs_dir_for(sid: str, *, runs_root: Path | str | None = None) -> Path:
    """Return the canonical merge AI pipeline directory for ``sid``."""

    base = Path(runs_root) if runs_root is not None else RUNS_ROOT
    merge_paths = ensure_merge_paths(base, sid, create=False)
    return merge_paths.base


def _account_sort_key(path: Path) -> tuple[int, object]:
    name = path.name
    if name.isdigit():
        return (0, int(name))
    return (1, name)


def _count_validation_ai_packs(base_root: Path, sid: str) -> int:
    base_dir = base_root / sid / "ai_packs" / "validation" / "packs"
    if not base_dir.is_dir():
        return 0

    total = 0
    for entry in base_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(".tmp"):
            continue
        if entry.suffix.lower() != ".jsonl":
            continue
        total += 1
    return total


def _count_validation_ai_results(base_root: Path, sid: str) -> int:
    base_dir = base_root / sid / "ai_packs" / "validation" / "results"
    if not base_dir.is_dir():
        return 0

    seen: set[str] = set()
    for entry in base_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if name.endswith(".tmp"):
            continue
        if name.endswith(".error.json"):
            continue
        if name.endswith(".jsonl"):
            stem = name[: -len(".jsonl")]
        elif name.endswith(".json"):
            stem = name[: -len(".json")]
        else:
            continue
        seen.add(stem)

    return len(seen)


def _discover_strategy_accounts(accounts_root: Path) -> list[str]:
    if not accounts_root.is_dir():
        return []

    planned: list[str] = []
    for account_dir in accounts_root.iterdir():
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
                if (bureau_dir / "plan.json").exists():
                    has_plans = True
                    break
                if any(candidate.is_file() for candidate in bureau_dir.glob("plan_wd*.json")):
                    has_plans = True
                    break
                if has_plans:
                    break
        except OSError:  # pragma: no cover - defensive directory guard
            continue
        if has_plans:
            planned.append(account_dir.name)

    return sorted(planned)


def run_validation_requirements_for_all_accounts(
    sid: str, *, runs_root: Path | str | None = None
) -> dict[str, object]:
    """Run validation requirement extraction for each account of ``sid``."""

    base_root = Path(runs_root) if runs_root is not None else RUNS_ROOT
    accounts_root = base_root / sid / "cases" / "accounts"

    stats = {
        "sid": sid,
        "total_accounts": 0,
        "processed_accounts": 0,
        "findings": 0,
        "missing_bureaus": 0,
        "errors": 0,
    }

    runflow_stage_start("validation", sid=sid)

    def _build_summary(
        *,
        empty_ok: bool,
        reason: str | None = None,
        extra: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        summary: dict[str, object] = {
            "findings_count": int(stats.get("findings", 0) or 0),
            "ai_packs_built": _count_validation_ai_packs(base_root, sid),
            "ai_results_received": _count_validation_ai_results(base_root, sid),
            "empty_ok": bool(empty_ok),
        }
        if reason:
            summary["reason"] = reason
        if extra:
            summary.update({str(key): value for key, value in extra.items()})
        return summary

    try:
        if not accounts_root.exists():
            runflow_step(
                sid,
                "validation",
                "requirements",
                status="success",
                metrics={
                    "total_accounts": stats["total_accounts"],
                    "processed_accounts": stats["processed_accounts"],
                    "findings": stats["findings"],
                    "missing_bureaus": stats["missing_bureaus"],
                    "errors": stats["errors"],
                },
                out={"reason": "no_accounts"},
            )
            stats["findings_count"] = stats["findings"]
            stats["empty_ok"] = True
            summary = _build_summary(empty_ok=True, reason="no_accounts")
            runflow_stage_end(
                "validation",
                sid=sid,
                summary=summary,
                empty_ok=True,
            )
            return stats

        account_paths = [path for path in accounts_root.iterdir() if path.is_dir()]
        sorted_accounts = sorted(account_paths, key=_account_sort_key)
        stats["total_accounts"] = len(sorted_accounts)

        for position, account_path in enumerate(_maybe_slice(sorted_accounts), start=1):
            account_label = account_path.name
            logger.info("PROCESSING account_id=%s index=%d", account_label, position)
            try:
                result = build_validation_requirements_for_account(account_path)
            except Exception:  # pragma: no cover - defensive logging
                stats["errors"] += 1
                logger.exception(
                    "ERROR account_id=%s index=%d sid=%s path=%s",
                    account_label,
                    position,
                    sid,
                    account_path,
                )
                continue

            status = str(result.get("status") or "")
            if status == "no_bureaus_json":
                stats["missing_bureaus"] += 1
                continue
            if status != "ok":
                stats["errors"] += 1
                continue

            stats["processed_accounts"] += 1
            stats["findings"] += int(result.get("count") or 0)

        logger.info(
            "VALIDATION_REQUIREMENTS_SUMMARY sid=%s accounts=%d processed=%d findings=%d missing=%d errors=%d",
            sid,
            stats["total_accounts"],
            stats["processed_accounts"],
            stats["findings"],
            stats["missing_bureaus"],
            stats["errors"],
        )

        stats["ok"] = stats["errors"] == 0
        stats["findings_count"] = stats["findings"]
        if stats["errors"]:
            stats["notes"] = f"errors={stats['errors']}"
        else:
            stats["notes"] = None

        metrics = {
            "total_accounts": stats["total_accounts"],
            "processed_accounts": stats["processed_accounts"],
            "findings": stats["findings"],
            "missing_bureaus": stats["missing_bureaus"],
            "errors": stats["errors"],
        }
        out_payload = {}
        if stats.get("notes"):
            out_payload["notes"] = stats["notes"]
        status = "success" if stats["errors"] == 0 else "error"
        runflow_step(
            sid,
            "validation",
            "requirements",
            status=status,
            metrics=metrics,
            out=out_payload or None,
        )

        empty_ok = stats["total_accounts"] == 0
        stats["empty_ok"] = empty_ok
        summary = _build_summary(empty_ok=empty_ok)
        if stats.get("notes"):
            summary["notes"] = stats["notes"]
        stage_status = "success" if stats["errors"] == 0 else "error"
        runflow_stage_end(
            "validation",
            sid=sid,
            status=stage_status,
            summary=summary,
            empty_ok=empty_ok,
        )

        return stats
    except Exception as exc:
        summary = _build_summary(
            empty_ok=False,
            extra={"error": exc.__class__.__name__},
        )
        runflow_stage_error(
            "validation",
            sid=sid,
            error_type=exc.__class__.__name__,
            message=str(exc),
            traceback_tail=format_exception_tail(exc),
            hint=compose_hint("validation requirements", exc),
            summary=summary,
        )
        raise


def run_strategy_planner_for_all_accounts(
    sid: str,
    *,
    runs_root: Path | str | None = None,
    stats: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Run the strategy planner stage for ``sid`` and return updated stats."""

    base_root = Path(runs_root) if runs_root is not None else RUNS_ROOT
    manifest_path = base_root / sid / "manifest.json"
    logger.info(
        "STRATEGY_MANIFEST_LOCATE sid=%s runs_root=%s manifest=%s exists=%s",
        sid,
        str(base_root),
        str(manifest_path),
        manifest_path.exists(),
    )

    working_stats: dict[str, Any]
    if stats is None:
        working_stats = {"sid": sid}
    else:
        working_stats = dict(stats)

    snapshot: Mapping[str, Any] | None
    try:
        snapshot = get_runflow_snapshot(sid, runs_root=runs_root)
    except Exception:  # pragma: no cover - defensive snapshot guard
        snapshot = None

    def _stage_int(stage_info: Mapping[str, Any] | None, key: str, default: int = 0) -> int:
        if not isinstance(stage_info, Mapping):
            return default
        direct = stage_info.get(key)
        if isinstance(direct, (int, float)) and not isinstance(direct, bool):
            return int(direct)
        for container_key in ("metrics", "counts", "summary"):
            payload = stage_info.get(container_key)
            if isinstance(payload, Mapping):
                value = payload.get(key)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    return int(value)
                if container_key == "summary":
                    summary_metrics = payload.get("metrics")
                    if isinstance(summary_metrics, Mapping):
                        nested = summary_metrics.get(key)
                        if isinstance(nested, (int, float)) and not isinstance(nested, bool):
                            return int(nested)
        return default

    if isinstance(snapshot, Mapping):
        stages_payload = snapshot.get("stages")
        if isinstance(stages_payload, Mapping):
            strategy_stage = stages_payload.get("strategy")
            if isinstance(strategy_stage, Mapping):
                strategy_status = str(strategy_stage.get("status") or "").strip().lower()
                if strategy_status == "success":
                    logger.info(
                        "STRATEGY_RUN_SHORT_CIRCUIT sid=%s reason=stage_success",
                        sid,
                    )
                    working_stats.setdefault("sid", sid)
                    working_stats["planner_stage_status"] = "success"
                    working_stats["planner_plans_written"] = _stage_int(strategy_stage, "plans_written")
                    working_stats["planner_errors"] = _stage_int(strategy_stage, "planner_errors")
                    working_stats["planner_accounts_seen"] = _stage_int(strategy_stage, "accounts_seen")
                    working_stats["planner_accounts_with_openers"] = _stage_int(
                        strategy_stage, "accounts_with_openers"
                    )
                    working_stats.setdefault("planner_env_enabled", True)
                    working_stats.setdefault("planner_env_error", False)
                    working_stats.setdefault("planner_accounts_planned", [])
                    return working_stats

    strategy_stats = {
        "plans_written": 0,
        "planner_errors": 0,
        "accounts_seen": 0,
        "accounts_with_openers": 0,
    }

    planned_account_keys: list[str] = []
    planner_env_error = False
    try:
        planner_env = PlannerEnv.from_env()
    except ValueError as exc:
        logger.error("PLANNER_ENV_INVALID sid=%s error=%s", sid, exc)
        planner_env = None
        planner_env_error = True
        strategy_stats["planner_errors"] += 1
    else:
        logger.info(
            "PLANNER_ENV sid=%s enabled=%s mode=%s weekend=%s timezone=%s",
            sid,
            planner_env.enabled,
            planner_env.mode,
            sorted(planner_env.weekend),
            planner_env.timezone,
        )

    planner_env_enabled = bool(planner_env and planner_env.enabled)

    manifest_data: Mapping[str, Any] | None = None
    manifest_obj: RunManifest | None = None
    if planner_env and planner_env.enabled:
        manifest_obj = RunManifest.load_or_create(manifest_path, sid)
        manifest_obj.mark_strategy_started()
        manifest_obj.save()
        manifest_data = load_manifest_from_source(manifest_path)

    if planner_env and planner_env.enabled and manifest_data is not None:
        try:
            for acc_ctx in iterate_accounts(manifest_data):
                planner_result: PlannerHookResult | None = None
                try:
                    planner_result = maybe_run_planner_for_account(acc_ctx, planner_env)
                except Exception:  # pragma: no cover - defensive planner guard
                    logger.exception(
                        "PLANNER_STAGE_FAILED sid=%s account_id=%s",
                        sid,
                        acc_ctx.account_id,
                    )
                    strategy_stats["planner_errors"] += 1
                    continue

                if planner_result is None:
                    continue

                if planner_result.saw_account:
                    strategy_stats["accounts_seen"] += 1
                if planner_result.has_openers:
                    strategy_stats["accounts_with_openers"] += 1
                if planner_result.wrote_plan:
                    strategy_stats["plans_written"] += 1
                strategy_stats["planner_errors"] += int(planner_result.errors or 0)

                if planner_result.wrote_plan:
                    planned_account_keys.append(str(acc_ctx.account_key))
        except Exception:  # pragma: no cover - defensive iteration guard
            logger.exception(
                "PLANNER_ACCOUNT_ITERATION_FAILED sid=%s", sid
            )
            strategy_stats["planner_errors"] += 1

    if planner_env_error:
        stage_status = "error"
    elif strategy_stats["planner_errors"] > 0:
        stage_status = "error"
    elif planner_env_enabled:
        stage_status = "success"
    else:
        stage_status = "skipped"

    try:
        record_strategy_stage(
            base_root,
            sid,
            status=stage_status,
            plans_written=strategy_stats["plans_written"],
            planner_errors=strategy_stats["planner_errors"],
            accounts_seen=strategy_stats["accounts_seen"],
            accounts_with_openers=strategy_stats["accounts_with_openers"],
        )
    except Exception:  # pragma: no cover - defensive stage write
        logger.exception("PLANNER_RUNFLOW_RECORD_FAILED sid=%s", sid)

    logger.info(
        "AI_STRATEGY_SUMMARY sid=%s status=%s plans=%d errors=%d accounts_seen=%d accounts_with_openers=%d",
        sid,
        stage_status,
        strategy_stats["plans_written"],
        strategy_stats["planner_errors"],
        strategy_stats["accounts_seen"],
        strategy_stats["accounts_with_openers"],
    )

    accounts_root = base_root / sid / "cases" / "accounts"
    filesystem_accounts = _discover_strategy_accounts(accounts_root)
    planned_account_keys = sorted({str(acc_id) for acc_id in planned_account_keys} | set(filesystem_accounts))

    if planner_env_enabled and manifest_obj is not None:
        logger.info(
            "STRATEGY_MANIFEST_UPDATE_BEGIN sid=%s accounts=%s",
            sid,
            planned_account_keys,
        )
        # FIX 3: Reload manifest from disk right before writeback to pick up any
        # VALIDATION_MANIFEST_INJECTED / auto-repair changes that happened after
        # this task started. This prevents strategy from clobbering validation data.
        try:
            manifest_obj = RunManifest.for_sid(sid, runs_root=base_root, allow_create=False)
            logger.debug(
                "STRATEGY_MANIFEST_RELOADED sid=%s path=%s",
                sid,
                manifest_obj.path,
            )
        except Exception:
            logger.warning(
                "STRATEGY_MANIFEST_RELOAD_FAILED sid=%s, using stale instance",
                sid,
                exc_info=True,
            )
            # Fall back to the original manifest_obj if reload fails
        
        # Always persist planner outputs even if the stage start update failed earlier.
        for account_id in planned_account_keys:
            manifest_obj.register_strategy_artifacts_for_account(
                account_id,
                runs_root=base_root,
            )

        manifest_obj.mark_strategy_completed(
            {
                "plans_written": strategy_stats["plans_written"],
                "planner_errors": strategy_stats["planner_errors"],
                "accounts_seen": strategy_stats["accounts_seen"],
                "accounts_with_openers": strategy_stats["accounts_with_openers"],
            },
            failed=stage_status == "error",
            state=stage_status,
        ).save()
        logger.info(
            "STRATEGY_MANIFEST_UPDATE_DONE sid=%s accounts=%s state=%s",
            sid,
            planned_account_keys,
            stage_status,
        )
    elif planner_env_enabled:
        logger.error(
            "STRATEGY_MANIFEST_UPDATE_SKIPPED sid=%s reason=manifest_unavailable planned=%s",
            sid,
            planned_account_keys,
        )

    working_stats["planner_stage_status"] = stage_status
    working_stats["planner_plans_written"] = strategy_stats["plans_written"]
    working_stats["planner_errors"] = strategy_stats["planner_errors"]
    working_stats["planner_accounts_seen"] = strategy_stats["accounts_seen"]
    working_stats["planner_accounts_with_openers"] = strategy_stats["accounts_with_openers"]
    working_stats["planner_env_enabled"] = planner_env_enabled
    working_stats["planner_env_error"] = planner_env_error
    working_stats["planner_accounts_planned"] = planned_account_keys

    return working_stats


def run_validation_and_strategy_for_all_accounts(
    sid: str, *, runs_root: Path | str | None = None
) -> dict[str, object]:
    """Run validation requirements and the strategy planner for ``sid``."""
    import os
    # Quarantine legacy slowpath orchestration to dev/test only.
    if str(os.getenv("ENABLE_LEGACY_VALIDATION_ORCHESTRATION", "")).strip().lower() not in {"1", "true", "yes", "on"}:
        logger.info("LEGACY_VALIDATION_ORCHESTRATION_DISABLED sid=%s path=%s", sid, "AUTO_AI_SLOWPATH_VALIDATION")
        return {"sid": sid, "ok": False, "reason": "legacy_orchestrator_disabled"}

    stats = run_validation_requirements_for_all_accounts(sid, runs_root=runs_root)

    base_root = Path(runs_root) if runs_root is not None else RUNS_ROOT

    build_validation_packs_for_run(sid, runs_root=base_root)
    _maybe_send_validation_packs(sid, base_root, stage="validation")
    
    # Skip merge/apply in orchestrator mode; orchestrator finalizes
    if str(os.getenv("VALIDATION_ORCHESTRATOR_MODE", "1")).strip().lower() in {"", "0", "false", "no", "off"}:
        from backend.pipeline.validation_merge_helpers import (
            apply_validation_merge_and_update_state,
        )
        apply_validation_merge_and_update_state(
            sid,
            runs_root=base_root,
            source="auto_ai_pipeline",
        )

    stats = run_strategy_planner_for_all_accounts(
        sid, runs_root=base_root, stats=stats
    )
    return stats


def run_consistency_writeback_for_all_accounts(
    sid: str, *, runs_root: Path | str | None = None
) -> dict[str, object]:
    """Compute and persist field consistency snapshots for each account of ``sid``."""

    base_root = Path(runs_root) if runs_root is not None else RUNS_ROOT
    accounts_root = base_root / sid / "cases" / "accounts"

    stats = {
        "sid": sid,
        "total_accounts": 0,
        "processed_accounts": 0,
        "fields": 0,
        "missing_inputs": 0,
        "errors": 0,
    }

    if not accounts_root.exists():
        return stats

    account_paths = [path for path in accounts_root.iterdir() if path.is_dir()]
    for account_path in _maybe_slice(sorted(account_paths, key=_account_sort_key)):
        stats["total_accounts"] += 1

        bureaus_path = account_path / "bureaus.json"
        summary_path = account_path / "summary.json"

        if not (bureaus_path.exists() and summary_path.exists()):
            stats["missing_inputs"] += 1
            continue

        try:
            raw_text = bureaus_path.read_text(encoding="utf-8")
        except OSError:
            stats["errors"] += 1
            logger.warning(
                "CONSISTENCY_READ_FAILED sid=%s account_dir=%s", sid, account_path, exc_info=True
            )
            continue

        try:
            bureaus_payload = json.loads(raw_text)
        except json.JSONDecodeError:
            stats["errors"] += 1
            logger.warning(
                "CONSISTENCY_INVALID_JSON sid=%s account_dir=%s", sid, account_path, exc_info=True
            )
            continue

        if not isinstance(bureaus_payload, Mapping):
            stats["errors"] += 1
            logger.warning(
                "CONSISTENCY_INVALID_TYPE sid=%s account_dir=%s type=%s",
                sid,
                account_path,
                type(bureaus_payload).__name__,
            )
            continue

        try:
            field_consistency = compute_field_consistency(dict(bureaus_payload))
        except Exception:  # pragma: no cover - defensive logging
            stats["errors"] += 1
            logger.exception(
                "CONSISTENCY_COMPUTE_FAILED sid=%s account_dir=%s", sid, account_path
            )
            continue

        date_hits, amount_hits = _count_tolerance_hits(field_consistency)
        normalized_consistency = _normalize_consistency_without_raw(field_consistency)

        def _update_summary(summary_data: MutableMapping[str, object]) -> MutableMapping[str, object]:
            if not isinstance(summary_data, MutableMapping):
                summary_data = {}  # type: ignore[assignment]
            changed = False

            if normalized_consistency:
                existing = summary_data.get("field_consistency")
                if not isinstance(existing, Mapping) or dict(existing) != normalized_consistency:
                    summary_data["field_consistency"] = dict(normalized_consistency)
                    changed = True
            else:
                if summary_data.pop("field_consistency", None) is not None:
                    changed = True

            if changed and os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
                compact_merge_sections(summary_data)

            return summary_data

        try:
            update_json_in_place(summary_path, _update_summary)
        except ValueError:
            stats["errors"] += 1
            logger.warning(
                "CONSISTENCY_SUMMARY_UPDATE_FAILED sid=%s account_dir=%s",
                sid,
                account_path,
                exc_info=True,
            )
            continue

        stats["processed_accounts"] += 1
        stats["fields"] += len(normalized_consistency)

        runflow_step(
            sid,
            "validation",
            "compute_field_consistency",
            account=account_path.name,
            metrics={
                "fields": len(normalized_consistency),
                "date_tolerance_hits": date_hits,
                "amount_tolerance_hits": amount_hits,
            },
        )

        logger.info(
            "AI_CONSISTENCY_WRITEBACK sid=%s account=%s fields=%d",
            sid,
            account_path.name,
            len(normalized_consistency),
        )

    logger.info(
        "AI_CONSISTENCY_SUMMARY sid=%s accounts=%d processed=%d fields=%d missing=%d errors=%d",
        sid,
        stats["total_accounts"],
        stats["processed_accounts"],
        stats["fields"],
        stats["missing_inputs"],
        stats["errors"],
    )

    return stats


def _lock_age_seconds(path: Path, *, now: float | None = None) -> float | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    reference = now if now is not None else time.time()
    return max(0.0, reference - stat.st_mtime)


def _lock_is_stale(
    path: Path,
    *,
    ttl_seconds: int,
    now: float | None = None,
) -> bool:
    if ttl_seconds <= 0:
        return True
    age = _lock_age_seconds(path, now=now)
    if age is None:
        return True
    return age >= ttl_seconds


def maybe_run_auto_ai_pipeline(
    sid: str,
    *,
    summary: Mapping[str, object] | None = None,
    force: bool = False,
    inflight_ttl_seconds: int = DEFAULT_INFLIGHT_TTL_SECONDS,
    now: float | None = None,
) -> dict[str, object]:
    """Backward-compatible shim that queues the auto-AI pipeline."""

    _ = summary  # preserved for compatibility with older call sites
    manifest = RunManifest.for_sid(sid, allow_create=False)
    runs_root = manifest.path.parent.parent
    return maybe_queue_auto_ai_pipeline(
        sid,
        runs_root=runs_root,
        flag_env=os.environ,
        force=force,
        inflight_ttl_seconds=inflight_ttl_seconds,
        now=now,
    )


def maybe_queue_auto_ai_pipeline(
    sid: str,
    *,
    runs_root: Path,
    flag_env: Mapping[str, str],
    force: bool = False,
    inflight_ttl_seconds: int = DEFAULT_INFLIGHT_TTL_SECONDS,
    now: float | None = None,
) -> dict[str, object]:
    """Queue the automatic AI adjudication pipeline when enabled."""

    flag_value = str(flag_env.get("ENABLE_AUTO_AI_PIPELINE", "0"))
    if flag_value != "1":
        logger.info("AUTO_AI_SKIP_DISABLED sid=%s", sid)
        return {"queued": False, "reason": "disabled"}

    runs_root_path = Path(runs_root)
    run_dir = runs_root_path / sid
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        logger.warning(
            "RUN_DIR_CREATE_BLOCKED sid=%s caller=%s", sid, "auto_ai.maybe_queue"
        )
        return {"queued": False, "reason": "manifest_missing"}

    merge_paths = ensure_merge_paths(runs_root_path, sid, create=False)
    merge_paths.base.mkdir(parents=True, exist_ok=True)
    merge_paths.packs_dir.mkdir(parents=True, exist_ok=True)
    merge_paths.results_dir.mkdir(parents=True, exist_ok=True)
    base_dir = merge_paths.base
    packs_dir = merge_paths.packs_dir
    lock_path = base_dir / INFLIGHT_LOCK_FILENAME
    last_ok_path = base_dir / LAST_OK_FILENAME

    lock_exists = lock_path.exists()
    lock_age = _lock_age_seconds(lock_path, now=now) if lock_exists else None
    lock_stale = lock_exists and _lock_is_stale(
        lock_path, ttl_seconds=inflight_ttl_seconds, now=now
    )

    if lock_exists and not (lock_stale or force):
        logger.info(
            "AUTO_AI_SKIP_INFLIGHT sid=%s lock=%s age=%s ttl=%s",
            sid,
            lock_path,
            f"{lock_age:.1f}" if lock_age is not None else "unknown",
            inflight_ttl_seconds,
        )
        return {"queued": False, "reason": "inflight"}

    if lock_exists and (lock_stale or force):
        logger.info(
            "AUTO_AI_LOCK_CLEAR sid=%s lock=%s stale=%s force=%s age=%s",
            sid,
            lock_path,
            1 if lock_stale else 0,
            1 if force else 0,
            f"{lock_age:.1f}" if lock_age is not None else "unknown",
        )
        try:
            lock_path.unlink()
        except OSError:  # pragma: no cover - defensive logging
            logger.warning(
                "AUTO_AI_LOCK_CLEAR_FAILED sid=%s lock=%s", sid, lock_path, exc_info=True
            )

    # Candidate gate: allow override to proceed even when no AI merge best tags
    if not has_ai_merge_best_pairs(sid, runs_root_path):
        allow_on_empty = str(flag_env.get("AUTO_AI_QUEUE_ON_NO_CANDIDATES", "0")).strip() == "1"
        if allow_on_empty:
            logger.info("AUTO_AI_PROCEED_NO_CANDIDATES sid=%s override=1", sid)
        else:
            logger.info("AUTO_AI_SKIP_NO_CANDIDATES sid=%s", sid)
            logger.info("AUTO_AI_SKIPPED sid=%s reason=no_candidates", sid)
            return {"queued": False, "reason": "no_candidates"}

    run_dir = runs_root_path / sid
    accounts_dir = run_dir / "cases" / "accounts"

    lock_payload = {
        "sid": sid,
        "runs_root": str(runs_root_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if force:
        lock_payload["force"] = True

    try:
        base_dir.mkdir(parents=True, exist_ok=True)
        packs_dir.mkdir(parents=True, exist_ok=True)
        merge_paths.results_dir.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(lock_payload, ensure_ascii=False), encoding="utf-8")
    except OSError:  # pragma: no cover - defensive logging
        logger.warning("AUTO_AI_LOCK_WRITE_FAILED sid=%s path=%s", sid, lock_path)

    logger.info(
        "AUTO_AI_QUEUING sid=%s runs_root=%s accounts_dir=%s lock=%s",
        sid,
        runs_root_path,
        accounts_dir,
        lock_path,
    )

    try:
        from backend.pipeline import auto_ai_tasks

        task_id = auto_ai_tasks.enqueue_auto_ai_chain(sid, runs_root=runs_root_path)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_QUEUE_FAILED sid=%s", sid, exc_info=True)
        try:
            if lock_path.exists():
                lock_path.unlink()
        except OSError:
            logger.warning(
                "AUTO_AI_LOCK_CLEANUP_FAILED sid=%s path=%s", sid, lock_path
            )
        raise

    logger.info("AUTO_AI_QUEUED sid=%s", sid)
    manifest = RunManifest.for_sid(sid, allow_create=False)
    manifest.set_ai_enqueued()
    persist_manifest(manifest)
    logger.info("MANIFEST_AI_ENQUEUED sid=%s", sid)
    payload: dict[str, object] = {"queued": True, "reason": "queued"}
    if task_id:
        payload["task_id"] = task_id
    payload["lock_path"] = str(lock_path)
    payload["pipeline_dir"] = str(base_dir)
    payload["last_ok_path"] = str(last_ok_path)
    return payload


def maybe_trigger_validation_chain_if_merge_ready(
    sid: str,
    runs_root: Path | str | None = None,
    *,
    flag_env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """
    Centralized entrypoint to trigger the Auto-AI validation chain
    once merge is ready.

    Rules:
    - If merge_ready is False -> do NOT trigger, just log DEFER.
    - If merge_ready is True but validation is already success -> SKIP.
    - If merge_ready is True and validation is not complete -> trigger chain.

    This MUST NOT start validation before merge_ready=true.
    """
    from backend.runflow.decider import _compute_umbrella_barriers, get_runflow_snapshot
    
    # 1) Resolve runs_root
    if runs_root is None:
        runs_root = RUNS_ROOT
    runs_root_path = Path(runs_root)
    
    # 2) Load runflow / umbrella barriers
    run_dir = runs_root_path / sid
    
    try:
        barriers = _compute_umbrella_barriers(run_dir)
        merge_ready = barriers.get("merge_ready", False)
    except Exception:
        logger.debug(
            "VALIDATION_CHAIN_BARRIER_CHECK_FAILED sid=%s",
            sid,
            exc_info=True,
        )
        # Defensive: if we can't check barriers, don't trigger
        logger.warning(
            "VALIDATION_CHAIN_DEFER sid=%s reason=barrier_check_failed",
            sid,
        )
        return {"triggered": False, "reason": "barrier_check_failed"}
    
    # 3) Check validation status
    validation_status = None
    try:
        snapshot = get_runflow_snapshot(sid, runs_root=runs_root_path)
        validation_stage = snapshot.get("stages", {}).get("validation", {})
        validation_status = validation_stage.get("status")
    except Exception:
        logger.debug(
            "VALIDATION_CHAIN_STATUS_CHECK_FAILED sid=%s",
            sid,
            exc_info=True,
        )
        # Continue - if we can't check status, try to trigger
    
    # 4) Apply decision logic
    
    # If merge not ready, defer
    if not merge_ready:
        logger.info(
            "VALIDATION_CHAIN_DEFER sid=%s reason=merge_not_ready",
            sid,
        )
        return {"triggered": False, "reason": "merge_not_ready"}
    
    # If validation already successful, skip
    if validation_status == "success":
        logger.info(
            "VALIDATION_CHAIN_SKIP_ALREADY_DONE sid=%s reason=validation_success",
            sid,
        )
        return {"triggered": False, "reason": "validation_already_success"}
    
    # Merge is ready and validation not complete - trigger chain
    logger.info(
        "VALIDATION_CHAIN_TRIGGER sid=%s merge_ready=%s",
        sid,
        merge_ready,
    )
    
    result = maybe_queue_auto_ai_pipeline(
        sid,
        runs_root=runs_root_path,
        flag_env=flag_env or os.environ,
        force=False,
    )
    
    logger.info(
        "VALIDATION_CHAIN_ENQUEUED sid=%s result=%s",
        sid,
        result,
    )
    
    return {"triggered": True, "reason": "triggered", "result": result}


def has_ai_merge_best_tags(sid: str) -> bool:
    """Return ``True`` when any account tags request AI merge adjudication."""

    return has_ai_merge_best_pairs(sid, RUNS_ROOT)


def _as_amount(value: object) -> Decimal:
    """Best-effort conversion of ``value`` into a Decimal amount."""

    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return Decimal(0)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return Decimal(0)
        cleaned = cleaned.replace("$", "").replace(",", "")
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return Decimal(0)
    if isinstance(value, Mapping):
        if "amount" in value:
            return _as_amount(value.get("amount"))
    return Decimal(0)


def _load_account_flat_fields(
    accounts_root: Path, account_idx: int, cache: MutableMapping[int, Mapping[str, object] | None]
) -> Mapping[str, object] | None:
    if account_idx in cache:
        return cache[account_idx]

    fields_path = accounts_root / str(account_idx) / "fields_flat.json"
    try:
        raw = fields_path.read_text(encoding="utf-8")
    except OSError:
        cache[account_idx] = None
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        cache[account_idx] = None
        return None
    if not isinstance(payload, Mapping):
        cache[account_idx] = None
        return None

    cache[account_idx] = payload
    return payload


def _is_zero_debt_pair(a: Mapping[str, object] | None, b: Mapping[str, object] | None) -> bool:
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        return False

    a_bal = _as_amount(a.get("balance_owed", 0))
    b_bal = _as_amount(b.get("balance_owed", 0))
    a_due = _as_amount(a.get("past_due_amount", 0))
    b_due = _as_amount(b.get("past_due_amount", 0))
    return (a_bal == 0 == b_bal) and (a_due == 0 == b_due)


def has_ai_merge_best_pairs(sid: str, runs_root: Path | str) -> bool:
    """Return ``True`` if any account tags require AI merge adjudication."""

    runs_root_path = Path(runs_root)
    accounts_root = runs_root_path / sid / "cases" / "accounts"
    if not accounts_root.exists():
        return False

    cache: dict[int, Mapping[str, object] | None] = {}

    for tags_path in sorted(accounts_root.glob("*/tags.json")):
        try:
            account_idx = int(tags_path.parent.name)
        except ValueError:
            account_idx = None
        try:
            raw = tags_path.read_text(encoding="utf-8")
        except OSError:
            continue
        if not raw.strip():
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Skipping invalid JSON at %s", tags_path, exc_info=True)
            continue

        for tag in _iter_tag_entries(payload):
            if not _is_ai_merge_best_tag(tag):
                continue

            if account_idx is None:
                return True

            partner_idx_raw = tag.get("with")
            try:
                partner_idx = int(partner_idx_raw)
            except (TypeError, ValueError):
                return True

            account_fields = (
                _load_account_flat_fields(accounts_root, account_idx, cache)
                if account_idx is not None
                else None
            )
            partner_fields = _load_account_flat_fields(accounts_root, partner_idx, cache)

            if _is_zero_debt_pair(account_fields, partner_fields):
                logger.info(
                    "AI_CANDIDATE_SKIPPED_ZERO_DEBT sid=%s a=%s b=%s",
                    sid,
                    account_idx,
                    partner_idx,
                )
                continue

            return True

    return False


def _build_ai_packs(sid: str, runs_root: Path) -> None:
    argv = ["--sid", sid, "--runs-root", str(runs_root)]
    try:
        build_ai_merge_packs_main(argv)
    except SystemExit as exc:  # pragma: no cover - defensive
        if exc.code not in (None, 0):
            raise RuntimeError(
                f"build_ai_merge_packs failed for sid={sid} exit={exc.code}"
            ) from exc


def _send_ai_packs(sid: str, runs_root: Path | None = None) -> None:
    argv = ["--sid", sid]
    if runs_root is not None:
        argv.extend(["--runs-root", str(runs_root)])
    try:
        send_ai_merge_packs_main(argv)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise RuntimeError(
                f"send_ai_merge_packs failed for sid={sid} exit={exc.code}"
            ) from exc


def _normalize_indices(indices: Sequence[object]) -> set[int]:
    normalized: set[int] = set()
    for value in indices:
        try:
            normalized.add(int(value))
        except (TypeError, ValueError):
            continue
    return normalized


def _load_ai_index(path: Path) -> list[Mapping[str, object]]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid AI pack index JSON: {path}") from exc

    if isinstance(data, Mapping):
        pairs = data.get("pairs")
        if not isinstance(pairs, list):
            return []
        entries: list[Mapping[str, object]] = []
        for entry in pairs:
            if isinstance(entry, Mapping):
                entries.append(entry)
        return entries

    if isinstance(data, list):  # pragma: no cover - legacy support
        entries: list[Mapping[str, object]] = []
        for entry in data:
            if isinstance(entry, Mapping):
                entries.append(entry)
        return entries

    raise ValueError(f"AI pack index must be a list or mapping: {path}")


def _indices_from_index(index_entries: Iterable[Mapping[str, object]]) -> set[int]:
    values: set[int] = set()
    for entry in index_entries:
        for key in ("a", "b"):
            if key not in entry:
                continue
            try:
                values.add(int(entry[key]))
            except (TypeError, ValueError):
                continue
    return values


def _filter_zero_debt_index_entries(
    sid: str, runs_root: Path | str, entries: Sequence[Mapping[str, object]]
) -> tuple[list[Mapping[str, object]], list[tuple[int, int]]]:
    accounts_root = Path(runs_root) / sid / "cases" / "accounts"
    cache: dict[int, Mapping[str, object] | None] = {}

    kept: list[Mapping[str, object]] = []
    skipped: list[tuple[int, int]] = []

    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        a_raw = entry.get("a")
        b_raw = entry.get("b")
        try:
            a_idx = int(a_raw)
            b_idx = int(b_raw)
        except (TypeError, ValueError):
            kept.append(entry)
            continue

        account_fields = _load_account_flat_fields(accounts_root, a_idx, cache)
        partner_fields = _load_account_flat_fields(accounts_root, b_idx, cache)

        if _is_zero_debt_pair(account_fields, partner_fields):
            skipped.append((a_idx, b_idx))
            continue

        kept.append(entry)

    return kept, skipped


def _compact_accounts(accounts_dir: Path, indices: Iterable[int]) -> None:
    unique_indices = sorted({int(idx) for idx in indices})
    if not unique_indices:
        return

    for idx in unique_indices:
        account_dir = accounts_dir / f"{idx}"
        if not account_dir.exists():
            continue
        try:
            compact_account_tags(account_dir)
        except Exception:  # pragma: no cover - defensive logging
            logger.warning(
                "AUTO_AI_PIPELINE compact failed account=%s dir=%s",
                idx,
                account_dir,
                exc_info=True,
            )


def _iter_tag_entries(payload: object) -> Iterable[Mapping[str, object]]:
    if isinstance(payload, list):
        for entry in payload:
            if isinstance(entry, Mapping):
                yield entry
        return

    if isinstance(payload, Mapping):
        tags = payload.get("tags")
        if isinstance(tags, list):
            for entry in tags:
                if isinstance(entry, Mapping):
                    yield entry


def _is_ai_merge_best_tag(tag: Mapping[str, object]) -> bool:
    if not isinstance(tag, Mapping):
        return False

    kind = str(tag.get("kind", "")).strip().lower()
    if kind != "merge_best":
        return False

    decision_raw = tag.get("decision")
    decision = str(decision_raw).strip().lower() if decision_raw is not None else ""
    if decision != "ai":
        return False

    return True


@contextmanager
def ai_inflight_lock(runs_root: Path, sid: str):
    """
    Prevents concurrent AI runs on the same SID.
    Creates runs/<sid>/ai_packs/merge/inflight.lock; removes it on exit.
    """

    run_dir = Path(runs_root) / sid
    if not run_dir.exists():
        logger.warning(
            "RUN_DIR_CREATE_BLOCKED sid=%s caller=%s", sid, "auto_ai.ai_inflight_lock"
        )
        raise FileNotFoundError(f"run directory missing for {sid}")

    merge_paths = ensure_merge_paths(runs_root, sid, create=False)
    merge_paths.base.mkdir(parents=True, exist_ok=True)
    merge_paths.packs_dir.mkdir(parents=True, exist_ok=True)
    merge_paths.results_dir.mkdir(parents=True, exist_ok=True)
    ai_dir = merge_paths.base
    lock = ai_dir / INFLIGHT_LOCK_FILENAME
    if lock.exists():
        # Someone else is running; caller should skip.
        raise RuntimeError("AI pipeline already inflight")
    lock.write_text("1", encoding="utf-8")
    try:
        yield
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) in ("1", "true", "True")


def maybe_run_ai_pipeline(sid: str) -> dict[str, object]:
    """Queue the lightweight automatic AI pipeline after case building."""

    if not _env_flag("ENABLE_AUTO_AI_PIPELINE", "1"):
        return {"sid": sid, "skipped": "feature_off"}

    if not has_ai_merge_best_tags(sid):
        manifest = RunManifest.for_sid(sid, allow_create=False)
        manifest.set_ai_skipped("no_candidates")
        persist_manifest(manifest)
        logger.info("AUTO_AI_SKIPPED sid=%s reason=no_candidates", sid)
        return {"sid": sid, "skipped": "no_ai_candidates"}

    try:
        with ai_inflight_lock(RUNS_ROOT, sid):
            return _run_auto_ai_pipeline(sid)
    except RuntimeError:
        return {"sid": sid, "skipped": "inflight"}


@shared_task(name="pipeline.maybe_run_ai_pipeline")
def maybe_run_ai_pipeline_task(sid: str):
    """Celery task wrapper that launches the auto-AI pipeline asynchronously."""

    return maybe_run_ai_pipeline(sid)


def _run_auto_ai_pipeline(sid: str):
    # === 1) score → (re)write merge_* tags
    from backend.core.logic.merge.scorer import score_bureau_pairs_cli

    score_bureau_pairs_cli(sid=sid, write_tags=True, runs_root=RUNS_ROOT)

    if not has_ai_merge_best_pairs(sid, RUNS_ROOT):
        logger.info("AUTO_AI_BUILDER_BYPASSED_ZERO_DEBT sid=%s", sid)
        manifest = RunManifest.for_sid(sid, allow_create=False)
        manifest.set_ai_skipped("no_candidates")
        persist_manifest(manifest)
        logger.info("AUTO_AI_SKIPPED sid=%s reason=no_candidates", sid)
        return {"sid": sid, "skipped": "no_ai_candidates"}

    # === 2) build packs
    _build_ai_packs(sid, RUNS_ROOT)

    manifest = RunManifest.for_sid(sid, allow_create=False)
    merge_paths = ensure_merge_paths(RUNS_ROOT, sid, create=False)
    merge_paths.base.mkdir(parents=True, exist_ok=True)
    merge_paths.packs_dir.mkdir(parents=True, exist_ok=True)
    merge_paths.results_dir.mkdir(parents=True, exist_ok=True)
    base_dir = merge_paths.base
    packs_dir = merge_paths.packs_dir
    index_path = merge_paths.index_file

    index_read_path = index_path
    packs_source_dir = packs_dir
    legacy_dir = None
    if not index_read_path.exists():
        legacy_dir = probe_legacy_ai_packs(RUNS_ROOT, sid)
        if legacy_dir is not None:
            legacy_index = legacy_dir / "index.json"
            if legacy_index.exists():
                index_read_path = legacy_index
                packs_source_dir = legacy_dir

    manifest_pairs = 0
    ai_section = manifest.data.get("ai") if isinstance(manifest.data, dict) else {}
    if isinstance(ai_section, dict):
        packs_section = ai_section.get("packs")
        if isinstance(packs_section, dict):
            try:
                manifest_pairs = int(packs_section.get("pairs") or 0)
            except (TypeError, ValueError):
                manifest_pairs = 0

    if not index_read_path.exists():
        if manifest_pairs > 0:
            logger.error(
                "AUTO_AI_NO_PACKS_INDEX_MISSING sid=%s manifest_pairs=%d",
                sid,
                manifest_pairs,
            )
            return {"sid": sid, "skipped": "no_packs"}
        manifest.set_ai_skipped("no_packs")
        persist_manifest(manifest)
        logger.info(
            "AUTO_AI_SKIP_NO_PACKS sid=%s packs_dir=%s (index missing)",
            sid,
            packs_source_dir,
        )
        return {"sid": sid, "skipped": "no_packs"}

    try:
        index_data = json.loads(index_read_path.read_text(encoding="utf-8"))
        if not isinstance(index_data, dict):
            index_data = {}
    except Exception as exc:
        logger.exception(
            "AUTO_AI_SKIP_NO_PACKS sid=%s reason=index_load_error error=%s", sid, exc
        )
        if manifest_pairs > 0:
            logger.error(
                "AUTO_AI_NO_PACKS_INDEX_ERROR sid=%s manifest_pairs=%d",
                sid,
                manifest_pairs,
            )
            return {"sid": sid, "skipped": "no_packs"}
        manifest.set_ai_skipped("no_packs")
        persist_manifest(manifest)
        return {"sid": sid, "skipped": "no_packs"}

    packs_entries = index_data.get("packs")
    if isinstance(packs_entries, list):
        non_zero_entries, skipped_pairs = _filter_zero_debt_index_entries(
            sid, RUNS_ROOT, packs_entries
        )
        if skipped_pairs:
            for a_idx, b_idx in skipped_pairs:
                logger.info(
                    "AI_CANDIDATE_SKIPPED_ZERO_DEBT sid=%s a=%s b=%s", sid, a_idx, b_idx
                )
        if not non_zero_entries:
            logger.info("INDEX_ONLY_ZERO_DEBT_PAIRS sid=%s", sid)
            manifest.set_ai_skipped("no_packs")
            persist_manifest(manifest)
            try:
                index_path.unlink()
            except OSError:
                logger.debug(
                    "AUTO_AI_INDEX_UNLINK_FAILED sid=%s path=%s",
                    sid,
                    index_path,
                    exc_info=True,
                )
            return {"sid": sid, "skipped": "no_packs"}
        if len(non_zero_entries) != len(packs_entries):
            index_data["packs"] = non_zero_entries
            index_data["pairs_count"] = len(non_zero_entries)
            index_path.write_text(
                json.dumps(index_data, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    pairs_count = int(index_data.get("pairs_count") or len(index_data.get("packs") or []))

    if pairs_count <= 0:
        if manifest_pairs > 0:
            logger.error(
                "AUTO_AI_NO_PACKS_COUNT_MISMATCH sid=%s manifest_pairs=%d",
                sid,
                manifest_pairs,
            )
            return {"sid": sid, "skipped": "no_packs"}
        manifest.set_ai_skipped("no_packs")
        persist_manifest(manifest)
        logger.info(
            "AUTO_AI_SKIP_NO_PACKS sid=%s packs_dir=%s (pairs_count=0)",
            sid,
            packs_source_dir,
        )
        return {"sid": sid, "skipped": "no_packs"}

    logger.info(
        "AUTO_AI_PACKS_FOUND sid=%s dir=%s pairs=%d", sid, packs_source_dir, pairs_count
    )
    logger.info("AUTO_AI_BUILT sid=%s pairs=%d", sid, pairs_count)
    manifest = manifest.set_ai_built(base_dir, pairs_count)
    persist_manifest(manifest)

    # === 3) send to AI (writes ai_decision / same_debt_pair)
    argv = ["--sid", sid, "--packs-dir", str(packs_source_dir), "--runs-root", str(RUNS_ROOT)]
    send_ai_merge_packs_main(argv)

    manifest.set_ai_sent()
    persist_manifest(manifest)
    logger.info("AUTO_AI_SENT sid=%s dir=%s", sid, packs_source_dir)

    try:
        detect_and_persist_date_convention(sid, runs_root=RUNS_ROOT)
    except Exception:  # pragma: no cover - defensive logging
        logger.error("AUTO_AI_DATE_CONVENTION_FAILED sid=%s", sid, exc_info=True)

    # === 4) compact tags (keep only tags; push explanations to summary.json)
    compact_tags_for_sid(sid)
    manifest.set_ai_compacted()
    persist_manifest(manifest)
    logger.info("AUTO_AI_COMPACTED sid=%s", sid)

    # Validation + strategy now orchestrated only via enqueue_auto_ai_chain.
    # This function handles merge scoring/building/sending only.
    logger.info("AUTO_AI_DONE sid=%s", sid)
    return {"sid": sid, "ok": True}

