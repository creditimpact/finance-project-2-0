"""Validation summary orchestration helpers."""

from __future__ import annotations

import inspect
import json
import logging
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from backend.core.logic.validation_requirements import (
    apply_validation_summary,
    build_validation_requirements_for_account,
)
from backend.core.summary.summary_reader import (
    load_findings_from_summary as core_load_findings_from_summary,
)
from backend.validation.build_packs import (
    load_manifest_from_source,
    resolve_manifest_paths,
)
from backend.strategy.config import PlannerEnv
from backend.strategy.exceptions import StrategyPlannerError, UnsupportedDurationUnitError
from backend.strategy.io import (
    WrittenPaths,
    append_strategy_logs,
    load_findings_from_summary as strategy_load_findings,
)
from backend.strategy.manifest import update_manifest_for_account
from backend.strategy.planner import BUREAUS
from backend.strategy.runner import run_for_summary
from backend.strategy.runflow import record_strategy_stage

log = logging.getLogger(__name__)


@dataclass
class AccountContext:
    """Lightweight wrapper describing a single validation account."""

    sid: str
    runs_root: Path
    index: int | str
    account_key: str
    account_id: str
    account_dir: Path
    summary_path: Path
    bureaus_path: Path
    cached_findings: list[Mapping[str, Any]] | None = field(default=None)


SummaryBuilder = Callable[..., Mapping[str, Any]]
PackBuilder = Callable[[str, int | str, Path, Path], Sequence[Any]]
SendCallback = Callable[[AccountContext, Sequence[Mapping[str, Any]], Sequence[Any] | None], None]


@dataclass
class ValidationPipelineConfig:
    """Runtime configuration for the validation summary pipeline."""

    build_packs: bool = True
    summary_builder: SummaryBuilder | None = None
    pack_builder: PackBuilder | None = None
    send_callback: SendCallback | None = None


@dataclass
class PlannerHookResult:
    wrote_plan: bool = False
    errors: int = 0
    written_paths: WrittenPaths | None = None
    saw_account: bool = False
    has_openers: bool = False
    exit_status: str = "unstarted"


def iterate_accounts(manifest: Mapping[str, Any] | Path | str) -> Iterator[AccountContext]:
    """Yield :class:`AccountContext` objects for every account in ``manifest``."""

    _, paths, runs_root = _prepare_manifest(manifest)
    yield from _iter_account_dirs(paths.sid, runs_root, paths.accounts_dir)


def maybe_run_planner_for_account(acc_ctx: AccountContext, env: PlannerEnv) -> PlannerHookResult:
    """Run the strategy planner for ``acc_ctx`` when enabled."""

    result = PlannerHookResult()

    if not env.enabled:
        result.exit_status = "disabled"
        return result

    summary_path = acc_ctx.summary_path
    account_id = acc_ctx.account_id

    exit_status = "error"

    def _log_events(events: Iterable[Mapping[str, Any]]) -> None:
        append_strategy_logs(
            summary_path,
            events,
            account_subdir=env.account_subdir,
            account_id=account_id,
        )

    def _trim_trace() -> str:
        trace = traceback.format_exc()
        return trace if len(trace) <= 2000 else trace[-2000:]

    def _detect_per_bureau_outputs() -> bool:
        account_subdir = env.account_subdir or "strategy"
        strategy_root = summary_path.parent / account_subdir
        if not strategy_root.exists():
            return False
        for bureau in BUREAUS:
            master_candidate = strategy_root / bureau / env.master_name
            if master_candidate.exists():
                return True
        return False

    _log_events(
        (
            {
                "event": "planner_enter",
                "account": account_id,
                "mode": env.mode,
            },
        )
    )
    result.saw_account = True

    try:
        try:
            findings = strategy_load_findings(summary_path)
        except (UnsupportedDurationUnitError, StrategyPlannerError) as exc:
            log.warning(
                "PLANNER_LOAD_FAILED account_id=%s error=%s",
                account_id,
                exc,
            )
            result.errors = 1
            result.exit_status = "error"
            _log_events(
                (
                    {
                        "event": "planner_error",
                        "account": account_id,
                        "mode": env.mode,
                        "error": str(exc),
                        "trace": _trim_trace(),
                    },
                )
            )
            return result

        openers = [
            finding
            for finding in findings
            if (finding.category or "").strip().lower() != "natural_text"
            and (finding.default_decision or "").strip().lower() == "strong_actionable"
        ]

        if not openers:
            _log_events(
                (
                    {
                        "event": "planner_skipped_no_openers",
                        "account": account_id,
                        "mode": env.mode,
                        "reason": "no_openers",
                        "findings": len(findings),
                    },
                )
            )
            result.exit_status = "skipped"
            exit_status = "skipped"
            return result

        result.has_openers = True

        planner_mode = env.mode
        forced_start = env.forced_start

        log.info(
            "Planner delegating to per-bureau runner summary_path=%s account=%s",
            summary_path,
            account_id,
        )

        try:
            run_for_summary(
                summary_path,
                mode=planner_mode,
                forced_start=forced_start,
                env=env,
            )
        except Exception as exc:
            log.warning(
                "PLANNER_RUNNER_FAILED account_id=%s error=%s",
                account_id,
                exc,
            )
            result.errors = 1
            result.exit_status = "error"
            _log_events(
                (
                    {
                        "event": "planner_error",
                        "account": account_id,
                        "mode": env.mode,
                        "error": str(exc),
                        "trace": _trim_trace(),
                    },
                )
            )
            return result

        result.wrote_plan = _detect_per_bureau_outputs()
        result.exit_status = "written" if result.wrote_plan else "no_output"
        exit_status = result.exit_status
        return result
    finally:
        result.exit_status = exit_status
        _log_events(
            (
                {
                    "event": "planner_exit",
                    "account": account_id,
                    "mode": env.mode,
                    "status": exit_status,
                },
            )
        )


def write_summary_for_account(
    acc_ctx: AccountContext,
    *,
    cfg: ValidationPipelineConfig | None = None,
    runs_root: Path | str | None = None,
    sid: str | None = None,
) -> Mapping[str, Any]:
    """Build and persist validation requirements for ``acc_ctx``."""

    if runs_root is not None:
        acc_ctx.runs_root = Path(runs_root)
    if sid is not None:
        acc_ctx.sid = str(sid)

    config = cfg or ValidationPipelineConfig()
    builder = config.summary_builder or _default_summary_builder

    raw_result = _invoke_summary_builder(builder, acc_ctx)

    if isinstance(raw_result, Mapping):
        result: dict[str, Any] = dict(raw_result)
    else:
        result = {}

    block = result.get("validation_requirements")
    if not isinstance(block, Mapping):
        empty_payload: Mapping[str, Any] = {"schema_version": 3, "findings": []}
        try:
            summary_after = apply_validation_summary(acc_ctx.summary_path, empty_payload)
        except Exception:  # pragma: no cover - defensive summary guard
            log.exception(
                "SUMMARY_WRITE_EMPTY_FAILED sid=%s account_id=%s", acc_ctx.sid, acc_ctx.account_id
            )
            result.setdefault("validation_requirements", dict(empty_payload))
        else:
            normalized_block = (
                summary_after.get("validation_requirements")
                if isinstance(summary_after, Mapping)
                else None
            )
            if isinstance(normalized_block, Mapping):
                result["validation_requirements"] = dict(normalized_block)
            else:
                result["validation_requirements"] = dict(empty_payload)
        block = result.get("validation_requirements")

    findings = _sanitize_findings(block.get("findings")) if isinstance(block, Mapping) else []
    acc_ctx.cached_findings = findings

    return result


def _invoke_summary_builder(
    builder: SummaryBuilder,
    acc_ctx: AccountContext,
) -> Mapping[str, Any]:
    """Invoke ``builder`` with ``acc_ctx`` using optional context hints."""

    try:
        signature = inspect.signature(builder)
    except (TypeError, ValueError):
        signature = None

    if signature is None:
        return builder(acc_ctx.account_dir)

    accepts_var_kw = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )

    kwargs: dict[str, Any] = {}

    if accepts_var_kw or "sid" in signature.parameters:
        kwargs["sid"] = acc_ctx.sid

    if accepts_var_kw or "runs_root" in signature.parameters:
        kwargs["runs_root"] = acc_ctx.runs_root

    return builder(acc_ctx.account_dir, **kwargs)


def load_findings_from_summary(
    runs_root: Path | str,
    sid: str,
    account_key: int | str,
) -> list[Mapping[str, Any]]:
    """Return cached findings for ``account_key`` from summary.json."""

    raw_findings = core_load_findings_from_summary(runs_root, sid, account_key)
    return _sanitize_findings(raw_findings)


def build_and_queue_packs(
    acc_ctx: AccountContext,
    *,
    findings: Sequence[Mapping[str, Any]] | None,
    cfg: ValidationPipelineConfig | None = None,
) -> Sequence[Any]:
    """Build validation packs for ``acc_ctx`` and optionally enqueue send."""

    config = cfg or ValidationPipelineConfig()
    if not config.build_packs:
        return []

    builder = config.pack_builder or _default_pack_builder
    try:
        pack_lines = builder(
            acc_ctx.sid,
            acc_ctx.index,
            acc_ctx.summary_path,
            acc_ctx.bureaus_path,
        )
    except Exception:  # pragma: no cover - defensive pack builder guard
        log.exception(
            "VALIDATION_PACK_BUILD_FAILED sid=%s account_id=%s",
            acc_ctx.sid,
            acc_ctx.account_id,
        )
        return []

    callback = config.send_callback
    if callback is not None:
        try:
            callback(acc_ctx, list(findings or []), pack_lines)
        except Exception:  # pragma: no cover - defensive queue handling
            log.exception(
                "VALIDATION_PACK_QUEUE_FAILED sid=%s account_id=%s",
                acc_ctx.sid,
                acc_ctx.account_id,
            )

    return pack_lines


def run_validation_summary_pipeline(
    manifest: Mapping[str, Any] | Path | str,
    *,
    cfg: ValidationPipelineConfig | None = None,
) -> dict[str, Any]:
    """Build validation summaries (and packs) for every account in ``manifest``."""

    config = cfg or ValidationPipelineConfig()
    _, paths, runs_root = _prepare_manifest(manifest)
    
    # Mandatory initialization: Ensure validation manifest paths are populated
    # BEFORE any pack building occurs. This prevents missing paths in manifest.
    try:
        from backend.ai.manifest import ensure_validation_section
        ensure_validation_section(paths.sid, runs_root=runs_root)
    except Exception:
        log.exception("VALIDATION_MANIFEST_INIT_FAILED sid=%s", paths.sid)
        raise

    stats = {
        "sid": paths.sid,
        "total_accounts": 0,
        "summaries_written": 0,
        "planner_plans_written": 0,
        "planner_errors": 0,
        "planner_accounts_seen": 0,
        "planner_accounts_with_openers": 0,
        "packs_built": 0,
        "skipped_accounts": 0,
        "errors": 0,
    }

    run_dir = runs_root / paths.sid
    manifest_path = run_dir / "manifest.json"

    stats_strategy = {
        "plans_written": 0,
        "planner_errors": 0,
        "accounts_seen": 0,
        "accounts_with_openers": 0,
    }

    planner_env_error = False
    try:
        planner_env = PlannerEnv.from_env()
    except ValueError as exc:
        log.error("PLANNER_ENV_INVALID sid=%s error=%s", paths.sid, exc)
        planner_env = None
        planner_env_error = True
        stats_strategy["planner_errors"] += 1
        stats["planner_errors"] += 1
    else:
        log.info(
            "PLANNER_ENV sid=%s enabled=%s mode=%s weekend=%s timezone=%s",
            paths.sid,
            planner_env.enabled,
            planner_env.mode,
            sorted(planner_env.weekend),
            planner_env.timezone,
        )

    planner_env_enabled = bool(planner_env and planner_env.enabled)

    for acc_ctx in _iter_account_dirs(paths.sid, runs_root, paths.accounts_dir):
        stats["total_accounts"] += 1
        planner_result: PlannerHookResult | None = None
        try:
            result = write_summary_for_account(acc_ctx, cfg=config)
            status = str(result.get("status") or "") if isinstance(result, Mapping) else ""

            # ── Tradeline Check Hook ──────────────────────────────────────
            # Run tradeline_check per account if enabled (env gated).
            # This runs BEFORE strategy planner within the same loop iteration.
            try:
                from backend.tradeline_check import run_for_account as run_tradeline_check
                tradeline_result = run_tradeline_check(acc_ctx)
                log.debug(
                    "TRADELINE_CHECK_RESULT sid=%s account_id=%s result=%s",
                    acc_ctx.sid,
                    acc_ctx.account_id,
                    tradeline_result,
                )
            except Exception:  # pragma: no cover - defensive tradeline_check guard
                log.exception(
                    "TRADELINE_CHECK_STAGE_FAILED sid=%s account_id=%s",
                    acc_ctx.sid,
                    acc_ctx.account_id,
                )
            # ──────────────────────────────────────────────────────────────

            if planner_env and planner_env.enabled:
                try:
                    planner_result = maybe_run_planner_for_account(acc_ctx, planner_env)
                except Exception:  # pragma: no cover - defensive planner guard
                    log.exception(
                        "PLANNER_STAGE_FAILED sid=%s account_id=%s",
                        acc_ctx.sid,
                        acc_ctx.account_id,
                    )
                    stats_strategy["planner_errors"] += 1
                    stats["planner_errors"] += 1
                    planner_result = None
                if planner_result is not None:
                    if planner_result.saw_account:
                        stats_strategy["accounts_seen"] += 1
                    if planner_result.has_openers:
                        stats_strategy["accounts_with_openers"] += 1
                    if planner_result.wrote_plan:
                        stats_strategy["plans_written"] += 1
                        stats["planner_plans_written"] += 1
                    stats_strategy["planner_errors"] += planner_result.errors
                    stats["planner_errors"] += planner_result.errors

                    if planner_result.written_paths:
                        if not manifest_path.exists():
                            log.warning(
                                "PLANNER_MANIFEST_MISSING sid=%s manifest=%s",
                                acc_ctx.sid,
                                manifest_path,
                            )
                            stats_strategy["planner_errors"] += 1
                            stats["planner_errors"] += 1
                        else:
                            try:
                                weekdays_map = {
                                    key: Path(value)
                                    for key, value in planner_result.written_paths["weekdays"].items()
                                }
                                log_value = planner_result.written_paths.get("log")
                                log_path = Path(log_value) if log_value else None
                                update_manifest_for_account(
                                    manifest_path,
                                    acc_ctx.account_key,
                                    Path(planner_result.written_paths["dir"]),
                                    Path(planner_result.written_paths["master"]),
                                    weekdays_map,
                                    log_path=log_path,
                                )
                            except Exception:
                                log.exception(
                                    "PLANNER_MANIFEST_UPDATE_FAILED sid=%s account_id=%s",
                                    acc_ctx.sid,
                                    acc_ctx.account_id,
                                )
                                stats_strategy["planner_errors"] += 1
                                stats["planner_errors"] += 1

            if status != "ok":
                stats["skipped_accounts"] += 1
                continue

            stats["summaries_written"] += 1
            findings = acc_ctx.cached_findings
            if findings is None:
                raw_findings = core_load_findings_from_summary(
                    acc_ctx.runs_root, acc_ctx.sid, acc_ctx.account_key
                )
                findings = _sanitize_findings(raw_findings)
                acc_ctx.cached_findings = findings

            if config.build_packs and _should_queue_pack(findings):
                pack_lines = build_and_queue_packs(
                    acc_ctx, findings=findings, cfg=config
                )
                if pack_lines:
                    stats["packs_built"] += 1
        except Exception:  # pragma: no cover - defensive pipeline guard
            stats["errors"] += 1
            log.exception(
                "ACCOUNT FAILED, continuing. account_id=%s",
                acc_ctx.account_id,
            )
            continue

    try:
        if planner_env_error:
            stage_status = "error"
        elif stats_strategy["planner_errors"] > 0:
            stage_status = "error"
        elif planner_env_enabled:
            stage_status = "success"
        else:
            stage_status = "skipped"

        record_strategy_stage(
            runs_root,
            paths.sid,
            status=stage_status,
            plans_written=stats_strategy["plans_written"],
            planner_errors=stats_strategy["planner_errors"],
            accounts_seen=stats_strategy["accounts_seen"],
            accounts_with_openers=stats_strategy["accounts_with_openers"],
        )
    except Exception:  # pragma: no cover - defensive stage write
        log.exception("PLANNER_RUNFLOW_RECORD_FAILED sid=%s", paths.sid)

    stats["planner_accounts_seen"] = stats_strategy["accounts_seen"]
    stats["planner_accounts_with_openers"] = stats_strategy["accounts_with_openers"]

    return stats


def _prepare_manifest(
    manifest: Mapping[str, Any] | Path | str,
):
    manifest_data = load_manifest_from_source(manifest)
    try:
        paths = resolve_manifest_paths(manifest_data)
        runs_root = _infer_runs_root(paths.accounts_dir, paths.sid)
        return manifest_data, paths, runs_root
    except ValueError as exc:
        # Auto-repair missing validation pack paths in manifest when possible
        missing_keys = str(exc)
        needs_validation_paths = (
            "ai.packs.validation.packs_dir" in missing_keys
            or "ai.packs.validation.results_dir" in missing_keys
            or "ai.packs.validation.index" in missing_keys
            or "ai.packs.validation.logs" in missing_keys
        )
        if not needs_validation_paths:
            raise

        try:
            sid = str(manifest_data.get("sid") or "").strip()
            base_dirs = manifest_data.get("base_dirs")
            accounts_dir_raw = base_dirs.get("cases_accounts_dir") if isinstance(base_dirs, Mapping) else None
            accounts_dir = Path(str(accounts_dir_raw)).resolve() if accounts_dir_raw else None
            runs_root = _infer_runs_root(accounts_dir, sid) if accounts_dir is not None else None
        except Exception:
            sid = ""
            runs_root = None

        if sid and runs_root is not None:
            try:
                from backend.ai.manifest import ensure_validation_section
                ensure_validation_section(sid, runs_root=runs_root)
                # Reload from disk to pick up repaired values
                if isinstance(manifest, (str, Path)):
                    manifest_data = load_manifest_from_source(manifest)
                else:
                    # Attempt to read canonical manifest file under runs_root
                    candidate = Path(runs_root) / sid / "manifest.json"
                    if candidate.exists():
                        manifest_data = load_manifest_from_source(candidate)
                paths = resolve_manifest_paths(manifest_data)
                runs_root = _infer_runs_root(paths.accounts_dir, paths.sid)
                log.info(
                    "VALIDATION_MANIFEST_AUTO_REPAIRED sid=%s packs_dir=%s",
                    sid,
                    getattr(paths, "packs_dir", None),
                )
                return manifest_data, paths, runs_root
            except Exception:
                log.exception("VALIDATION_MANIFEST_REPAIR_FAILED sid=%s", sid)
                raise
        # If we couldn't repair, re-raise original error
        raise


def _iter_account_dirs(
    sid: str, runs_root: Path, accounts_dir: Path
) -> Iterable[AccountContext]:
    if not accounts_dir.is_dir():
        return []

    for account_dir in sorted(accounts_dir.iterdir(), key=_account_sort_key):
        if not account_dir.is_dir():
            continue

        account_key = account_dir.name
        coerced_index = _coerce_account_index(account_key)
        account_id = _resolve_account_id(account_dir, account_key)

        yield AccountContext(
            sid=sid,
            runs_root=runs_root,
            index=coerced_index if coerced_index is not None else account_key,
            account_key=account_key,
            account_id=account_id,
            account_dir=account_dir,
            summary_path=account_dir / "summary.json",
            bureaus_path=account_dir / "bureaus.json",
        )


def _account_sort_key(path: Path) -> tuple[int, Any]:
    name = path.name
    try:
        return (0, int(name))
    except (TypeError, ValueError):
        return (1, name)


def _coerce_account_index(value: str) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _resolve_account_id(account_dir: Path, default: str) -> str:
    meta_path = account_dir / "meta.json"
    try:
        raw_text = meta_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return str(default)
    except OSError:
        log.debug("META_READ_FAILED path=%s", meta_path, exc_info=True)
        return str(default)

    try:
        meta = json.loads(raw_text)
    except Exception:
        log.debug("META_PARSE_FAILED path=%s", meta_path, exc_info=True)
        return str(default)

    if not isinstance(meta, Mapping):
        return str(default)

    for key in ("account_id", "accountId", "account", "id"):
        value = meta.get(key)
        if value:
            return str(value)

    idx_value = meta.get("account_index") or meta.get("index")
    if idx_value:
        return str(idx_value)

    return str(default)


def _sanitize_findings(findings: Any) -> list[Mapping[str, Any]]:
    if not isinstance(findings, Sequence) or isinstance(findings, (str, bytes, bytearray)):
        return []

    sanitized: list[Mapping[str, Any]] = []
    for entry in findings:
        if isinstance(entry, Mapping):
            sanitized.append(dict(entry))
    return sanitized


def _should_queue_pack(findings: Sequence[Mapping[str, Any]] | None) -> bool:
    if not findings:
        return False
    return any(bool(entry.get("send_to_ai")) for entry in findings if isinstance(entry, Mapping))


def _infer_runs_root(accounts_dir: Path, sid: str) -> Path:
    resolved = accounts_dir.resolve()
    for parent in resolved.parents:
        if parent.name == sid:
            return parent.parent.resolve()
    try:
        return resolved.parents[2].resolve()
    except IndexError:
        return resolved.parent.resolve()


def _default_summary_builder(
    account_dir: str | Path,
    *,
    sid: str | None = None,
    runs_root: Path | str | None = None,
) -> Mapping[str, Any]:
    return build_validation_requirements_for_account(
        account_dir,
        build_pack=False,
        sid=sid,
        runs_root=runs_root,
    )


def _default_pack_builder(
    sid: str, account_key: int | str, summary_path: Path, bureaus_path: Path
) -> Sequence[Any]:
    from backend.ai.validation_builder import build_validation_pack_for_account

    return build_validation_pack_for_account(sid, account_key, summary_path, bureaus_path)


__all__ = [
    "AccountContext",
    "ValidationPipelineConfig",
    "build_and_queue_packs",
    "iterate_accounts",
    "load_findings_from_summary",
    "run_validation_summary_pipeline",
    "write_summary_for_account",
]

