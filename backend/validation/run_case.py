"""Orchestrate Validation AI pack creation and inference for a case run."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.config import ENABLE_VALIDATION_AI, PREVALIDATION_DETECT_DATES

from .build_packs import resolve_manifest_paths
from .pipeline import _infer_runs_root
from .preflight import detect_dates_for_sid
from .send_packs import send_validation_packs

log = logging.getLogger(__name__)


def _read_manifest(manifest_path: Path | str) -> Mapping[str, Any]:
    manifest_text = Path(manifest_path).read_text(encoding="utf-8")
    return json.loads(manifest_text)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _append_log(log_path: Path, event: str, **payload: Any) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": _utc_now(), "event": event, **payload}
    serialized = json.dumps(record, ensure_ascii=False, sort_keys=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(serialized + "\n")


def run_case(manifest_path: Path | str) -> Mapping[str, Any]:
    """Execute the validation AI build/send flow for ``manifest_path``."""

    manifest = _read_manifest(manifest_path)
    
    # Extract SID to initialize validation section before resolving paths
    sid = manifest.get("sid")
    if not sid:
        raise ValueError("Manifest missing 'sid' field")
    
    # Infer runs_root early to enable validation section initialization
    base_dirs = manifest.get("base_dirs", {})
    accounts_dir_raw = base_dirs.get("cases_accounts_dir")
    if accounts_dir_raw:
        accounts_dir_candidate = Path(str(accounts_dir_raw)).resolve()
        runs_root = _infer_runs_root(accounts_dir_candidate, sid)
    else:
        # Fallback: assume manifest is in runs/<sid>/manifest.json
        manifest_parent = Path(manifest_path).resolve().parent
        runs_root = manifest_parent.parent
    
    # Mandatory initialization: Ensure validation manifest paths exist
    # BEFORE resolve_manifest_paths attempts to read them
    try:
        from backend.ai.manifest import ensure_validation_section
        ensure_validation_section(sid, runs_root=runs_root)
        # Reload manifest after initialization to pick up any changes written to disk
        manifest = _read_manifest(manifest_path)
    except Exception:
        log.exception("VALIDATION_MANIFEST_INIT_FAILED sid=%s in run_case", sid)
        raise
    
    paths = resolve_manifest_paths(manifest)
    runs_root = _infer_runs_root(paths.accounts_dir, paths.sid)

    if PREVALIDATION_DETECT_DATES:
        try:
            detect_dates_for_sid(paths.sid, runs_root=runs_root)
        except Exception:  # pragma: no cover - defensive logging
            log.exception(
                "VALIDATION_DATE_DETECT_FAILED sid=%s runs_root=%s",
                paths.sid,
                runs_root,
            )

    if not ENABLE_VALIDATION_AI:
        _append_log(paths.log_path, "validation_ai_skipped", reason="disabled")
        return {"enabled": False, "reason": "validation_ai_disabled"}

    _append_log(paths.log_path, "validation_ai_start", sid=paths.sid)
    try:
        from .pipeline import ValidationPipelineConfig, run_validation_summary_pipeline

        pipeline_cfg = ValidationPipelineConfig()
        summary_stats = run_validation_summary_pipeline(manifest, cfg=pipeline_cfg)
        _append_log(
            paths.log_path,
            "validation_ai_built",
            sid=paths.sid,
            total_accounts=summary_stats.get("total_accounts", 0),
            summaries=summary_stats.get("summaries_written", 0),
            packs=summary_stats.get("packs_built", 0),
            skipped=summary_stats.get("skipped_accounts", 0),
            errors=summary_stats.get("errors", 0),
        )

        _append_log(paths.log_path, "VALIDATION_PIPELINE_ENTRY", sid=paths.sid, path="VALIDATION_RUN_CASE")
        send_results = send_validation_packs(manifest)
        _append_log(
            paths.log_path,
            "validation_ai_sent",
            sid=paths.sid,
            accounts=len(send_results),
        )
        # Automatically merge AI results into summaries (disabled in orchestrator mode)
        import os as _os
        if _os.getenv("VALIDATION_ORCHESTRATOR_MODE", "1").strip().lower() in {"", "0", "false", "no", "off"}:
            try:
                from backend.pipeline.validation_merge_helpers import (
                    apply_validation_merge_and_update_state,
                )
                stats = apply_validation_merge_and_update_state(
                    paths.sid,
                    runs_root=runs_root,
                    source="run_case",
                )
                _append_log(
                    paths.log_path,
                    "VALIDATION_AI_MERGE_APPLIED",
                    sid=paths.sid,
                    accounts=int(stats.get("accounts_updated", 0) or 0),
                    fields=int(stats.get("fields_updated", 0) or 0),
                )
            except Exception as exc:
                _append_log(paths.log_path, "VALIDATION_AI_MERGE_FAILED", sid=paths.sid, error=str(exc))

        # Trigger strategy recovery chain if validation is complete and strategy is required
        try:
            from backend.runflow.decider import reconcile_umbrella_barriers
            reconcile_umbrella_barriers(paths.sid, runs_root=runs_root)
            _append_log(paths.log_path, "RUN_CASE_BARRIERS_RECONCILED", sid=paths.sid)
        except Exception as exc:
            _append_log(paths.log_path, "RUN_CASE_BARRIERS_RECONCILE_FAILED", sid=paths.sid, error=str(exc))
    except Exception as exc:
        _append_log(
            paths.log_path,
            "validation_ai_error",
            sid=paths.sid,
            error=str(exc),
        )
        raise

    return {
        "enabled": True,
        "build": summary_stats,
        "send": send_results,
    }


__all__ = ["run_case"]
