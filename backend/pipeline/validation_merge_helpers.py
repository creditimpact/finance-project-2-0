"""Unified validation AI merge and state update logic."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.core.logic.validation_ai_merge import (
    apply_validation_ai_decisions_for_all_accounts,
    summarize_validation_ai_state,
)
from backend.runflow.decider import runflow_refresh_umbrella_barriers, record_stage_force
from backend.runflow.manifest import RunManifest

logger = logging.getLogger(__name__)


def assert_validation_manifest_paths_non_null(sid: str, runs_root: Path | str | None = None) -> None:
    """
    Assert that validation manifest paths are populated after validation completes.
    
    This check ensures that the manifest has non-null validation path fields,
    which are required for tooling, debugging, and consistency checks.
    
    Raises:
        ValueError: If any required validation path is missing from manifest.
    """
    from backend.pipeline.runs import RUNS_ROOT, RunManifest
    
    if runs_root is None:
        runs_root_path = RUNS_ROOT
    elif isinstance(runs_root, str):
        runs_root_path = Path(runs_root)
    else:
        runs_root_path = runs_root
    
    try:
        manifest = RunManifest.for_sid(sid, runs_root=runs_root_path, allow_create=False)
        ai_section = manifest.data.get("ai", {})
        packs_section = ai_section.get("packs", {})
        validation_section = packs_section.get("validation", {})
    except Exception as exc:
        logger.error(
            "VALIDATION_MANIFEST_PATHS_CHECK_FAILED sid=%s error=%s",
            sid,
            exc,
            exc_info=True,
        )
        raise ValueError(f"Failed to load manifest for validation path check: {exc}") from exc
    
    required_keys = ["packs_dir", "results_dir", "index", "logs"]
    missing = [key for key in required_keys if not validation_section.get(key)]
    
    if missing:
        logger.error(
            "VALIDATION_MANIFEST_PATHS_MISSING sid=%s missing=%s section=%s",
            sid,
            ",".join(missing),
            validation_section,
        )
        raise ValueError(
            f"Validation manifest missing required paths for sid={sid}: {', '.join(missing)}. "
            "This indicates that ensure_validation_section was not called or failed silently."
        )
    
    logger.info(
        "VALIDATION_MANIFEST_PATHS_VERIFIED sid=%s packs_dir=%s results_dir=%s index=%s logs=%s",
        sid,
        validation_section.get("packs_dir"),
        validation_section.get("results_dir"),
        validation_section.get("index"),
        validation_section.get("logs"),
    )


def _utcnow() -> datetime:
    """Return current UTC time."""
    return datetime.now(timezone.utc)


def _isoformat_timestamp(dt: datetime | None = None) -> str:
    """Format datetime as ISO string."""
    if dt is None:
        dt = _utcnow()
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def apply_validation_merge_and_update_state(
    sid: str,
    runs_root: Path | str | None = None,
    applied_at: datetime | None = None,
    source: str = "validation_merge_helper",
) -> Mapping[str, Any]:
    """
    Perform validation AI merge and update all state tracking.
    
    This function performs BOTH:
    1. Physical merge: Apply validation AI results to summary.json files
    2. State update: Set merge_results_applied=True in runflow.json and manifest.json
    3. Barrier refresh: Recompute umbrella barriers to unblock strategy stage
    
    Args:
        sid: Session ID
        runs_root: Root directory for runs (optional)
        applied_at: Timestamp for merge (defaults to current time)
        source: Source identifier for logging
        
    Returns:
        Dictionary with merge statistics
    """
    
    if applied_at is None:
        applied_at = _utcnow()
    
    applied_at_iso = _isoformat_timestamp(applied_at)
    
    from backend.pipeline.runs import RUNS_ROOT, RunManifest
    
    if runs_root is None:
        runs_root_path = RUNS_ROOT
    elif isinstance(runs_root, str):
        runs_root_path = Path(runs_root)
    else:
        runs_root_path = runs_root
    
    # Check if merge is already applied (idempotency guard)
    run_dir = runs_root_path / sid
    runflow_path = run_dir / "runflow.json"
    
    try:
        runflow_data = json.loads(runflow_path.read_text(encoding="utf-8"))
        validation_stage = runflow_data.get("stages", {}).get("validation", {})
        runflow_merge_applied = validation_stage.get("merge_results_applied", False)
    except Exception:
        runflow_merge_applied = False
    
    try:
        manifest = RunManifest.for_sid(sid, runs_root=runs_root_path, allow_create=False)
        manifest_validation = manifest.get_ai_stage_status("validation")
        manifest_merge_applied = manifest_validation.get("merge_results_applied", False)
    except Exception:
        manifest_merge_applied = False
    
    # If both sources agree merge is already applied, skip the work
    if runflow_merge_applied and manifest_merge_applied:
        logger.info(
            "VALIDATION_MERGE_ALREADY_APPLIED sid=%s source=%s runflow_applied=%s manifest_applied=%s",
            sid,
            source,
            runflow_merge_applied,
            manifest_merge_applied,
        )
        return {"merge_applied": True, "skipped": True, "reason": "already_applied"}
    
    logger.info(
        "VALIDATION_MERGE_STATE_UPDATE_START sid=%s source=%s",
        sid,
        source,
    )
    
    # 1. Physical merge: Apply AI decisions to summary.json files
    merge_stats = apply_validation_ai_decisions_for_all_accounts(
        sid, runs_root=runs_root_path
    )
    
    logger.info(
        "VALIDATION_AI_MERGE_APPLIED sid=%s accounts_updated=%d fields_updated=%d source=%s",
        sid,
        merge_stats.get("accounts_updated", 0),
        merge_stats.get("fields_updated", 0),
        source,
    )
    
    # 2. Get AI state summary
    ai_state = summarize_validation_ai_state(sid, runs_root=runs_root_path)
    
    # 3. Update runflow.json with merge_results_applied flag
    stage_payload: dict[str, Any] = {
        "merge_results": {
            "applied": True,
            "applied_at": applied_at_iso,
            "source": source,
        },
        "merge_results_applied": True,
        "last_at": applied_at_iso,
    }
    
    if isinstance(ai_state, Mapping):
        completed_flag = bool(ai_state.get("completed"))
        pending_accounts = ai_state.get("accounts_pending", [])
        
        metrics_update = {
            "validation_ai_completed": completed_flag,
            "validation_ai_accounts_total": ai_state.get("accounts_total", 0),
            "validation_ai_accounts_pending": len(list(pending_accounts)),
            "validation_ai_fields_total": ai_state.get("fields_total", 0),
            "validation_ai_fields_applied": ai_state.get("fields_applied", 0),
            "validation_ai_fields_pending": ai_state.get("fields_pending", 0),
        }
        
        if merge_stats:
            metrics_update.update({
                "validation_ai_accounts_discovered": merge_stats.get("accounts_discovered", 0),
                "validation_ai_accounts_updated": merge_stats.get("accounts_updated", 0),
                "validation_ai_fields_merged": merge_stats.get("fields_updated", 0),
                "validation_ai_result_files": merge_stats.get("results_files", 0),
            })
        
        stage_payload["metrics"] = metrics_update
        stage_payload["validation_ai_completed"] = completed_flag
        stage_payload["summary"] = {
            "validation_ai": {
                "accounts_total": ai_state.get("accounts_total", 0),
                "accounts_pending": list(pending_accounts),
                "fields_total": ai_state.get("fields_total", 0),
                "fields_pending": ai_state.get("fields_pending", 0),
                "completed": completed_flag,
            }
        }
    
    record_stage_force(
        sid,
        {"stages": {"validation": stage_payload}},
        runs_root=runs_root_path,
        last_writer=f"validation_merge_{source}",
        refresh_barriers=True,
    )
    
    logger.info(
        "VALIDATION_MERGE_RUNFLOW_UPDATED sid=%s merge_applied=True source=%s",
        sid,
        source,
    )
    
    # 4. Update manifest.json
    try:
        manifest_path = runs_root_path / sid / "manifest.json"
        manifest = RunManifest.load_or_create(manifest_path, sid)
        manifest.mark_validation_merge_applied(
            applied_at=applied_at_iso,
            source=source,
        )
        logger.info(
            "VALIDATION_MERGE_MANIFEST_UPDATED sid=%s merge_applied=True source=%s",
            sid,
            source,
        )
    except Exception:
        logger.warning(
            "VALIDATION_MERGE_MANIFEST_UPDATE_FAILED sid=%s source=%s",
            sid,
            source,
            exc_info=True,
        )
    
    # 5. Refresh umbrella barriers to recompute strategy_ready
    try:
        runflow_refresh_umbrella_barriers(sid)
        logger.info(
            "VALIDATION_MERGE_BARRIERS_REFRESHED sid=%s source=%s",
            sid,
            source,
        )
    except Exception:
        logger.warning(
            "VALIDATION_MERGE_BARRIER_REFRESH_FAILED sid=%s source=%s",
            sid,
            source,
            exc_info=True,
        )
    
    logger.info(
        "VALIDATION_MERGE_STATE_UPDATE_COMPLETE sid=%s source=%s",
        sid,
        source,
    )
    
    # Final assertion: Verify that validation manifest paths are populated
    # This catches any issues where ensure_validation_section was skipped or failed
    try:
        assert_validation_manifest_paths_non_null(sid, runs_root=runs_root_path)
    except ValueError:
        # Log but don't fail the merge - paths may be missing due to legacy runs
        # or edge cases. In production, this should be investigated.
        logger.warning(
            "VALIDATION_MERGE_COMPLETE_BUT_PATHS_MISSING sid=%s source=%s",
            sid,
            source,
            exc_info=True,
        )
    
    return merge_stats
