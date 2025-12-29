"""Apply validation AI results to account summary.json files.

This module merges validation AI adjudication results into the requirement blocks
within each account's summary.json file.

Flow:
1. Load manifest and discover validation result files via index
2. For each result record (by account_id):
   - Load that account's summary.json
   - Find the matching requirement block (by reason_code + send_to_ai flag)
   - Enrich that requirement with AI fields: decision, rationale, citations, checks
   - Save updated summary.json back to disk
3. Return summary stats: accounts updated, results applied, unmatched results

Design:
- Idempotent: re-running for same SID overwrites AI fields in requirement blocks
- Disk-first: uses manifest API for safe concurrent access
- Defensive: logs warnings when results don't match any requirement
- Clear logging: VALIDATION_V2_APPLY_* markers for observability
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

from backend.pipeline.runs import load_manifest_from_disk
from backend.validation.index_schema import load_validation_index

log = logging.getLogger(__name__)


def _now_iso() -> str:
    """Return ISO-8601 timestamp in UTC with second precision."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_json(path: Path) -> dict[str, Any] | None:
    """Load JSON file from disk, return None on error."""
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8")
        return json.loads(text)
    except Exception:
        log.warning("VALIDATION_V2_APPLY_JSON_READ_FAILED path=%s", path, exc_info=True)
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> bool:
    """Write JSON file to disk, return True on success."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        path.write_text(text, encoding="utf-8")
        return True
    except Exception:
        log.warning("VALIDATION_V2_APPLY_JSON_WRITE_FAILED path=%s", path, exc_info=True)
        return False


def _find_matching_requirement(
    findings: Sequence[Mapping[str, Any]],
    result: Mapping[str, Any],
) -> tuple[int, MutableMapping[str, Any]] | None:
    """Find requirement block matching this AI result.
    
    Match criteria:
    - send_to_ai == true
    - reason_code == result.checks.mismatch_code
    - (optionally field name if available)
    
    Returns:
        Tuple of (index, requirement) if found, else None
    """
    checks = result.get("checks", {})
    result_mismatch_code = checks.get("mismatch_code")
    
    if not result_mismatch_code:
        return None
    
    # Optional: extract field name from result if available
    result_field = result.get("field")
    
    for idx, req in enumerate(findings):
        if not isinstance(req, MutableMapping):
            continue
        
        # Must be sent to AI
        if req.get("send_to_ai") is not True:
            continue
        
        # Must have matching reason_code
        req_reason_code = req.get("reason_code")
        if req_reason_code != result_mismatch_code:
            continue
        
        # Optional field match (if both are present)
        if result_field:
            req_field = req.get("field")
            if req_field and req_field != result_field:
                continue
        
        return idx, req
    
    return None


def _enrich_requirement_with_ai_result(
    requirement: MutableMapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    """Merge AI fields into requirement block (in-place).
    
    AI fields added:
    - ai_validation_decision
    - ai_validation_rationale
    - ai_validation_citations
    - ai_validation_checks
    - ai_validation_completed_at
    """
    requirement["ai_validation_decision"] = result.get("decision", "")
    requirement["ai_validation_rationale"] = result.get("rationale", "")
    requirement["ai_validation_citations"] = result.get("citations", [])
    requirement["ai_validation_checks"] = result.get("checks", {})
    requirement["ai_validation_completed_at"] = result.get("completed_at", _now_iso())


def _load_result_records_from_disk(
    results_dir: Path,
    index_records: Sequence[Any],
) -> list[dict[str, Any]]:
    """Load all validation AI result records from disk.
    
    Reads result.jsonl files discovered via index, parses JSONL lines.
    Returns list of result dicts with account_id, decision, rationale, etc.
    
    Args:
        results_dir: Directory containing result files
        index_records: List of ValidationPackRecord objects or dicts
    """
    all_results: list[dict[str, Any]] = []
    
    for record in index_records:
        # Handle both ValidationPackRecord objects and dicts
        if hasattr(record, "result_jsonl"):
            result_jsonl = record.result_jsonl
        elif isinstance(record, Mapping):
            result_jsonl = record.get("result_jsonl")
        else:
            continue
        
        if not result_jsonl:
            log.debug("VALIDATION_V2_APPLY_RECORD_NO_RESULT_PATH record=%s", record)
            continue
        
        # result_jsonl can be:
        # - relative to validation base dir: "results/acc_009.result.jsonl"
        # - just filename: "acc_009.result.jsonl"
        # - absolute path
        if Path(result_jsonl).is_absolute():
            result_path = Path(result_jsonl)
        elif "/" in result_jsonl or "\\" in result_jsonl:
            # Has directory component, resolve relative to validation base dir (parent of results_dir)
            validation_base = results_dir.parent
            result_path = validation_base / result_jsonl
        else:
            # Just filename, resolve in results_dir
            result_path = results_dir / result_jsonl
        
        log.debug("VALIDATION_V2_APPLY_CHECKING_RESULT path=%s exists=%s", result_path, result_path.exists())
        
        if not result_path.exists():
            log.warning("VALIDATION_V2_APPLY_RESULT_FILE_MISSING path=%s", result_path)
            continue
        
        try:
            text = result_path.read_text(encoding="utf-8")
            for line in text.strip().split("\n"):
                if not line.strip():
                    continue
                result = json.loads(line)
                all_results.append(result)
        except Exception:
            log.warning(
                "VALIDATION_V2_APPLY_RESULT_PARSE_FAILED path=%s",
                result_path,
                exc_info=True,
            )
    
    return all_results


def apply_validation_results_for_sid(
    sid: str,
    runs_root: Path | str,
) -> dict[str, Any]:
    """Apply validation AI results to account summary.json files.
    
    For each validation AI result:
    1. Load the account's summary.json
    2. Find matching requirement block (by reason_code + send_to_ai)
    3. Enrich requirement with AI fields (decision, rationale, citations, checks)
    4. Save updated summary.json
    
    Args:
        sid: Run session ID
        runs_root: Path to runs directory
        
    Returns:
        Summary dict with stats:
        - sid: session ID
        - accounts_total: number of unique accounts seen in results
        - accounts_updated: number of accounts where summaries were successfully updated
        - results_total: number of result records processed
        - results_applied: number of results successfully merged into requirements
        - results_unmatched: number of results that didn't match any requirement
        - completed_at: ISO timestamp
    """
    runs_root_path = Path(runs_root).resolve()
    run_dir = runs_root_path / sid
    
    apply_started_at = _now_iso()
    log.info(
        "VALIDATION_V2_APPLY_START sid=%s runs_root=%s started_at=%s",
        sid,
        runs_root_path,
        apply_started_at,
    )
    
    # Load manifest JSON directly to avoid serialization issues
    manifest_path = run_dir / "manifest.json"
    try:
        manifest_data = _load_json(manifest_path)
        if not manifest_data:
            raise ValueError("Manifest is empty or invalid")
    except Exception:
        log.exception("VALIDATION_V2_APPLY_MANIFEST_LOAD_FAILED sid=%s", sid)
        failed_summary = {
            "sid": sid,
            "error": "manifest_load_failed",
            "accounts_total": 0,
            "accounts_updated": 0,
            "results_total": 0,
            "results_applied": 0,
            "results_unmatched": 0,
            "completed_at": _now_iso(),
            "results_apply_at": apply_started_at,
            "results_apply_done": _now_iso(),
            "results_apply_ok": False,
        }
        _persist_apply_stats_to_manifest(manifest_path, failed_summary)
        return failed_summary
    
    # Extract validation paths from manifest
    artifacts = manifest_data.get("artifacts", {})
    ai_section = artifacts.get("ai", {})
    packs_section = ai_section.get("packs", {})
    validation_section = packs_section.get("validation", {})
    
    validation_index_rel = validation_section.get("index")
    validation_results_dir_rel = validation_section.get("results_dir")
    
    if not validation_index_rel:
        log.warning("VALIDATION_V2_APPLY_NO_INDEX_PATH sid=%s", sid)
        validation_index_path = run_dir / "ai_packs" / "validation" / "index.json"
    else:
        validation_index_path = run_dir / validation_index_rel
    
    if not validation_results_dir_rel:
        log.warning("VALIDATION_V2_APPLY_NO_RESULTS_DIR sid=%s", sid)
        validation_results_dir = run_dir / "ai_packs" / "validation" / "results"
    else:
        validation_results_dir = run_dir / validation_results_dir_rel
    
    # Load validation index
    try:
        validation_index = load_validation_index(validation_index_path)
    except Exception:
        log.exception(
            "VALIDATION_V2_APPLY_INDEX_LOAD_FAILED sid=%s path=%s",
            sid,
            validation_index_path,
        )
        failed_summary = {
            "sid": sid,
            "error": "index_load_failed",
            "accounts_total": 0,
            "accounts_updated": 0,
            "results_total": 0,
            "results_applied": 0,
            "results_unmatched": 0,
            "completed_at": _now_iso(),
            "results_apply_at": apply_started_at,
            "results_apply_done": _now_iso(),
            "results_apply_ok": False,
        }
        _persist_apply_stats_to_manifest(manifest_path, failed_summary)
        return failed_summary
    
    index_records = validation_index.packs if hasattr(validation_index, "packs") else []
    
    # Load all result records from disk
    log.debug(
        "VALIDATION_V2_APPLY_LOADING_RESULTS sid=%s results_dir=%s index_records=%d",
        sid,
        validation_results_dir,
        len(index_records),
    )
    all_results = _load_result_records_from_disk(validation_results_dir, index_records)
    
    log.info(
        "VALIDATION_V2_APPLY_RESULTS_LOADED sid=%s count=%d",
        sid,
        len(all_results),
    )
    
    if not all_results:
        log.warning("VALIDATION_V2_APPLY_NO_RESULTS sid=%s index_records=%d", sid, len(index_records))
        empty_summary = {
            "sid": sid,
            "accounts_total": 0,
            "accounts_updated": 0,
            "results_total": 0,
            "results_applied": 0,
            "results_unmatched": 0,
            "completed_at": _now_iso(),
            "results_apply_at": apply_started_at,
            "results_apply_done": _now_iso(),
            "results_apply_ok": False,
        }
        _persist_apply_stats_to_manifest(manifest_path, empty_summary)
        return empty_summary
    
    # Discover cases/accounts directory from manifest
    cases_section = artifacts.get("cases", {})
    accounts_section = cases_section.get("accounts", {})
    
    # Group results by account_id
    results_by_account: dict[int, list[dict[str, Any]]] = {}
    for result in all_results:
        account_id = result.get("account_id")
        if account_id is None:
            log.warning("VALIDATION_V2_APPLY_RESULT_NO_ACCOUNT_ID result=%s", result)
            continue
        
        account_id_int = int(account_id)
        if account_id_int not in results_by_account:
            results_by_account[account_id_int] = []
        results_by_account[account_id_int].append(result)
    
    # Process each account
    accounts_total = len(results_by_account)
    accounts_updated = 0
    results_applied = 0
    results_unmatched = 0
    
    for account_id, results in results_by_account.items():
        account_key = str(account_id)
        
        # Find account's summary.json path from manifest
        account_info = accounts_section.get(account_key, {})
        summary_rel_path = account_info.get("summary")
        
        if not summary_rel_path:
            # Fallback: construct standard path
            summary_path = run_dir / "cases" / "accounts" / account_key / "summary.json"
            log.debug(
                "VALIDATION_V2_APPLY_ACCOUNT_PATH_FALLBACK account=%s path=%s",
                account_key,
                summary_path,
            )
        else:
            summary_path = run_dir / summary_rel_path
        
        # Load summary.json
        summary = _load_json(summary_path)
        if not summary:
            log.warning(
                "VALIDATION_V2_APPLY_SUMMARY_LOAD_FAILED account=%s path=%s",
                account_key,
                summary_path,
            )
            results_unmatched += len(results)
            continue
        
        # Find validation_requirements block
        validation_block = summary.get("validation_requirements")
        if not isinstance(validation_block, dict):
            log.warning(
                "VALIDATION_V2_APPLY_NO_VALIDATION_BLOCK account=%s",
                account_key,
            )
            results_unmatched += len(results)
            continue
        
        findings = validation_block.get("findings")
        if not isinstance(findings, list):
            log.warning(
                "VALIDATION_V2_APPLY_NO_FINDINGS account=%s",
                account_key,
            )
            results_unmatched += len(results)
            continue
        
        # Match each result to a requirement and enrich
        account_updated = False
        for result in results:
            match = _find_matching_requirement(findings, result)
            
            if match is None:
                log.warning(
                    "VALIDATION_V2_APPLY_NO_MATCH sid=%s account=%s mismatch_code=%s",
                    sid,
                    account_key,
                    result.get("checks", {}).get("mismatch_code"),
                )
                results_unmatched += 1
                continue
            
            idx, requirement = match
            _enrich_requirement_with_ai_result(requirement, result)
            
            log.debug(
                "VALIDATION_V2_APPLY_MATCH sid=%s account=%s field=%s reason_code=%s decision=%s",
                sid,
                account_key,
                requirement.get("field"),
                requirement.get("reason_code"),
                result.get("decision"),
            )
            
            results_applied += 1
            account_updated = True
        
        # Save updated summary.json
        if account_updated:
            if _write_json(summary_path, summary):
                accounts_updated += 1
                log.info(
                    "VALIDATION_V2_APPLY_ACCOUNT_UPDATED sid=%s account=%s results=%d",
                    sid,
                    account_key,
                    len(results),
                )
            else:
                log.error(
                    "VALIDATION_V2_APPLY_ACCOUNT_WRITE_FAILED sid=%s account=%s",
                    sid,
                    account_key,
                )
    
    # After processing accounts, update validation index pack statuses to 'completed'
    # Use dataclasses.replace() to avoid FrozenInstanceError
    try:
        from dataclasses import replace as dataclass_replace
        
        applied_accounts_set = set(results_by_account.keys())
        updated_packs = []
        index_updated = 0
        
        for record in getattr(validation_index, "packs", []):
            try:
                if hasattr(record, "account_id") and record.account_id in applied_accounts_set:
                    current_status = getattr(record, "status", None)
                    if current_status != "completed":
                        # Create new record with updated status (immutable pattern)
                        updated_record = dataclass_replace(record, status="completed")
                        updated_packs.append(updated_record)
                        index_updated += 1
                    else:
                        updated_packs.append(record)
                else:
                    updated_packs.append(record)
            except Exception:
                log.warning(
                    "VALIDATION_V2_INDEX_RECORD_UPDATE_FAILED sid=%s account=%s",
                    sid,
                    getattr(record, "account_id", None),
                    exc_info=True,
                )
                # Keep original record on error
                updated_packs.append(record)
        
        if index_updated > 0:
            try:
                # Rebuild ValidationIndex with updated packs
                updated_index = dataclass_replace(validation_index, packs=updated_packs)
                updated_index.write()
                log.info(
                    "VALIDATION_V2_INDEX_PACKS_COMPLETED sid=%s packs_completed=%s packs_total=%s",
                    sid,
                    index_updated,
                    len(updated_packs),
                )
            except Exception:
                log.warning(
                    "VALIDATION_V2_INDEX_WRITE_FAILED sid=%s path=%s",
                    sid,
                    getattr(validation_index, "index_path", None),
                    exc_info=True,
                )
    except Exception:
        log.warning("VALIDATION_V2_INDEX_UPDATE_BLOCK_FAILED sid=%s", sid, exc_info=True)

    # Determine success
    success_flag = bool(results_applied >= 0 and results_unmatched >= 0)

    summary_stats = {
        "sid": sid,
        "accounts_total": accounts_total,
        "accounts_updated": accounts_updated,
        "results_total": len(all_results),
        "results_applied": results_applied,
        "results_unmatched": results_unmatched,
        "completed_at": _now_iso(),
        "results_apply_at": apply_started_at,
        "results_apply_done": True,
        # Base success flag used by canonical helper; counts guard applied in helper
        "results_apply_ok": bool(results_applied > 0),
        "applied": bool(results_applied > 0),
        # Do not compute canonical here; helper will set manifest flag consistently
        "validation_ai_applied": bool(results_applied > 0),
    }
    
    _persist_apply_stats_to_manifest(manifest_path, summary_stats)

    # After manifest is updated, refresh runflow so snapshot reflects V2 apply
    try:
        from backend.runflow.decider import refresh_validation_stage_from_index
        refresh_validation_stage_from_index(sid, runs_root=runs_root_path)
        # Note: refresh_validation_stage_from_index logs VALIDATION_STAGE_PROMOTED when promotion succeeds
        # Only log our V2-specific success marker if index was actually updated
        if index_updated > 0:
            log.info(
                "VALIDATION_V2_RUNFLOW_REFRESHED sid=%s results_total=%s results_applied=%s validation_ai_applied=%s",
                sid,
                summary_stats.get("results_total"),
                summary_stats.get("results_applied"),
                summary_stats.get("validation_ai_applied"),
            )
    except Exception:
        log.warning("VALIDATION_V2_RUNFLOW_REFRESH_FAILED sid=%s", sid, exc_info=True)

    log.info(
        "VALIDATION_V2_APPLY_SUMMARY sid=%s accounts_total=%d accounts_updated=%d "
        "results_total=%d results_applied=%d results_unmatched=%d apply_ok=%s",
        sid,
        accounts_total,
        accounts_updated,
        len(all_results),
        results_applied,
        results_unmatched,
        summary_stats.get("results_apply_ok"),
    )

    return summary_stats


def _set_canonical_validation_apply_status(validation_status: dict, stats: Mapping[str, Any]) -> None:
    """Set canonical V2 apply flags in-place on validation_status using stats.

    Canonical rule:
    - Base apply_ok = results_apply_ok or applied
    - Canonical applied starts with apply_ok
    - If results_total/results_applied are present, require:
        results_applied >= results_total and results_total > 0
    """
    results_total = stats.get("results_total")
    results_applied = stats.get("results_applied")
    apply_ok = bool(stats.get("results_apply_ok") or stats.get("applied"))

    canonical = bool(apply_ok)
    try:
        if results_total is not None and results_applied is not None:
            # Accept ints and numeric-like values
            try:
                rt = int(results_total)  # type: ignore[arg-type]
            except Exception:
                rt = None
            try:
                ra = int(results_applied)  # type: ignore[arg-type]
            except Exception:
                ra = None
            if isinstance(rt, int) and isinstance(ra, int):
                canonical = canonical and (ra >= rt) and (rt > 0)
    except Exception:
        # Defensive: keep base apply_ok if counts are malformed
        canonical = bool(apply_ok)

    validation_status["validation_ai_applied"] = bool(canonical)
    # Also persist helpers for back-compat (applied kept for transitional consumers)
    validation_status["results_apply_ok"] = bool(apply_ok)
    validation_status["applied"] = bool(apply_ok)

    # Strip legacy merge fields if they linger
    for legacy_key in ("merge_results_applied", "merge_results_applied_at", "merge_results"):
        if legacy_key in validation_status:
            validation_status.pop(legacy_key, None)


def _persist_apply_stats_to_manifest(manifest_path: Path, stats: Mapping[str, Any]) -> None:
    """Persist apply stats into manifest.ai.status.validation.* fields.

    Adds/updates the following keys under ai.status.validation:
      - results_total (int)
      - results_applied (int)
      - results_unmatched (int)
      - results_apply_at (timestamp)
      - results_apply_done (timestamp)
      - results_apply_ok (bool)

    Tolerates malformed or missing manifest gracefully.
    """
    try:
        data = _load_json(manifest_path)
        if not isinstance(data, Mapping):
            return
        ai = data.setdefault("ai", {}) if isinstance(data.get("ai"), Mapping) else data.setdefault("ai", {})
        status = ai.setdefault("status", {}) if isinstance(ai.get("status"), Mapping) else ai.setdefault("status", {})
        validation = status.setdefault("validation", {}) if isinstance(status.get("validation"), Mapping) else status.setdefault("validation", {})

        # Write counts
        for k in (
            "results_total",
            "results_applied",
            "results_unmatched",
        ):
            v = stats.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                validation[k] = int(v)
        # Timestamps and boolean done flag
        at_v = stats.get("results_apply_at")
        if isinstance(at_v, str) and at_v.strip():
            validation["results_apply_at"] = at_v.strip()
        validation["results_apply_done"] = bool(stats.get("results_apply_done") is True)
        # Canonical success flags (single helper for all apply-related writes)
        _set_canonical_validation_apply_status(validation, stats)

        # Remove duplicated artifacts.ai.status.validation if present
        artifacts = data.get("artifacts")
        if isinstance(artifacts, dict):
            ai_artifacts = artifacts.get("ai")
            if isinstance(ai_artifacts, dict):
                art_status = ai_artifacts.get("status")
                if isinstance(art_status, dict) and "validation" in art_status:
                    art_status.pop("validation", None)

        try:
            manifest_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            log.warning(
                "VALIDATION_V2_MANIFEST_APPLY_WRITE_FAILED path=%s", manifest_path, exc_info=True
            )
    except Exception:
        log.warning(
            "VALIDATION_V2_MANIFEST_APPLY_PERSIST_FAILED path=%s", manifest_path, exc_info=True
        )
