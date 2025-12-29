"""
Clean validation AI sender inspired by note_style pattern.

Design notes (pattern study from note_style):
    1. note_style flow:
       build_packs → send_note_style_packs_for_sid → per-pack loop → client.chat_completion → 
       write result → update index → runflow refresh
    
    2. Key components reused:
       - get_ai_client() for OpenAI client
       - client.chat_completion(...) for sending
       - load manifest from disk for paths
       - ValidationIndex schema for pack discovery
       - refresh_validation_stage_from_index for runflow update
    
    3. Differences from note_style:
       - Validation packs use JSON schema-based response (like merge)
       - No note hashing or idempotency checks (simpler)
       - Results written as single .json or .jsonl per account
       - Direct sequential send (no complex retry orchestration)

This sender is designed for Phase 2 orchestrator mode only.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.core.services.ai_client import get_ai_client
from backend.pipeline.runs import load_manifest_from_disk, save_manifest_to_disk, RunManifest
from backend.validation.index_schema import load_validation_index, ValidationIndex
from backend.runflow.decider import refresh_validation_stage_from_index

log = logging.getLogger(__name__)


def _utc_now() -> str:
    """Return ISO 8601 timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_pack_lines(pack_path: Path) -> list[dict[str, Any]]:
    """Load JSONL pack file and return list of pack line dicts."""
    if not pack_path.exists() or not pack_path.is_file():
        return []
    
    lines: list[dict[str, Any]] = []
    try:
        text = pack_path.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("VALIDATION_V2_PACK_READ_FAILED path=%s error=%s", pack_path, exc)
        return []
    
    for line_num, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                lines.append(payload)
        except json.JSONDecodeError as exc:
            log.warning(
                "VALIDATION_V2_PACK_LINE_INVALID path=%s line=%d error=%s",
                pack_path,
                line_num,
                exc,
            )
    return lines


def _build_validation_messages(pack_line: dict[str, Any]) -> list[dict[str, Any]]:
    """Build OpenAI messages from a validation pack line."""
    prompt = pack_line.get("prompt")
    if not isinstance(prompt, dict):
        raise ValueError("Pack line missing 'prompt' dict")
    
    system = prompt.get("system")
    user = prompt.get("user")
    
    if not isinstance(system, str) or not isinstance(user, str):
        raise ValueError("Pack line prompt must have 'system' and 'user' strings")
    
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _send_pack_line_to_ai(
    client: Any,
    pack_line: dict[str, Any],
    *,
    model: str,
) -> dict[str, Any]:
    """
    Send a single validation pack line to OpenAI and return the parsed result.
    
    Returns a dict with keys: decision, rationale, citations, checks, etc.
    Raises on errors (caller logs and continues).
    """
    messages = _build_validation_messages(pack_line)
    
    # Validation uses JSON mode with structured schema
    response_format = {"type": "json_object"}
    
    start = time.perf_counter()
    try:
        response = client.chat_completion(
            messages=messages,
            model=model,
            temperature=0,
            response_format=response_format,
        )
    except Exception as exc:
        latency = time.perf_counter() - start
        log.error(
            "VALIDATION_V2_API_ERROR model=%s latency=%.3fs error=%s",
            model,
            latency,
            str(exc),
        )
        raise
    
    latency = time.perf_counter() - start
    
    # Extract content JSON from response
    parsed = response.get("content_json")
    if not isinstance(parsed, dict):
        raw_content = response.get("raw_content") or response.get("content")
        if isinstance(raw_content, str):
            try:
                parsed = json.loads(raw_content)
            except json.JSONDecodeError as exc:
                log.error(
                    "VALIDATION_V2_PARSE_FAILED model=%s latency=%.3fs error=%s",
                    model,
                    latency,
                    str(exc),
                )
                raise ValueError(f"Failed to parse AI response as JSON: {exc}")
        else:
            raise ValueError("No content_json or parsable content in AI response")
    
    # Validate required fields
    required = ["decision", "rationale", "citations", "checks"]
    for field in required:
        if field not in parsed:
            raise ValueError(f"AI response missing required field: {field}")
    
    log.debug(
        "VALIDATION_V2_API_SUCCESS model=%s latency=%.3fs decision=%s",
        model,
        latency,
        parsed.get("decision"),
    )
    
    return parsed


def _write_result_file(
    result_path: Path,
    account_id: str | int,
    pack_lines_results: list[dict[str, Any]],
    *,
    sid: str,
) -> None:
    """Write validation results to disk as JSONL (one result per line)."""
    result_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines_out: list[str] = []
    for result in pack_lines_results:
        # Enrich each result with metadata
        enriched = dict(result)
        enriched["sid"] = sid
        enriched["account_id"] = account_id
        enriched["completed_at"] = _utc_now()
        
        line = json.dumps(enriched, ensure_ascii=False, sort_keys=True)
        lines_out.append(line)
    
    content = "\n".join(lines_out) + "\n"
    try:
        result_path.write_text(content, encoding="utf-8")
    except OSError as exc:
        log.error(
            "VALIDATION_V2_RESULT_WRITE_FAILED path=%s error=%s",
            result_path,
            exc,
        )
        raise


def run_validation_send_for_sid_v2(
    sid: str,
    runs_root: Path,
    *,
    model: str | None = None,
) -> dict[str, Any]:
    """
    Send validation packs for a single SID using the new clean sender.
    
    Flow:
      1. Load manifest from disk to get validation paths
      2. Load validation index to discover packs
      3. For each pack record:
         - Load pack JSONL
         - Send each line to AI via client.chat_completion
         - Write results to results_dir
      4. Refresh runflow validation stage from index
      5. Update manifest.ai.status.validation
    
    Returns:
        dict with keys: sid, expected, sent, written, failed, errors
    
    Raises:
        RuntimeError if validation paths are missing or index unavailable
    """
    import os
    
    log.info("VALIDATION_V2_SEND_START sid=%s", sid)
    
    # Load manifest from disk (disk-first)
    manifest = load_manifest_from_disk(runs_root, sid)
    
    # Extract validation paths from manifest
    data: Mapping[str, Any] = manifest.data if isinstance(manifest.data, Mapping) else {}
    ai = data.get("ai") or {}
    packs = ai.get("packs") or {}
    validation_section = packs.get("validation") or {}
    
    packs_dir_str = validation_section.get("packs_dir") or validation_section.get("packs")
    results_dir_str = validation_section.get("results_dir") or validation_section.get("results")
    index_path_str = validation_section.get("index")
    
    if not packs_dir_str or not results_dir_str or not index_path_str:
        log.error(
            "VALIDATION_V2_PATHS_MISSING sid=%s packs_dir=%s results_dir=%s index=%s",
            sid,
            packs_dir_str,
            results_dir_str,
            index_path_str,
        )
        raise RuntimeError("Manifest missing validation paths (packs_dir/results_dir/index)")
    
    packs_dir = Path(packs_dir_str)
    results_dir = Path(results_dir_str)
    index_path = Path(index_path_str)
    
    # Load validation index
    if not index_path.exists():
        log.error("VALIDATION_V2_INDEX_MISSING sid=%s path=%s", sid, index_path)
        raise RuntimeError(f"Validation index missing: {index_path}")
    
    try:
        index = load_validation_index(index_path)
    except Exception as exc:
        log.error("VALIDATION_V2_INDEX_LOAD_FAILED sid=%s path=%s error=%s", sid, index_path, exc)
        raise RuntimeError(f"Failed to load validation index: {exc}")
    
    expected = len(index.packs)
    log.info("VALIDATION_V2_INDEX_LOADED sid=%s packs=%d", sid, expected)
    
    if expected == 0:
        log.info("VALIDATION_V2_NO_PACKS sid=%s", sid)
        return {
            "sid": sid,
            "expected": 0,
            "sent": 0,
            "written": 0,
            "failed": 0,
            "errors": [],
        }
    
    # Get AI client
    client = get_ai_client()
    
    # Determine model
    effective_model = model or os.getenv("VALIDATION_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o"
    
    log.info("VALIDATION_V2_CONFIG sid=%s model=%s expected_packs=%d", sid, effective_model, expected)
    
    # Per-pack send loop
    sent_count = 0
    written_count = 0
    failed_count = 0
    errors: list[dict[str, Any]] = []
    
    for record in index.packs:
        account_id = record.account_id
        
        # Resolve pack path
        try:
            pack_path = index.resolve_pack_path(record)
        except Exception as exc:
            log.warning(
                "VALIDATION_V2_PACK_RESOLVE_FAILED sid=%s account_id=%s error=%s",
                sid,
                account_id,
                exc,
            )
            failed_count += 1
            errors.append({"account_id": account_id, "phase": "pack_resolve", "error": str(exc)})
            continue
        
        # Check if result already exists (idempotency)
        try:
            result_path = index.resolve_result_jsonl_path(record)
        except Exception:
            # Fallback if resolve fails
            result_path = results_dir / f"val_acc_{account_id:03d}.result.jsonl"
        
        if result_path.exists() and result_path.stat().st_size > 0:
            log.info(
                "VALIDATION_V2_SKIP_EXISTING sid=%s account_id=%s result=%s",
                sid,
                account_id,
                result_path.name,
            )
            # Count as written (already complete)
            written_count += 1
            continue
        
        log.info(
            "VALIDATION_V2_SEND_START sid=%s account_id=%s pack=%s",
            sid,
            account_id,
            pack_path.name,
        )
        
        # Load pack lines
        pack_lines = _load_pack_lines(pack_path)
        if not pack_lines:
            log.warning(
                "VALIDATION_V2_PACK_EMPTY sid=%s account_id=%s pack=%s",
                sid,
                account_id,
                pack_path.name,
            )
            failed_count += 1
            errors.append({"account_id": account_id, "phase": "pack_load", "error": "empty_pack"})
            continue
        
        # Send each line to AI
        pack_results: list[dict[str, Any]] = []
        for line_num, pack_line in enumerate(pack_lines, start=1):
            try:
                ai_result = _send_pack_line_to_ai(client, pack_line, model=effective_model)
                pack_results.append(ai_result)
            except Exception as exc:
                log.error(
                    "VALIDATION_V2_SEND_LINE_FAILED sid=%s account_id=%s line=%d error=%s",
                    sid,
                    account_id,
                    line_num,
                    exc,
                )
                # Record error but continue with next line
                pack_results.append({
                    "decision": "no_case",
                    "rationale": f"AI send failed: {str(exc)}",
                    "citations": [],
                    "checks": {
                        "materiality": False,
                        "supports_consumer": False,
                        "doc_requirements_met": False,
                        "mismatch_code": "unknown",
                    },
                    "error": str(exc),
                })
        
        sent_count += 1
        
        # Write results to disk
        try:
            _write_result_file(result_path, account_id, pack_results, sid=sid)
            written_count += 1
            log.info(
                "VALIDATION_V2_SEND_DONE sid=%s account_id=%s result=%s lines=%d",
                sid,
                account_id,
                result_path.name,
                len(pack_results),
            )
        except Exception as exc:
            log.error(
                "VALIDATION_V2_WRITE_FAILED sid=%s account_id=%s error=%s",
                sid,
                account_id,
                exc,
            )
            failed_count += 1
            errors.append({"account_id": account_id, "phase": "result_write", "error": str(exc)})
    
    # Summary
    summary = {
        "sid": sid,
        "expected": expected,
        "sent": sent_count,
        "written": written_count,
        "failed": failed_count,
        "errors": errors,
    }
    
    log.info(
        "VALIDATION_V2_SUMMARY sid=%s expected=%d sent=%d written=%d failed=%d",
        sid,
        expected,
        sent_count,
        written_count,
        failed_count,
    )
    
    # Update index with result paths for records where results were written
    try:
        from backend.validation.index_schema import ValidationPackRecord
        
        # Reload index to get fresh copy
        index = load_validation_index(index_path)
        updated_records: list[ValidationPackRecord] = []
        changes_made = False
        
        for record in index.packs:
            # Check if result exists on disk
            try:
                result_path = index.resolve_result_jsonl_path(record)
                result_exists = result_path.exists() and result_path.stat().st_size > 0
            except Exception:
                result_exists = False
            
            # If result exists and record doesn't have result paths, update it
            if result_exists and not record.result_jsonl:
                # Build relative path for result_jsonl
                result_rel = f"results/{result_path.name}"
                
                updated_record = ValidationPackRecord(
                    account_id=record.account_id,
                    pack=record.pack,
                    result_jsonl=result_rel,
                    result_json=None,  # We only write JSONL
                    weak_fields=record.weak_fields,
                    lines=record.lines,
                    status="sent",  # Update status to reflect completion
                    source_hash=record.source_hash,
                    built_at=record.built_at,
                )
                updated_records.append(updated_record)
                changes_made = True
                log.debug(
                    "VALIDATION_V2_INDEX_UPDATE_RECORD account_id=%s result=%s",
                    record.account_id,
                    result_rel,
                )
            else:
                updated_records.append(record)
        
        if changes_made:
            # Write updated index
            from backend.validation.index_schema import ValidationIndex
            updated_index = ValidationIndex(
                index_path=index.index_path,
                sid=index.sid,
                root=index.root,
                packs_dir=index.packs_dir,
                results_dir=index.results_dir,
                packs=tuple(updated_records),
                schema_version=index.schema_version,
            )
            updated_index.write()
            log.info("VALIDATION_V2_INDEX_UPDATED sid=%s records=%d", sid, len(updated_records))
    except Exception as exc:
        log.warning("VALIDATION_V2_INDEX_UPDATE_FAILED sid=%s error=%s", sid, exc, exc_info=True)
    
    # Apply validation results to account summaries (V2)
    # This merges AI fields (decision, rationale, citations) into requirement blocks in summary.json
    apply_stats = {}
    apply_success = False
    if written_count > 0:
        try:
            from backend.validation.apply_results_v2 import apply_validation_results_for_sid
            apply_stats = apply_validation_results_for_sid(sid, runs_root)
            apply_success = (
                apply_stats.get("results_applied", 0) > 0 and
                apply_stats.get("error") is None
            )
            log.info(
                "VALIDATION_V2_APPLY_DONE sid=%s accounts_updated=%s results_applied=%s success=%s",
                sid,
                apply_stats.get("accounts_updated"),
                apply_stats.get("results_applied"),
                apply_success,
            )
        except Exception as exc:
            log.error("VALIDATION_V2_APPLY_FAILED sid=%s error=%s", sid, exc, exc_info=True)
            apply_stats = {"error": str(exc)}
            apply_success = False
    
    # Update manifest.ai.status.validation
    # Only mark as completed if results were written AND applied to summaries
    completed = (written_count == expected) and (failed_count == 0) and apply_success

    # New granular apply stats fields
    apply_total = apply_stats.get("results_total") if isinstance(apply_stats, dict) else None
    apply_applied = apply_stats.get("results_applied") if isinstance(apply_stats, dict) else None
    apply_unmatched = apply_stats.get("results_unmatched") if isinstance(apply_stats, dict) else None
    apply_at = apply_stats.get("results_apply_at") if isinstance(apply_stats, dict) else None
    apply_done = apply_stats.get("results_apply_done") if isinstance(apply_stats, dict) else None
    apply_ok = apply_stats.get("results_apply_ok") if isinstance(apply_stats, dict) else apply_success

    pending_flag = (written_count > 0) and not apply_success and failed_count == 0

    def _mutate_status(fm: RunManifest) -> None:
        status = fm.ensure_ai_stage_status("validation")
        status["built"] = True
        status["sent"] = True
        # counts
        if isinstance(apply_total, (int, float)) and not isinstance(apply_total, bool):
            status["results_total"] = int(apply_total)
        if isinstance(apply_applied, (int, float)) and not isinstance(apply_applied, bool):
            status["results_applied"] = int(apply_applied)
        if isinstance(apply_unmatched, (int, float)) and not isinstance(apply_unmatched, bool):
            status["results_unmatched"] = int(apply_unmatched)
        # timestamps + ok flag
        if isinstance(apply_at, str) and apply_at.strip():
            status["results_apply_at"] = apply_at.strip()
        if isinstance(apply_done, bool):
            status["results_apply_done"] = apply_done
        elif isinstance(apply_done, str) and apply_done.strip():
            # Back-compat if older apply provided timestamp; normalize to True
            status["results_apply_done"] = True
            status["results_apply_at"] = status.get("results_apply_at") or apply_done.strip()
        status["results_apply_ok"] = bool(apply_ok)
        # legacy boolean style (for any previous readers)
        status["validation_ai_applied"] = bool(apply_success)
        status["completed_at"] = _utc_now() if completed else None
        status["failed"] = not completed and failed_count > 0
        if completed:
            status["state"] = "completed"
        elif failed_count > 0:
            status["state"] = "error"
        elif pending_flag:
            status["state"] = "results_pending"
        else:
            status["state"] = "in_progress"
    
    try:
        save_manifest_to_disk(runs_root, sid, _mutate_status, caller="validation_sender_v2")
        log.info(
            "VALIDATION_V2_MANIFEST_UPDATED sid=%s completed=%s apply_success=%s",
            sid,
            completed,
            apply_success,
        )
    except Exception as exc:
        log.warning("VALIDATION_V2_MANIFEST_UPDATE_FAILED sid=%s error=%s", sid, exc, exc_info=True)

    # Refresh runflow AFTER V2 apply succeeded so snapshot reflects applied stats
    if apply_success:
        try:
            refresh_validation_stage_from_index(sid, runs_root=runs_root)
            log.info(
                "VALIDATION_V2_RUNFLOW_REFRESHED sid=%s results_total=%s results_applied=%s validation_ai_applied=%s",
                sid,
                apply_total,
                apply_applied,
                bool(apply_ok),
            )
        except Exception as exc:
            log.warning(
                "VALIDATION_V2_RUNFLOW_REFRESH_FAILED sid=%s error=%s", sid, exc, exc_info=True
            )
    
    # Add apply stats to summary for caller visibility
    summary["apply_stats"] = apply_stats
    summary["apply_success"] = apply_success
    
    return summary


__all__ = ["run_validation_send_for_sid_v2"]
