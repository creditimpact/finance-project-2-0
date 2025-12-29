"""Helpers for ingesting AI validation responses."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.ai.validation_index import ValidationPackIndexWriter
from backend.telemetry.metrics import emit_counter
from backend.core.ai.paths import (
    ensure_validation_paths,
    validation_pack_filename_for_account,
    validation_result_json_filename_for_account,
    validation_result_jsonl_filename_for_account,
    validation_result_summary_filename_for_account,
    validation_write_json_enabled,
)
from backend.core.runflow import runflow_barriers_refresh
from backend.runflow.decider import (
    reconcile_umbrella_barriers,
    refresh_validation_stage_from_index,
)
from backend.runflow.manifest import update_manifest_ai_stage_result
from backend.validation.io import write_json, write_jsonl
from backend.core.ai.eligibility_policy import (
    canonicalize_history,
    canonicalize_scalar,
)
from backend.core.ai.report_compare import compute_reason_flags, classify_reporting_pattern
from backend.core.logic.validation_field_sets import CONDITIONAL_FIELDS


log = logging.getLogger(__name__)

_BUREAUS: tuple[str, ...] = ("transunion", "experian", "equifax")


def _reasons_enabled() -> bool:
    raw = os.getenv("VALIDATION_REASON_ENABLED")
    if raw is None:
        return False

    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return False


def _clone_jsonish(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _clone_jsonish(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_clone_jsonish(entry) for entry in value]
    return value


def _emit_reason_observability(
    sid: str | None,
    account_id: Any,
    field_name: str,
    pattern: str,
    flags: Mapping[str, Any],
) -> None:
    if not _reasons_enabled():
        return

    missing = bool(flags.get("missing", False))
    mismatch = bool(flags.get("mismatch", False))
    eligible = bool(flags.get("eligible", False))
    ai_needed = field_name in CONDITIONAL_FIELDS and eligible

    metric_pattern = pattern if isinstance(pattern, str) and pattern else "unknown"

    log.info(
        "VALIDATION_ESCALATION_REASON sid=%s account_id=%s field=%s pattern=%s "
        "missing=%s mismatch=%s eligible=%s",
        sid,
        account_id,
        field_name,
        metric_pattern,
        missing,
        mismatch,
        eligible,
    )

    emit_counter(f"validation.pattern.{metric_pattern}")
    emit_counter(f"validation.eligible.{str(eligible).lower()}")
    emit_counter(f"validation.ai_needed.{str(ai_needed).lower()}")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        return Path("runs").resolve()
    return Path(runs_root).resolve()


def _index_writer(
    sid: str, runs_root: Path, paths: "ValidationPaths | None" = None
) -> ValidationPackIndexWriter:
    validation_paths = paths or ensure_validation_paths(runs_root, sid, create=True)
    return ValidationPackIndexWriter(
        sid=sid,
        index_path=validation_paths.index_file,
        packs_dir=validation_paths.packs_dir,
        results_dir=validation_paths.results_dir,
    )


def mark_validation_pack_sent(
    sid: str,
    account_id: int | str,
    *,
    runs_root: Path | str | None = None,
    request_lines: int | None = None,
    model: str | None = None,
) -> dict[str, object] | None:
    """Mark the pack for ``account_id`` as sent in the validation index."""

    runs_root_path = _resolve_runs_root(runs_root)
    validation_paths = ensure_validation_paths(runs_root_path, sid, create=True)
    pack_filename = validation_pack_filename_for_account(account_id)
    pack_path = validation_paths.packs_dir / pack_filename
    writer = _index_writer(sid, runs_root_path, validation_paths)
    return writer.mark_sent(
        pack_path,
        request_lines=request_lines,
        model=model,
    )


def _normalize_result_payload(
    sid: str,
    account_id: int | str,
    payload: Mapping[str, Any],
    *,
    status: str,
    request_lines: int | None,
    model: str | None,
    error: str | None,
    completed_at: str | None,
) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["sid"] = sid
    try:
        normalized["account_id"] = int(account_id)
    except (TypeError, ValueError):
        normalized["account_id"] = account_id

    if request_lines is not None:
        try:
            normalized["request_lines"] = int(request_lines)
        except (TypeError, ValueError):
            normalized.pop("request_lines", None)

    if model is not None:
        normalized["model"] = str(model)
    elif "model" in normalized and normalized["model"] is not None:
        normalized["model"] = str(normalized["model"])

    normalized["status"] = status

    if status == "error" and error:
        normalized.setdefault("error", str(error))

    timestamp = completed_at or normalized.get("completed_at")
    if not isinstance(timestamp, str) or not timestamp.strip():
        normalized["completed_at"] = _utc_now()
    else:
        normalized["completed_at"] = timestamp

    return normalized


def _coerce_account_int(account_id: int | str) -> int | None:
    try:
        return int(account_id)
    except (TypeError, ValueError):
        try:
            return int(str(account_id).strip())
        except (TypeError, ValueError):
            return None


def _load_pack_lookup(pack_path: Path) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    try:
        raw_lines = pack_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return lookup
    except OSError:
        return lookup

    for line in raw_lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue

        for key_name in ("id", "field_key", "field"):
            key_value = payload.get(key_name)
            if isinstance(key_value, str) and key_value.strip():
                lookup.setdefault(key_value.strip(), payload)

    return lookup


def _default_reason_payload() -> dict[str, Any]:
    return {
        "schema": 1,
        "pattern": "unknown",
        "missing": False,
        "mismatch": False,
        "both": False,
        "eligible": False,
        "coverage": {
            "missing_bureaus": [],
            "present_bureaus": [],
        },
        "values": {},
    }


def _build_reason_from_pack(field_name: str, pack_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    existing = pack_payload.get("reason")
    if isinstance(existing, Mapping):
        cloned = _clone_jsonish(existing)
        cloned.setdefault("schema", 1)
        return cloned

    bureaus = pack_payload.get("bureaus")
    if not isinstance(bureaus, Mapping):
        return _default_reason_payload()

    raw_values: dict[str, Any] = {}
    for bureau in _BUREAUS:
        bureau_payload = bureaus.get(bureau)
        if isinstance(bureau_payload, Mapping):
            raw_values[bureau] = bureau_payload.get("raw")
        else:
            raw_values[bureau] = None

    try:
        pattern = classify_reporting_pattern(raw_values)
    except Exception:  # pragma: no cover - defensive
        log.exception(
            "VALIDATION_RESULT_REASON_CLASSIFY_FAILED field=%s", field_name
        )
        pattern = "unknown"

    if field_name in {"two_year_payment_history", "seven_year_history"}:
        canonicalizer = canonicalize_history
    else:
        canonicalizer = canonicalize_scalar

    canonical_values: dict[str, Any] = {}
    missing_bureaus: list[str] = []
    present_bureaus: list[str] = []
    for bureau in _BUREAUS:
        try:
            canonical = canonicalizer(raw_values.get(bureau))
        except Exception:  # pragma: no cover - defensive
            log.exception(
                "VALIDATION_RESULT_REASON_CANONICALIZE_FAILED field=%s bureau=%s",
                field_name,
                bureau,
            )
            canonical = None
        canonical_values[bureau] = canonical
        if canonical is None:
            missing_bureaus.append(bureau)
        else:
            present_bureaus.append(bureau)

    flags = compute_reason_flags(field_name, pattern, match_matrix={})

    _emit_reason_observability(
        str(pack_payload.get("sid") or "") or None,
        pack_payload.get("account_id"),
        field_name,
        pattern,
        flags,
    )

    return {
        "schema": 1,
        "pattern": pattern,
        "missing": flags.get("missing", False),
        "mismatch": flags.get("mismatch", False),
        "both": flags.get("both", False),
        "eligible": flags.get("eligible", False),
        "coverage": {
            "missing_bureaus": missing_bureaus,
            "present_bureaus": present_bureaus,
        },
        "values": canonical_values,
    }


def _normalize_decision(decision: Any) -> str:
    value = str(decision or "").strip().lower()
    if value in {"strong", "supportive", "neutral", "no_case"}:
        return value
    if value in {"no_claim", "no_claims"}:
        return "no_case"
    if value in {"", "unknown"}:
        return "no_case"
    return "no_case"


def _normalize_citations(raw: Any) -> list[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    citations: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            citations.append(item.strip())
    return citations


def _normalize_labels(raw: Any) -> list[str]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []
    labels: list[str] = []
    for item in raw:
        if isinstance(item, str):
            text = item.strip()
            if text:
                labels.append(text)
    return labels


def _normalize_confidence(raw: Any) -> float | None:
    try:
        if raw is None:
            return None
        confidence = float(raw)
    except (TypeError, ValueError):
        return None
    if confidence < 0 or confidence > 1:
        return None
    return round(confidence, 6)


def _fallback_result_id(account_int: int | None, field_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", field_name.lower()).strip("_")
    if not slug:
        slug = "field"
    if account_int is None:
        return f"acc__{slug}"
    return f"acc_{account_int:03d}__{slug}"


def _collect_result_entries(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for key in ("results", "decision_per_field"):
        raw = payload.get(key)
        if isinstance(raw, Sequence):
            for entry in raw:
                if isinstance(entry, Mapping):
                    yield entry


def _build_result_lines(
    account_id: int | str,
    payload: Mapping[str, Any],
    pack_lookup: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    account_int = _coerce_account_int(account_id)
    result_lines: list[dict[str, Any]] = []

    for entry in _collect_result_entries(payload):
        candidate_keys: list[str] = []
        for key_name in ("id", "field_key", "field"):
            value = entry.get(key_name)
            if isinstance(value, str) and value.strip():
                candidate_keys.append(value.strip())

        pack_payload: Mapping[str, Any] | None = None
        for key in candidate_keys:
            pack_payload = pack_lookup.get(key)
            if pack_payload:
                break

        if pack_payload is not None:
            field_name = (
                str(pack_payload.get("field") or "").strip() or candidate_keys[-1]
            )
            line_id = str(pack_payload.get("id") or "").strip()
        else:
            field_name = candidate_keys[-1] if candidate_keys else ""
            line_id = ""

        if not field_name:
            continue

        if not line_id:
            line_id = _fallback_result_id(account_int, field_name)

        rationale = entry.get("rationale")
        if not isinstance(rationale, str):
            rationale = ""

        reason_payload = None
        if _reasons_enabled() and isinstance(pack_payload, Mapping):
            try:
                reason_payload = _build_reason_from_pack(field_name, pack_payload)
            except Exception:  # pragma: no cover - defensive
                log.exception(
                    "VALIDATION_RESULT_REASON_BUILD_FAILED field=%s", field_name
                )
                reason_payload = _default_reason_payload()

        result_line = {
            "id": line_id,
            "account_id": account_int if account_int is not None else account_id,
            "field": field_name,
            "decision": _normalize_decision(entry.get("decision")),
            "rationale": rationale,
            "citations": _normalize_citations(entry.get("citations")),
        }
        result_line["legacy_decision"] = (
            "strong" if result_line["decision"] == "strong" else "no_case"
        )
        confidence_value = _normalize_confidence(entry.get("confidence"))
        if confidence_value is not None:
            result_line["confidence"] = confidence_value
        labels_value = _normalize_labels(entry.get("labels"))
        if labels_value:
            result_line["labels"] = labels_value

        if reason_payload is not None:
            result_line["reason"] = reason_payload

        result_lines.append(result_line)

    return result_lines


def store_validation_result(
    sid: str,
    account_id: int | str,
    response_payload: Mapping[str, Any],
    *,
    runs_root: Path | str | None = None,
    status: str = "done",
    error: str | None = None,
    request_lines: int | None = None,
    model: str | None = None,
    completed_at: str | None = None,
) -> Path:
    """Persist the AI response for ``account_id`` and update the index."""

    normalized_status = str(status).strip().lower()
    if normalized_status not in {"done", "error"}:
        raise ValueError("status must be 'done' or 'error'")

    runs_root_path = _resolve_runs_root(runs_root)
    validation_paths = ensure_validation_paths(runs_root_path, sid, create=True)
    summary_filename = validation_result_summary_filename_for_account(account_id)
    summary_path = validation_paths.results_dir / summary_filename

    jsonl_filename = validation_result_jsonl_filename_for_account(account_id)
    jsonl_path = validation_paths.results_dir / jsonl_filename
    json_path = (
        validation_paths.results_dir
        / validation_result_json_filename_for_account(account_id)
    )

    pack_filename = validation_pack_filename_for_account(account_id)
    pack_path = validation_paths.packs_dir / pack_filename
    pack_lookup = _load_pack_lookup(pack_path)

    result_lines = _build_result_lines(account_id, response_payload, pack_lookup)

    normalized_payload = _normalize_result_payload(
        sid,
        account_id,
        response_payload,
        status=normalized_status,
        request_lines=request_lines,
        model=model,
        error=error,
        completed_at=completed_at,
    )

    normalized_payload["results"] = result_lines

    missing_decisions = normalized_status == "done" and not result_lines
    if missing_decisions:
        log.warning(
            "VALIDATION_RESULTS_EMPTY sid=%s account_id=%s",
            sid,
            account_id,
        )
        normalized_status = "error"
        if not error:
            error = "missing_result_lines"
        normalized_payload["status"] = normalized_status
        if error:
            normalized_payload.setdefault("error", error)

    decisions: list[dict[str, Any]] = []
    for line in result_lines:
        field_id = str(line.get("id") or "").strip()
        if not field_id:
            field_id = str(line.get("field") or "").strip()
        decision_value = _clone_jsonish(line.get("decision"))
        rationale_value = line.get("rationale")
        citations_value = line.get("citations")

        decision_entry: dict[str, Any] = {
            "field_id": field_id,
            "decision": decision_value,
            "rationale": rationale_value if isinstance(rationale_value, str) else "",
            "citations": (
                list(citations_value)
                if isinstance(citations_value, Sequence)
                and not isinstance(citations_value, (str, bytes, bytearray))
                else []
            ),
        }
        decisions.append(decision_entry)

    request_lines_value = normalized_payload.get("request_lines")
    try:
        request_lines_count = (
            int(request_lines_value)
            if request_lines_value is not None
            else len(decisions)
        )
    except (TypeError, ValueError):
        request_lines_count = len(decisions)

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if result_lines:
        write_jsonl(jsonl_path, result_lines)
    else:
        jsonl_path.write_text("", encoding="utf-8")

    if validation_write_json_enabled():
        json_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(json_path, normalized_payload)
        summary_path = json_path
    else:
        summary_path = jsonl_path

    log.info(
        "VALIDATION_RESULTS_WRITTEN sid=%s account_id=%s summary=%s decisions=%s status=%s",
        sid,
        account_id,
        str(summary_path),
        len(result_lines),
        normalized_status,
    )

    writer = _index_writer(sid, runs_root_path, validation_paths)
    index_status = "completed" if normalized_status == "done" else "failed"
    writer.record_result(
        pack_path,
        status=index_status,
        error=error,
        request_lines=request_lines,
        model=normalized_payload.get("model"),
        completed_at=normalized_payload.get("completed_at"),
        result_path=summary_path if index_status == "completed" else None,
        line_count=request_lines_count,
    )

    try:
        runflow_barriers_refresh(sid)
    except Exception:  # pragma: no cover - defensive logging
        log.warning("VALIDATION_BARRIERS_REFRESH_FAILED sid=%s", sid, exc_info=True)

    try:
        refresh_validation_stage_from_index(sid, runs_root=runs_root_path)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "VALIDATION_STAGE_REFRESH_FAILED sid=%s", sid, exc_info=True
        )

    try:
        reconcile_umbrella_barriers(sid, runs_root=runs_root_path)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "VALIDATION_BARRIERS_RECONCILE_FAILED sid=%s", sid, exc_info=True
        )

    if normalized_status == "done" and result_lines:
        try:
            update_manifest_ai_stage_result(
                sid,
                "validation",
                runs_root=runs_root_path,
            )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "VALIDATION_MANIFEST_STAGE_RESULT_UPDATE_FAILED sid=%s account_id=%s path=%s",
                sid,
                account_id,
                str(summary_path),
                exc_info=True,
            )
    else:
        log.info(
            "VALIDATION_MANIFEST_STAGE_RESULT_SKIPPED sid=%s account_id=%s status=%s path=%s decisions=%s",
            sid,
            account_id,
            normalized_status,
            str(summary_path),
            len(result_lines),
        )

    return summary_path


__all__ = ["mark_validation_pack_sent", "store_validation_result"]

