"""PII-safe logging for Stage A candidate tokens.

This module provides utilities to persist a minimal set of account fields for
offline analysis.  The logger is designed to be safe-by-default, append-only and
observable via telemetry.  Historical helpers ``CandidateTokenLogger`` and
``StageATraceLogger`` are retained for backwards compatibility.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.pipeline.runs import RunManifest

from backend.config import (
    CANDIDATE_LOG_FORMAT,
    CASESTORE_DIR,
    ENABLE_CANDIDATE_TOKEN_LOGGER,
)
from backend.core.logic.validation_field_sets import ALL_VALIDATION_FIELDS
from backend.core.case_store.telemetry import emit

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def candidate_tokens_path(session_id: str) -> str:
    """Return absolute candidate tokens path based on ``CANDIDATE_LOG_FORMAT``."""

    ext = "jsonl" if CANDIDATE_LOG_FORMAT == "jsonl" else "json"
    return os.path.join(CASESTORE_DIR, f"{session_id}.candidate_tokens.{ext}")


_ALLOWED_FIELDS: tuple[str, ...] = ALL_VALIDATION_FIELDS


# PII regexes ---------------------------------------------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\b(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b")
LONG_DIGITS_RE = re.compile(r"\b\d{8,}\b")
ADDRESS_RE = re.compile(
    r"\b(?:street|st\.?|ave|road|rd\.?|blvd|apt|suite)(?:\b|$)",
    re.IGNORECASE,
)


@dataclass
class MaskCounts:
    total: int = 0
    email: int = 0
    phone: int = 0
    ssn: int = 0
    address: int = 0
    account_number: int = 0


def _sanitize_str(value: str, counts: MaskCounts) -> str:
    def _email_sub(match: re.Match[str]) -> str:
        counts.email += 1
        counts.total += 1
        return "[redacted]"

    value = EMAIL_RE.sub(_email_sub, value)

    def _phone_sub(match: re.Match[str]) -> str:
        counts.phone += 1
        counts.total += 1
        return "[redacted]"

    value = PHONE_RE.sub(_phone_sub, value)

    def _ssn_sub(match: re.Match[str]) -> str:
        counts.ssn += 1
        counts.total += 1
        return "[redacted]"

    value = SSN_RE.sub(_ssn_sub, value)

    def _mask_long_digits(match: re.Match[str]) -> str:
        counts.account_number += 1
        counts.total += 1
        digits = match.group()
        return "****" + digits[-4:]

    value = LONG_DIGITS_RE.sub(_mask_long_digits, value)
    if ADDRESS_RE.search(value):
        counts.address += 1
        counts.total += 1
        return "[redacted]"
    return value


def _sanitize_value(val: Any, counts: MaskCounts) -> Any:
    if isinstance(val, str):
        return _sanitize_str(val, counts)
    if isinstance(val, list):
        return [_sanitize_value(v, counts) for v in val]
    if isinstance(val, dict):
        return {k: _sanitize_value(v, counts) for k, v in val.items()}
    return val


def sanitize_fields_for_tokens(
    fields: Dict[str, Any]
) -> Tuple[Dict[str, Any], MaskCounts]:
    """Deep-copy & sanitize fields for logging."""

    counts = MaskCounts()
    cleaned: Dict[str, Any] = {}
    for name in _ALLOWED_FIELDS:
        if name in fields and fields[name] is not None:
            cleaned[name] = _sanitize_value(fields[name], counts)
    return cleaned, counts


def log_stageA_candidates(
    session_id: str,
    account_id: str,
    bureau: str,
    phase: str,
    fields: Dict[str, Any],
    decision: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Write one record to the session's candidate tokens file."""

    if not ENABLE_CANDIDATE_TOKEN_LOGGER:
        return

    sanitized_fields, masked = sanitize_fields_for_tokens(fields)
    record = {
        "session_id": session_id,
        "account_id": account_id,
        "bureau": bureau,
        "phase": phase,
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fields": sanitized_fields,
        "decision": decision,
    }
    if meta:
        record["meta"] = meta

    path = candidate_tokens_path(session_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = json.dumps(record, ensure_ascii=False)
    bytes_written = len(data.encode("utf-8"))

    try:
        if CANDIDATE_LOG_FORMAT == "jsonl":
            with open(path, "a", encoding="utf-8") as f:
                f.write(data + "\n")
                f.flush()
                os.fsync(f.fileno())
            bytes_written += 1  # newline
            records_count = 1
        else:
            records: List[Dict[str, Any]] = []
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, list):
                            records = loaded
                except Exception:
                    records = []
            records.append(record)
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path), text=True)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            records_count = len(records)

        try:
            emit(
                "candidate_tokens_write",
                session_id=session_id,
                account_id=account_id,
                phase=phase,
                bytes_written=bytes_written,
                records=records_count,
                fields_masked_total=masked.total,
                fields_masked_email=masked.email,
                fields_masked_phone=masked.phone,
                fields_masked_ssn=masked.ssn,
                fields_masked_address=masked.address,
                fields_masked_account=masked.account_number,
            )
        except Exception:
            pass
    except Exception as e:
        try:
            emit(
                "candidate_tokens_error",
                session_id=session_id,
                account_id=account_id,
                phase=phase,
                error=e.__class__.__name__,
            )
        except Exception:
            pass
        raise


# ---------------------------------------------------------------------------
# Legacy helpers retained for backwards compatibility
# ---------------------------------------------------------------------------


_FIELDS = [
    "balance_owed",
    "account_rating",
    "account_description",
    "dispute_status",
    "creditor_type",
    "account_status",
    "payment_status",
    "creditor_remarks",
    "account_type",
    "credit_limit",
    "late_payments",
    "past_due_amount",
]


def _redact(value: str) -> str:
    """Mask digits in string values to avoid logging PII."""
    if value.isdigit():
        return value
    return re.sub(r"\d", "X", value)


class CandidateTokenLogger:
    """Accumulates raw field values and persists them to disk."""

    def __init__(self) -> None:
        self._tokens: Dict[str, Set[str]] = {name: set() for name in _FIELDS}

    def collect(self, account: Dict[str, Any]) -> None:
        for field in _FIELDS:
            val = account.get(field)
            if val is None or val == "":
                continue
            if field == "late_payments" and isinstance(val, dict):
                for bureau, buckets in val.items():
                    for days, count in (buckets or {}).items():
                        token = f"{bureau}:{days}:{count}"
                        self._tokens[field].add(token)
            elif isinstance(val, dict):
                for v in val.values():
                    if v:
                        s = str(v)
                        self._tokens[field].add(_redact(s))
            else:
                s = str(val)
                if isinstance(val, (int, float)) or s.isdigit():
                    self._tokens[field].add(s)
                else:
                    self._tokens[field].add(_redact(s))

    def save(self, folder: Path) -> None:
        """Write collected tokens to ``folder/candidate_tokens.json``."""

        data = {k: sorted(v) for k, v in self._tokens.items() if v}
        if not data:
            return
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "candidate_tokens.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


class StageATraceLogger:
    """Append per-account Stage A decisions to a JSONL trace file."""

    def __init__(self, session_id: str, base_folder: Path | None = None) -> None:
        if base_folder is None:
            manifest = RunManifest.for_sid(session_id, allow_create=False)
            base = manifest.ensure_run_subdir("logs_dir", "logs")
        else:
            base = base_folder
        self.path = base / "stageA_trace.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: Dict[str, Any]) -> None:
        data = dict(row)
        data["ts"] = datetime.utcnow().isoformat() + "Z"
        with self.path.open("a", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.write("\n")
