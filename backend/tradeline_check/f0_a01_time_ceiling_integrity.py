"""F0.A01 — Time Ceiling Integrity (record-level, non-blocking).

Validates that at least one ceiling candidate date (date_reported, last_verified)
exists and is greater than or equal to every other parseable bureau date field.

Statuses: ok | conflict | skipped_missing_data (no unknown).
This branch is not gated by R1 and runs for every bureau. It is record-level and
writes under payload["record_integrity"]["F0"]["A01"].
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any, Mapping, Optional

from backend.core.logic.report_analysis.extractors.tokens import parse_date_any

log = logging.getLogger(__name__)

VERSION = "f0_a01_time_ceiling_integrity_v2"


def _is_missing(value: object, placeholders: set[str]) -> bool:
    """Check for missing/placeholder values."""
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return True
        if s.lower() in placeholders:
            return True
        return False
    return False


def _to_date(raw: object, field: str, placeholders: set[str]) -> Optional[date]:
    """Parse a raw field to date using parse_date_any; returns None on failure."""
    if _is_missing(raw, placeholders):
        return None
    if not isinstance(raw, str):
        return None

    iso = parse_date_any(raw)
    if not iso:
        log.debug("F0_A01_UNPARSEABLE_DATE field=%s value=%s", field, raw)
        return None
    try:
        return datetime.strptime(iso, "%Y-%m-%d").date()
    except Exception:
        log.debug("F0_A01_UNPARSEABLE_DATE field=%s value=%s", field, raw)
        return None


def evaluate_f0_a01(
    payload: Mapping[str, Any],
    bureau_obj: Mapping[str, object],
    bureaus_data: Mapping[str, object],
    bureau: str,
    placeholders: set[str],
) -> dict:
    """Evaluate F0.A01 ceiling integrity for a single bureau.

    Validates that at least one ceiling candidate (date_reported, last_verified)
    exists and is greater than or equal to every other bureau date field.

    Returns a record-level result dict to be stored under
    payload["record_integrity"]["F0"]["A01"].
    """

    result = {
        "version": VERSION,
        "status": "skipped_missing_data",
        "executed": True,
        "ceiling": {
            "effective_ceiling_date": None,
            "ceiling_sources": [],
            "conflict": False,
            "violations": [],
        },
        "explanation": "",
    }

    if not isinstance(bureau_obj, Mapping):
        result["status"] = "skipped_missing_data"
        result["explanation"] = "F0.A01 skipped_missing_data: bureau object missing or invalid"
        return result

    # Collect ceiling candidate dates
    ceiling_fields = ("date_reported", "last_verified")
    ceiling_dates: list[date] = []
    ceiling_sources: list[str] = []

    for field in ceiling_fields:
        dt = _to_date(bureau_obj.get(field), field, placeholders)
        if dt is not None:
            ceiling_dates.append(dt)
            ceiling_sources.append(field)

    if not ceiling_dates:
        result["status"] = "skipped_missing_data"
        result["explanation"] = "F0.A01 skipped_missing_data: no parseable ceiling candidate dates"
        return result

    effective_ceiling = max(ceiling_dates)
    result["ceiling"]["effective_ceiling_date"] = effective_ceiling.isoformat()
    result["ceiling"]["ceiling_sources"] = ceiling_sources

    # Collect event dates (all other parseable dates on the tradeline)
    violations = []
    for field, raw_val in bureau_obj.items():
        dt = _to_date(raw_val, field, placeholders)
        if dt is None:
            continue
        # Event dates that exceed the ceiling are violations
        if dt > effective_ceiling:
            violations.append({"field": field, "date": dt.isoformat()})

    if violations:
        result["status"] = "conflict"
        result["ceiling"]["conflict"] = True
        result["ceiling"]["violations"] = violations
        result["explanation"] = "F0.A01 conflict: at least one date exceeds the ceiling"
        return result

    result["status"] = "ok"
    result["explanation"] = "F0.A01 ok: ceiling covers all bureau date fields"
    return result
