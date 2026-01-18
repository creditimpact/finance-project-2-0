"""F0.A03 — Monthly Ceiling Integrity (record-level, non-blocking).

Validates that no monthly history entries (two_year_payment_history_monthly_tsv_v2)
exist after the effective ceiling month derived from date_reported/last_verified.

Statuses: ok | conflict | skipped_missing_data (no unknown).
This branch is not gated by R1 and runs for every bureau. It is record-level and
writes under payload["record_integrity"]["F0"]["A03"].
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any, Mapping, Optional

from backend.core.logic.report_analysis.extractors.tokens import parse_date_any

log = logging.getLogger(__name__)

VERSION = "f0_a03_monthly_ceiling_integrity_v1"


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
        log.debug("F0_A03_UNPARSEABLE_DATE field=%s value=%s", field, raw)
        return None
    try:
        return datetime.strptime(iso, "%Y-%m-%d").date()
    except Exception:
        log.debug("F0_A03_UNPARSEABLE_DATE field=%s value=%s", field, raw)
        return None


def evaluate_f0_a03(
    payload: Mapping[str, Any],
    bureau_obj: Mapping[str, object],
    bureaus_data: Mapping[str, object],
    bureau: str,
    placeholders: set[str],
) -> dict:
    """Evaluate F0.A03 monthly ceiling integrity for a single bureau.

    Validates that no monthly history entries exist after the ceiling month derived
    from date_reported/last_verified.

    Returns a record-level result dict to be stored under
    payload["record_integrity"]["F0"]["A03"].
    """

    result = {
        "version": VERSION,
        "status": "skipped_missing_data",
        "executed": True,
        "monthly_ceiling": {
            "effective_ceiling_date": None,
            "ceiling_month_key": None,
            "ceiling_sources": [],
            "conflict": False,
            "violations": [],
        },
        "explanation": "",
    }

    if not isinstance(bureau_obj, Mapping):
        result["status"] = "skipped_missing_data"
        result["explanation"] = "F0.A03 skipped_missing_data: bureau object missing or invalid"
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
        result["explanation"] = "F0.A03 skipped_missing_data: no parseable ceiling candidate dates"
        return result

    effective_ceiling = max(ceiling_dates)
    result["monthly_ceiling"]["effective_ceiling_date"] = effective_ceiling.isoformat()
    result["monthly_ceiling"]["ceiling_sources"] = ceiling_sources

    # Derive ceiling month for comparison
    ceiling_month_key = f"{effective_ceiling.year:04d}-{effective_ceiling.month:02d}"
    result["monthly_ceiling"]["ceiling_month_key"] = ceiling_month_key

    # Check monthly history
    monthly_data = bureaus_data.get("two_year_payment_history_monthly_tsv_v2")
    if not isinstance(monthly_data, Mapping):
        result["status"] = "skipped_missing_data"
        result["explanation"] = "F0.A03 skipped_missing_data: monthly history data missing or invalid"
        return result

    monthly_entries = monthly_data.get(bureau)
    if not isinstance(monthly_entries, list):
        result["status"] = "skipped_missing_data"
        result["explanation"] = "F0.A03 skipped_missing_data: monthly history data missing or invalid"
        return result

    if len(monthly_entries) == 0:
        result["status"] = "skipped_missing_data"
        result["explanation"] = "F0.A03 skipped_missing_data: monthly history data missing or invalid"
        return result

    # Validate monthly entries against ceiling
    violations = []
    for entry in monthly_entries:
        if not isinstance(entry, Mapping):
            continue
        entry_key = entry.get("month_year_key")
        if entry_key and isinstance(entry_key, str):
            # Any month after ceiling_month is a violation (status irrelevant)
            if entry_key > ceiling_month_key:
                violations.append({
                    "field": f"monthly_history[{entry_key}]",
                    "date": f"{entry_key}-01"
                })

    if violations:
        result["status"] = "conflict"
        result["monthly_ceiling"]["conflict"] = True
        result["monthly_ceiling"]["violations"] = violations
        result["explanation"] = "F0.A03 conflict: at least one monthly entry exceeds ceiling month"
        return result

    result["status"] = "ok"
    result["explanation"] = "F0.A03 ok: no monthly entries exceed ceiling month"
    return result
