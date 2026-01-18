"""F0.A04 — Monthly Floor Integrity (record-level, non-blocking).

Validates that no monthly history entries (two_year_payment_history_monthly_tsv_v2)
show non-missing status before the opening month derived from date_opened.

Monthly entries before opening month are allowed only if status == "--".

Statuses: ok | conflict | skipped_missing_data (no unknown).
This branch is not gated by R1 and runs for every bureau. It is record-level and
writes under payload["record_integrity"]["F0"]["A04"].
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any, Mapping, Optional

from backend.core.logic.report_analysis.extractors.tokens import parse_date_any

log = logging.getLogger(__name__)

VERSION = "f0_a04_monthly_floor_integrity_v1"


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
        log.debug("F0_A04_UNPARSEABLE_DATE field=%s value=%s", field, raw)
        return None
    try:
        return datetime.strptime(iso, "%Y-%m-%d").date()
    except Exception:
        log.debug("F0_A04_UNPARSEABLE_DATE field=%s value=%s", field, raw)
        return None


def evaluate_f0_a04(
    payload: Mapping[str, Any],
    bureau_obj: Mapping[str, object],
    bureaus_data: Mapping[str, object],
    bureau: str,
    placeholders: set[str],
) -> dict:
    """Evaluate F0.A04 monthly floor integrity for a single bureau.

    date_opened is a hard floor: no monthly entry may show non-missing status
    before the opening month. Monthly entries before opening_month are allowed
    only if status == "--".

    Returns a record-level result dict to be stored under
    payload["record_integrity"]["F0"]["A04"].
    """

    result = {
        "version": VERSION,
        "status": "skipped_missing_data",
        "executed": True,
        "monthly_floor": {
            "date_opened": None,
            "opened_month_key": None,
            "conflict": False,
            "violations": [],
        },
        "explanation": "",
    }

    if not isinstance(bureau_obj, Mapping):
        result["status"] = "skipped_missing_data"
        result["explanation"] = "F0.A04 skipped_missing_data: bureau object missing or invalid"
        return result

    # Parse date_opened (required anchor)
    date_opened = _to_date(bureau_obj.get("date_opened"), "date_opened", placeholders)
    if date_opened is None:
        result["status"] = "skipped_missing_data"
        result["explanation"] = "F0.A04 skipped_missing_data: date_opened missing or unparseable"
        return result

    result["monthly_floor"]["date_opened"] = date_opened.isoformat()
    
    # Derive opening month for monthly history comparison
    opened_month_key = f"{date_opened.year:04d}-{date_opened.month:02d}"
    result["monthly_floor"]["opened_month_key"] = opened_month_key

    # Check monthly history
    monthly_data = bureaus_data.get("two_year_payment_history_monthly_tsv_v2")
    if not isinstance(monthly_data, Mapping):
        result["status"] = "skipped_missing_data"
        result["explanation"] = "F0.A04 skipped_missing_data: monthly history data missing or invalid"
        return result

    monthly_entries = monthly_data.get(bureau)
    if not isinstance(monthly_entries, list):
        result["status"] = "skipped_missing_data"
        result["explanation"] = "F0.A04 skipped_missing_data: monthly history data missing or invalid"
        return result

    if len(monthly_entries) == 0:
        result["status"] = "skipped_missing_data"
        result["explanation"] = "F0.A04 skipped_missing_data: monthly history data missing or invalid"
        return result

    # Validate monthly entries against floor
    violations = []
    for entry in monthly_entries:
        if not isinstance(entry, Mapping):
            continue
        entry_key = entry.get("month_year_key")
        if not entry_key or not isinstance(entry_key, str):
            continue
        
        # Check if month is before opening month
        if entry_key < opened_month_key:
            # Month precedes opening - only allowed if status is "--"
            status = entry.get("status", "")
            if status != "--":
                # Non-missing status before opening is a violation
                violations.append({
                    "field": f"monthly_history[{entry_key}]",
                    "date": f"{entry_key}-01",
                    "status": status
                })
        # If entry_key == opened_month_key, allowed (opening month included)
        # If entry_key > opened_month_key, allowed (after opening)

    if violations:
        result["status"] = "conflict"
        result["monthly_floor"]["conflict"] = True
        result["monthly_floor"]["violations"] = violations
        result["explanation"] = "F0.A04 conflict: at least one non-missing monthly entry precedes opening month"
        return result

    result["status"] = "ok"
    result["explanation"] = "F0.A04 ok: no non-missing monthly entries before opening month"
    return result
