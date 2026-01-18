"""F0.A02 — Opening Date Lower Bound Integrity (record-level, non-blocking).

Validates that the date_opened field serves as a hard lower bound for all other
bureau date fields on the tradeline. No date field may be earlier than date_opened.

Statuses: ok | conflict | skipped_missing_data (no unknown).
This branch is not gated by R1 and runs for every bureau. It is record-level and
writes under payload["record_integrity"]["F0"]["A02"].
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any, Mapping, Optional

from backend.core.logic.report_analysis.extractors.tokens import parse_date_any

log = logging.getLogger(__name__)

VERSION = "f0_a02_opening_date_lower_bound_v2"


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
        log.debug("F0_A02_UNPARSEABLE_DATE field=%s value=%s", field, raw)
        return None
    try:
        return datetime.strptime(iso, "%Y-%m-%d").date()
    except Exception:
        log.debug("F0_A02_UNPARSEABLE_DATE field=%s value=%s", field, raw)
        return None


def evaluate_f0_a02(
    payload: Mapping[str, Any],
    bureau_obj: Mapping[str, object],
    bureaus_data: Mapping[str, object],
    bureau: str,
    placeholders: set[str],
) -> dict:
    """Evaluate F0.A02 opening date lower bound integrity for a single bureau.

    date_opened is a hard floor: no other bureau date field may be earlier.

    Returns a record-level result dict to be stored under
    payload["record_integrity"]["F0"]["A02"].
    """

    result = {
        "version": VERSION,
        "status": "skipped_missing_data",
        "executed": True,
        "floor": {
            "date_opened": None,
            "conflict": False,
            "violations": [],
        },
        "explanation": "",
    }

    if not isinstance(bureau_obj, Mapping):
        result["status"] = "skipped_missing_data"
        result["explanation"] = "F0.A02 skipped_missing_data: bureau object missing or invalid"
        return result

    # Parse date_opened (required anchor)
    date_opened = _to_date(bureau_obj.get("date_opened"), "date_opened", placeholders)
    if date_opened is None:
        result["status"] = "skipped_missing_data"
        result["explanation"] = "F0.A02 skipped_missing_data: date_opened missing or unparseable"
        return result

    result["floor"]["date_opened"] = date_opened.isoformat()

    # ── Check Bureau Object Fields ──
    violations = []
    for field, raw_val in bureau_obj.items():
        if field == "date_opened":
            continue  # Skip self-comparison
        dt = _to_date(raw_val, field, placeholders)
        if dt is None:
            continue
        # Event dates that precede the floor are violations
        if dt < date_opened:
            violations.append({"field": field, "date": dt.isoformat()})

    if violations:
        result["status"] = "conflict"
        result["floor"]["conflict"] = True
        result["floor"]["violations"] = violations
        result["explanation"] = "F0.A02 conflict: at least one date precedes date_opened"
        return result

    result["status"] = "ok"
    result["explanation"] = "F0.A02 ok: all bureau date fields >= date_opened"
    return result
