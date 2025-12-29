"""Q3 — Timeline coherence (non-blocking, bureau-isolated).

Determines if bureau-declared dates form a coherent, non-contradictory
high-level timeline. Uses only allowed fields, is deterministic, and never
modifies payload-level status, gate, coverage, findings, or blocked_questions.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Mapping, Optional

from backend.core.logic.report_analysis.extractors.tokens import parse_date_any

log = logging.getLogger(__name__)

ALLOWED_FIELDS = (
    "date_opened",
    "date_reported",
    "date_of_last_activity",
    "last_payment",
    "closed_date",
    "last_verified",  # context-only
)


def _is_missing(value: object, placeholders: set[str]) -> bool:
    """Presence-only missing check consistent with gate/coverage."""
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


def _to_date(raw: object, field: str) -> Optional[date]:
    """Parse a raw date string to datetime.date using parse_date_any.

    Returns None when missing or unparseable; logs at DEBUG on unparseable.
    """
    if not isinstance(raw, str):
        return None
    iso = parse_date_any(raw)
    if not iso:
        log.debug("Q3_UNPARSEABLE_DATE field=%s value=%s", field, raw)
        return None
    try:
        return datetime.strptime(iso, "%Y-%m-%d").date()
    except Exception:
        log.debug("Q3_UNPARSEABLE_DATE field=%s value=%s", field, raw)
        return None


def evaluate_q3(
    bureau_obj: Mapping[str, object],
    placeholders: set[str],
) -> dict:
    """Evaluate Q3 timeline coherence for a single bureau."""
    # Skeleton
    result = {
        "version": "q3_timeline_v1",
        "status": "unknown",
        "declared_timeline": "unknown",
        "conflicts": [],
        "evidence_fields": [],
        "evidence": {},
        "explanation": "",
        "confidence": None,
    }

    # Collect evidence (raw, allowed fields only)
    evidence: dict[str, object] = {}
    evidence_fields: list[str] = []
    for field in ALLOWED_FIELDS:
        raw_val = bureau_obj.get(field)
        if not _is_missing(raw_val, placeholders):
            evidence_fields.append(field)
            evidence[field] = raw_val

    # Minimum requirement: date_opened present AND at least one other timeline field present
    has_opened = not _is_missing(bureau_obj.get("date_opened"), placeholders)
    secondary_fields = (
        "date_reported",
        "date_of_last_activity",
        "last_payment",
        "closed_date",
    )
    has_secondary = any(not _is_missing(bureau_obj.get(f), placeholders) for f in secondary_fields)

    if not (has_opened and has_secondary):
        result.update(
            {
                "status": "skipped_missing_data",
                "declared_timeline": "unknown",
                "conflicts": [],
                "evidence_fields": evidence_fields,
                "evidence": evidence,
                "explanation": "Q3 skipped: missing minimum date_opened + one timeline date",
                "confidence": 0.0,
            }
        )
        return result

    # Parse dates we can
    parsed_dates: dict[str, Optional[date]] = {}
    for field in ("date_opened", "date_reported", "date_of_last_activity", "last_payment", "closed_date"):
        raw_val = bureau_obj.get(field)
        if _is_missing(raw_val, placeholders):
            parsed_dates[field] = None
            continue
        parsed_dates[field] = _to_date(raw_val, field)

    # Determine if we can run any comparison
    opened_dt = parsed_dates.get("date_opened")
    if opened_dt is None:
        # date_opened present but unparseable counts toward unknown path
        opened_dt = None

    comparisons_ran = False
    conflicts: list[str] = []

    # Helper to mark comparison executed only when both sides are present
    def compare_before(lhs: Optional[date], rhs: Optional[date], code: str) -> None:
        nonlocal comparisons_ran
        if lhs is None or rhs is None:
            return
        comparisons_ran = True
        if lhs < rhs:
            conflicts.append(code)

    def compare_after(lhs: Optional[date], rhs: Optional[date], code: str) -> None:
        nonlocal comparisons_ran
        if lhs is None or rhs is None:
            return
        comparisons_ran = True
        if lhs > rhs:
            conflicts.append(code)

    # Conflicts A/B/C: before open
    compare_before(parsed_dates.get("date_reported"), opened_dt, "report_before_open")
    compare_before(parsed_dates.get("date_of_last_activity"), opened_dt, "activity_before_open")
    compare_before(parsed_dates.get("last_payment"), opened_dt, "payment_before_open")

    # Conflicts D/E: after close (only if closed_date present)
    closed_dt = parsed_dates.get("closed_date")
    if closed_dt is not None:
        compare_after(parsed_dates.get("date_of_last_activity"), closed_dt, "activity_after_close")
        compare_after(parsed_dates.get("last_payment"), closed_dt, "payment_after_close")

    if not comparisons_ran:
        result.update(
            {
                "status": "unknown",
                "declared_timeline": "unknown",
                "conflicts": [],
                "evidence_fields": evidence_fields,
                "evidence": evidence,
                "explanation": "Q3 unknown: dates present but not parseable for comparisons",
                "confidence": 0.5,
            }
        )
        return result

    # Decide outcome
    if conflicts:
        status = "conflict"
        declared_timeline = "conflict"
        explanation = f"Timeline conflict: {','.join(conflicts)}"
    else:
        status = "ok"
        declared_timeline = "coherent"
        explanation = "Timeline coherent: no ordering conflicts detected"

    result.update(
        {
            "status": status,
            "declared_timeline": declared_timeline,
            "conflicts": conflicts,
            "evidence_fields": evidence_fields,
            "evidence": evidence,
            "explanation": explanation,
            "confidence": 1.0,
        }
    )

    return result
