"""Q2 — Activity expectation vs evidence (non-blocking, bureau-isolated).

Evaluates whether reported activity evidence aligns with the declared account state
(from Q1). Uses only allowed fields and never overrides Q1, gate, or payload status.
"""
from __future__ import annotations

from datetime import datetime, date
from typing import Mapping

from backend.core.logic.report_analysis.extractors.tokens import parse_date_any

ALLOWED_FIELDS = (
    "date_of_last_activity",
    "last_payment",
    "date_reported",
    "closed_date",
    "date_opened",
)


def _is_missing(value: object, placeholders: set[str]) -> bool:
    """Presence-only missing check consistent with gate/coverage.

    Missing when:
    - value is None
    - value is a string that is empty after trim
    - value lowercased matches any configured placeholder token
    """
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


def _to_date(raw: object, field: str, placeholders: set[str]) -> date | None:
    """Parse a raw date field to datetime.date; returns None on failure."""

    if _is_missing(raw, placeholders):
        return None
    if not isinstance(raw, str):
        return None

    iso = parse_date_any(raw)
    if not iso:
        return None

    try:
        return datetime.strptime(iso, "%Y-%m-%d").date()
    except Exception:
        return None


def evaluate_q2(
    bureau_obj: Mapping[str, object],
    q1_result: Mapping[str, object],
    placeholders: set[str],
) -> dict:
    """Evaluate Q2 for a single bureau.

    Parameters
    ----------
    bureau_obj: Mapping[str, object]
        Bureau-local object (e.g., bureaus.json["equifax"]).
    q1_result: Mapping[str, object]
        Output from Q1 (root_checks.Q1) for this bureau.
    placeholders: set[str]
        Lowercased placeholder tokens configured in environment.

    Returns
    -------
    dict
        Q2 result dictionary to be stored under payload["root_checks"]["Q2"].
    """
    declared_state = (
        q1_result.get("declared_state") if isinstance(q1_result, Mapping) else None
    )

    # Default skeleton
    result = {
        "version": "q2_activity_v1",
        "status": "unknown",
        "expected_activity": None,
        "observed_activity": None,
        "evidence_fields": [],
        "evidence": {},
        "explanation": "",
        "confidence": None,
    }

    # If Q1 is unknown/conflict, Q2 cannot proceed
    if declared_state in {"unknown", "conflict", None}:
        result["explanation"] = "Q2 skipped because Q1 state is unknown/conflict"
        return result

    expected_activity = declared_state == "open"
    result["expected_activity"] = expected_activity

    # Detect observed activity from allowed evidence fields in strict order
    # Only real customer-activity anchors; NOT reporting/snapshot timestamps
    evidence_fields: list[str] = []
    evidence: dict[str, object] = {}

    activity_field_order = (
        "date_of_last_activity",
        "last_payment",
    )

    activity_values: list[tuple[str, object]] = []
    for field in activity_field_order:
        raw_val = bureau_obj.get(field)
        if _is_missing(raw_val, placeholders):
            continue
        activity_values.append((field, raw_val))

    observed_activity = True if activity_values else None

    # Record primary evidence fields first (activity), then supporting anchors
    for field, raw_val in activity_values:
        evidence_fields.append(field)
        evidence[field] = raw_val

    # Collect supporting allowed fields (closed_date, date_opened) when present
    for field in ("closed_date", "date_opened"):
        raw_val = bureau_obj.get(field)
        if _is_missing(raw_val, placeholders):
            continue
        evidence_fields.append(field)
        evidence[field] = raw_val

    result["observed_activity"] = observed_activity
    result["evidence_fields"] = evidence_fields
    result["evidence"] = evidence

    # Decide status according to spec
    if expected_activity:
        if observed_activity:
            status = "ok"
            explanation = "expected activity (open) and activity observed"
        else:
            status = "skipped_missing_data"
            explanation = "expected activity (open) but no activity evidence found"
    else:
        closed_dt = _to_date(bureau_obj.get("closed_date"), "closed_date", placeholders)

        if closed_dt is None:
            if observed_activity:
                status = "skipped_missing_data"
                explanation = (
                    "closed_date missing or unparseable; activity evidence present but cannot compare to closure"
                )
            else:
                status = "skipped_missing_data"
                explanation = "closed_date missing or unparseable; no activity evidence to assess"
        else:
            if not observed_activity:
                status = "ok"
                explanation = "closed_date known; no activity evidence after closure"
            else:
                parsed_activity_dates: list[tuple[str, date]] = []
                unparseable_activity = False

                for field, raw_val in activity_values:
                    dt = _to_date(raw_val, field, placeholders)
                    if dt is None:
                        unparseable_activity = True
                    else:
                        parsed_activity_dates.append((field, dt))

                if unparseable_activity:
                    status = "skipped_missing_data"
                    explanation = (
                        "activity evidence present but at least one activity date is unparseable; cannot compare to closure"
                    )
                else:
                    has_post_close_activity = any(dt > closed_dt for _, dt in parsed_activity_dates)
                    if has_post_close_activity:
                        status = "conflict"
                        explanation = "activity evidence occurs after closed_date"
                    else:
                        status = "ok"
                        explanation = "activity evidence on/before closed_date; no post-closure activity detected"

    result["status"] = status
    result["explanation"] = explanation

    # Confidence heuristic: simple and deterministic
    if status in {"unknown", "skipped_missing_data"}:
        confidence = 0.5 if expected_activity else None
    elif observed_activity:
        confidence = 1.0
    else:
        confidence = 0.5

    result["confidence"] = confidence

    return result
