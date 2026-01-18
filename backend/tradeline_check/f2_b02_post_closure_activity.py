"""F2.B02 — Post-Closure Activity Contradiction (state-conditioned behavioral).

Evaluates whether any activity or payment dates occur AFTER the declared closed_date,
which is logically impossible for a closed account.

Eligible only for R1 state 2 (Q1=closed) in the Q1-only router.

Returns exactly 3 statuses:
- ok: closed_date valid, no post-closure activity detected (or activity fields missing/unparseable)
- conflict: closed_date valid, activity date > closed_date
- skipped_missing_data: closed_date missing/unparseable (cannot evaluate)

This branch does NOT read monthly history, date_reported, last_verified, or date_convention.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Mapping

from backend.core.logic.report_analysis.extractors.tokens import parse_date_any

log = logging.getLogger(__name__)

# Eligible R1 state: Q1=closed (state_num=2)
ELIGIBLE_R1_STATES = {2}

# Fields to check for post-closure activity (ordered)
ACTIVITY_FIELDS = ("date_of_last_activity", "last_payment")


def _is_missing(value: object, placeholders: set[str]) -> bool:
    """Check if a value is missing or placeholder."""
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


def _parse_to_date(raw: object) -> date | None:
    """Parse a raw date string to datetime.date; returns None on failure."""
    if not isinstance(raw, str):
        return None
    iso = parse_date_any(raw)
    if not iso:
        return None
    try:
        return datetime.strptime(iso, "%Y-%m-%d").date()
    except Exception:
        return None


def evaluate_f2_b02(
    bureau_obj: Mapping[str, object],
    payload: Mapping[str, Any],
    placeholders: set[str],
) -> dict:
    """Evaluate F2.B02 post-closure activity contradiction for a single bureau.

    Parameters
    ----------
    bureau_obj: Mapping[str, object]
        Bureau-local object (e.g., bureaus.json["equifax"]).
    payload: Mapping[str, Any]
        Full payload with routing and root_checks results.
    placeholders: set[str]
        Lowercased placeholder tokens configured in environment.

    Returns
    -------
    dict
        F2.B02 result dictionary to be stored under payload["branch_results"]["results"]["F2.B02"].
    """
    # Extract routing information
    routing = payload.get("routing", {})
    r1_result = routing.get("R1", {}) if isinstance(routing, Mapping) else {}
    r1_state_num = r1_result.get("state_num") if isinstance(r1_result, Mapping) else None

    # Extract Q1 status for trigger evidence
    root_checks = payload.get("root_checks", {})
    q1_result = root_checks.get("Q1", {}) if isinstance(root_checks, Mapping) else {}
    
    q1_declared_state = q1_result.get("declared_state") if isinstance(q1_result, Mapping) else None

    # Check eligibility based on R1 state
    if r1_state_num not in ELIGIBLE_R1_STATES:
        return {
            "version": "f2_b02_post_closure_activity_v1",
            "status": "skipped",
            "eligible": False,
            "executed": False,
            "fired": False,
            "trigger": {
                "r1_state_num": r1_state_num,
                "q1_declared_state": q1_declared_state,
            },
            "explanation": "F2.B02 skipped: not eligible (R1.state_num must be 2; requires Q1=closed)",
        }

    # Parse closed_date - required for evaluation
    closed_date_raw = bureau_obj.get("closed_date")
    
    # Check if closed_date is missing or placeholder
    if _is_missing(closed_date_raw, placeholders):
        return {
            "version": "f2_b02_post_closure_activity_v1",
            "status": "skipped_missing_data",
            "eligible": True,
            "executed": True,
            "fired": False,
            "trigger": {
                "r1_state_num": r1_state_num,
                "q1_declared_state": q1_declared_state,
            },
            "evidence": {
                "closed_date_raw": closed_date_raw,
                "reason": "closed_date missing or placeholder",
            },
            "explanation": "F2.B02 skipped: closed_date is missing or placeholder; cannot evaluate post-closure activity",
        }
    
    # Parse closed_date
    closed_date = _parse_to_date(closed_date_raw)
    if closed_date is None:
        return {
            "version": "f2_b02_post_closure_activity_v1",
            "status": "skipped_missing_data",
            "eligible": True,
            "executed": True,
            "fired": False,
            "trigger": {
                "r1_state_num": r1_state_num,
                "q1_declared_state": q1_declared_state,
            },
            "evidence": {
                "closed_date_raw": closed_date_raw,
                "reason": "closed_date unparseable",
            },
            "explanation": "F2.B02 skipped: closed_date is unparseable; cannot evaluate post-closure activity",
        }

    # Check each activity field for post-closure violations
    for field in ACTIVITY_FIELDS:
        raw_val = bureau_obj.get(field)
        
        # Skip missing/placeholder values
        if _is_missing(raw_val, placeholders):
            continue
        
        # Parse the activity date
        activity_date = _parse_to_date(raw_val)
        
        # Skip unparseable dates (not a violation, just can't compare)
        if activity_date is None:
            continue
        
        # Check for violation: activity date AFTER closed date
        if activity_date > closed_date:
            return {
                "version": "f2_b02_post_closure_activity_v1",
                "status": "conflict",
                "eligible": True,
                "executed": True,
                "fired": True,
                "trigger": {
                    "r1_state_num": r1_state_num,
                    "q1_declared_state": q1_declared_state,
                },
                "evidence": {
                    "closed_date_raw": closed_date_raw,
                    "closed_date_parsed": closed_date.isoformat(),
                    "violating_field": field,
                    "violating_date_raw": raw_val,
                    "violating_date_parsed": activity_date.isoformat(),
                },
                "explanation": f"F2.B02 fired: {field}={activity_date.isoformat()} is after closed_date={closed_date.isoformat()}",
            }
    
    # All activity dates were before/equal to closed_date, or missing/unparseable
    # (or activity fields are missing/unparseable, which we treat as "no activity evidence")
    checked_fields_present = [
        field for field in ACTIVITY_FIELDS
        if not _is_missing(bureau_obj.get(field), placeholders)
    ]
    
    # Count how many present fields were successfully parsed
    parsed_count = 0
    for field in checked_fields_present:
        raw_val = bureau_obj.get(field)
        if _parse_to_date(raw_val) is not None:
            parsed_count += 1
    
    return {
        "version": "f2_b02_post_closure_activity_v1",
        "status": "ok",
        "eligible": True,
        "executed": True,
        "fired": False,
        "trigger": {
            "r1_state_num": r1_state_num,
            "q1_declared_state": q1_declared_state,
        },
        "evidence": {
            "closed_date_raw": closed_date_raw,
            "closed_date_parsed": closed_date.isoformat(),
            "checked_fields_present": checked_fields_present,
            "parsed_fields_count": parsed_count,
        },
        "explanation": "F2.B02 ok: no activity or payments detected after closed_date",
    }
