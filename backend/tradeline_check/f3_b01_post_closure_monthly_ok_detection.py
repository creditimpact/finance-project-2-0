"""F3.B01 — Post-Closure Monthly "OK" Detection (Timeline Deep Check).

Detects invalid "ok" status in two_year_payment_history_monthly_tsv_v2 after
an account's closed_date. A closed account cannot receive valid payments
("ok" status) after closure.

Eligible for R1 states {2, 3, 4} (closed, unknown, conflict) in the Q1-only router.
NOT eligible for state 1 (open).

This branch returns only: "ok", "conflict", or "skipped_missing_data".
No "unknown" status.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Mapping

from backend.core.logic.report_analysis.extractors.tokens import parse_date_any

log = logging.getLogger(__name__)

# Eligible R1 states: 2 (closed), 3 (unknown), 4 (conflict)
ELIGIBLE_R1_STATES = {2, 3, 4}


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


def _compute_first_forbidden_month(closed_dt: date) -> str:
    """Compute the first forbidden month (month after closure).
    
    Parameters
    ----------
    closed_dt : date
        The parsed closed_date
        
    Returns
    -------
    str
        Month key in format "YYYY-MM" representing first month after closure
    """
    # Next month after closure
    if closed_dt.month == 12:
        next_month = 1
        next_year = closed_dt.year + 1
    else:
        next_month = closed_dt.month + 1
        next_year = closed_dt.year
    
    return f"{next_year:04d}-{next_month:02d}"


def _deduplicate_monthly_entries(entries: list[dict[str, Any]]) -> dict[str, str]:
    """Deduplicate monthly entries by month_year_key.
    
    If multiple entries exist for the same month:
    - If ANY entry has status="ok", that month is "ok"
    - Otherwise, use first non-"--" entry
    
    Parameters
    ----------
    entries : list[dict]
        Raw monthly entries from two_year_payment_history_monthly_tsv_v2[bureau]
        
    Returns
    -------
    dict[str, str]
        Mapping of month_year_key -> status (deduplicated)
    """
    month_statuses: dict[str, list[str]] = {}
    
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        
        month_key = entry.get("month_year_key")
        status = entry.get("status")
        
        if not month_key or not isinstance(month_key, str):
            continue
        if not status or not isinstance(status, str):
            continue
            
        if month_key not in month_statuses:
            month_statuses[month_key] = []
        month_statuses[month_key].append(status)
    
    # Deduplicate: prefer "ok" if any entry has it
    deduped: dict[str, str] = {}
    for month_key, statuses in month_statuses.items():
        if "ok" in statuses:
            deduped[month_key] = "ok"
        else:
            # Use first non-"--" status, or "--" if all are placeholders
            non_placeholder = [s for s in statuses if s != "--"]
            deduped[month_key] = non_placeholder[0] if non_placeholder else statuses[0]
    
    return deduped


def evaluate_f3_b01(
    bureau_obj: Mapping[str, object],
    bureaus_data: Mapping[str, object],
    bureau: str,
    payload: Mapping[str, Any],
    placeholders: set[str],
) -> dict:
    """Evaluate F3.B01 post-closure monthly "ok" detection for a single bureau.

    Parameters
    ----------
    bureau_obj: Mapping[str, object]
        Bureau-local object (e.g., bureaus.json["equifax"]).
    bureaus_data: Mapping[str, object]
        Full bureaus.json mapping (for accessing two_year_payment_history_monthly_tsv_v2).
    bureau: str
        Bureau name (lowercase: "equifax", "experian", "transunion").
    payload: Mapping[str, Any]
        Full payload with routing and root_checks results.
    placeholders: set[str]
        Lowercased placeholder tokens configured in environment.

    Returns
    -------
    dict
        F3.B01 result dictionary to be stored under payload["branch_results"]["results"]["F3.B01"].
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
            "version": "f3_b01_post_closure_monthly_ok_detection_v1",
            "status": "skipped",
            "eligible": False,
            "executed": False,
            "fired": False,
            "trigger": {
                "r1_state_num": r1_state_num,
                "q1_declared_state": q1_declared_state,
            },
            "explanation": (
                "F3.B01 skipped: not eligible "
                "(R1.state_num must be 2, 3, or 4; closed, unknown, or conflict states)"
            ),
        }

    # ── Validate Required Input 1: closed_date ──────────────────────────
    closed_date_raw = bureau_obj.get("closed_date")
    
    if _is_missing(closed_date_raw, placeholders):
        return {
            "version": "f3_b01_post_closure_monthly_ok_detection_v1",
            "status": "skipped_missing_data",
            "eligible": True,
            "executed": True,
            "fired": False,
            "trigger": {
                "r1_state_num": r1_state_num,
                "q1_declared_state": q1_declared_state,
            },
            "explanation": "F3.B01 skipped: closed_date is missing or placeholder",
        }
    
    closed_date = _parse_to_date(closed_date_raw)
    if closed_date is None:
        return {
            "version": "f3_b01_post_closure_monthly_ok_detection_v1",
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
            },
            "explanation": "F3.B01 skipped: closed_date is unparseable",
        }

    # ── Validate Required Input 2: two_year_payment_history_monthly_tsv_v2 ──
    monthly_entries = None
    if isinstance(bureaus_data, Mapping):
        monthly_block = bureaus_data.get("two_year_payment_history_monthly_tsv_v2")
        if isinstance(monthly_block, Mapping):
            candidate = monthly_block.get(bureau)
            if isinstance(candidate, list):
                monthly_entries = candidate

    if not monthly_entries or len(monthly_entries) == 0:
        return {
            "version": "f3_b01_post_closure_monthly_ok_detection_v1",
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
                "closed_date_parsed": closed_date.isoformat(),
            },
            "explanation": "F3.B01 skipped: two_year_payment_history_monthly_tsv_v2 is missing or empty for this bureau",
        }

    # ── Compute First Forbidden Month ───────────────────────────────────
    first_forbidden_month = _compute_first_forbidden_month(closed_date)

    # ── Deduplicate Monthly Entries ─────────────────────────────────────
    deduped_months = _deduplicate_monthly_entries(monthly_entries)

    # ── Detect Post-Closure "OK" Violations ─────────────────────────────
    post_closure_ok_months: list[dict[str, str]] = []
    
    for month_key, status in deduped_months.items():
        # Only check months on or after the forbidden month
        if month_key >= first_forbidden_month:
            if status == "ok":
                post_closure_ok_months.append({
                    "month_year_key": month_key,
                    "status": "ok",
                })

    # ── Determine Final Status ──────────────────────────────────────────
    if post_closure_ok_months:
        # CONFLICT: "ok" status found after closure
        return {
            "version": "f3_b01_post_closure_monthly_ok_detection_v1",
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
                "first_forbidden_month": first_forbidden_month,
                "post_closure_ok_months": post_closure_ok_months,
            },
            "explanation": (
                f"F3.B01 conflict: account closed on {closed_date.isoformat()}, "
                f"but {len(post_closure_ok_months)} month(s) report status='ok' after closure"
            ),
        }
    else:
        # OK: No "ok" status after closure
        return {
            "version": "f3_b01_post_closure_monthly_ok_detection_v1",
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
                "first_forbidden_month": first_forbidden_month,
                "total_months_checked": len([k for k in deduped_months if k >= first_forbidden_month]),
            },
            "explanation": "F3.B01 ok: no 'ok' status detected in monthly history after closure",
        }
