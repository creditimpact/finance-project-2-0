"""F3.B02 — Closed Date vs Monthly Coverage (state-conditioned behavioral).

This check validates that the reported closed_date is covered by the monthly
history. If closed_date is after the latest month in monthly history, it's a
conflict (temporal impossibility).

Eligible only for R1 states {2, 3, 4} (closed, unknown, conflict) in the Q1-only router.
NOT eligible for state 1 (open).

Returns: "ok", "conflict", "skipped_missing_data", or "skipped" (ineligible).
No "unknown" status.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Mapping, Optional

from backend.core.logic.report_analysis.extractors.tokens import parse_date_any

log = logging.getLogger(__name__)

VERSION = "f3_b02_closed_date_vs_monthly_coverage_v1"

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


def _extract_max_month_key(monthly_entries: list) -> Optional[str]:
    """Extract the maximum (latest) parseable month_year_key from monthly history.
    
    Parameters
    ----------
    monthly_entries : list
        Monthly history entries from two_year_payment_history_monthly_tsv_v2[bureau]
        
    Returns
    -------
    str or None
        Max month_year_key in "YYYY-MM" format, or None if no parseable entries
    """
    if not isinstance(monthly_entries, list):
        return None
    
    parseable_keys = []
    for entry in monthly_entries:
        if not isinstance(entry, Mapping):
            continue
        
        month_key = entry.get("month_year_key")
        if not isinstance(month_key, str):
            continue
        
        # Validate format YYYY-MM
        if len(month_key) == 7 and month_key[4] == "-":
            try:
                year = int(month_key[:4])
                month = int(month_key[5:7])
                if 2000 <= year <= 2100 and 1 <= month <= 12:
                    parseable_keys.append(month_key)
            except ValueError:
                continue
    
    if not parseable_keys:
        return None
    
    return max(parseable_keys)


def evaluate_f3_b02(
    bureau_obj: Mapping[str, object],
    bureaus_data: Mapping[str, object],
    bureau: str,
    payload: Mapping[str, Any],
    placeholders: set[str],
) -> dict:
    """Evaluate F3.B02: Validate closed_date is covered by monthly history.

    Parameters
    ----------
    bureau_obj : Mapping[str, object]
        Bureau-local object (e.g., bureaus.json["equifax"]).
    bureaus_data : Mapping[str, object]
        Full bureaus.json mapping (for accessing two_year_payment_history_monthly_tsv_v2).
    bureau : str
        Bureau name (lowercase: "equifax", "experian", "transunion").
    payload : Mapping[str, Any]
        Full payload with routing and root_checks results.
    placeholders : set[str]
        Lowercased placeholder tokens configured in environment.

    Returns
    -------
    dict
        F3.B02 result dictionary to be stored under payload["branch_results"]["results"]["F3.B02"].
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
            "version": VERSION,
            "status": "skipped",
            "eligible": False,
            "executed": False,
            "fired": False,
            "trigger": {
                "r1_state_num": r1_state_num,
                "q1_declared_state": q1_declared_state,
            },
            "explanation": (
                "F3.B02 skipped: not eligible "
                "(R1.state_num must be 2, 3, or 4; closed, unknown, or conflict states)"
            ),
        }

    # ── Validate Required Input 1: closed_date ────────────────────────────
    closed_date_raw = bureau_obj.get("closed_date")
    
    if _is_missing(closed_date_raw, placeholders):
        return {
            "version": VERSION,
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
                "closed_month_key": None,
                "max_month_key": None,
                "monthly_entries_total": 0,
                "monthly_entries_parseable_count": 0,
            },
            "explanation": "F3.B02 skipped: closed_date is missing or placeholder",
        }
    
    closed_date = _parse_to_date(closed_date_raw)
    if closed_date is None:
        return {
            "version": VERSION,
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
                "closed_month_key": None,
                "max_month_key": None,
                "monthly_entries_total": 0,
                "monthly_entries_parseable_count": 0,
            },
            "explanation": "F3.B02 skipped: closed_date is unparseable",
        }

    closed_month_key = f"{closed_date.year:04d}-{closed_date.month:02d}"

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
            "version": VERSION,
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
                "closed_month_key": closed_month_key,
                "max_month_key": None,
                "monthly_entries_total": 0,
                "monthly_entries_parseable_count": 0,
            },
            "explanation": "F3.B02 skipped: two_year_payment_history_monthly_tsv_v2 is missing or empty for this bureau",
        }

    # ── Extract max month_year_key ─────────────────────────────────────────

    max_month_key = _extract_max_month_key(monthly_entries)
    
    # Count parseable entries
    parseable_count = 0
    for entry in monthly_entries:
        if isinstance(entry, Mapping):
            mk = entry.get("month_year_key")
            if isinstance(mk, str) and len(mk) == 7 and mk[4] == "-":
                try:
                    year = int(mk[:4])
                    month = int(mk[5:7])
                    if 2000 <= year <= 2100 and 1 <= month <= 12:
                        parseable_count += 1
                except ValueError:
                    pass

    if max_month_key is None:
        return {
            "version": VERSION,
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
                "closed_month_key": closed_month_key,
                "max_month_key": None,
                "monthly_entries_total": len(monthly_entries),
                "monthly_entries_parseable_count": parseable_count,
            },
            "explanation": "F3.B02 skipped: no parseable month_year_key in monthly history",
        }

    # ── Check for coverage violation ───────────────────────────────────────

    if closed_month_key > max_month_key:
        return {
            "version": VERSION,
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
                "closed_month_key": closed_month_key,
                "max_month_key": max_month_key,
                "monthly_entries_total": len(monthly_entries),
                "monthly_entries_parseable_count": parseable_count,
            },
            "explanation": (
                f"F3.B02 conflict: closed_date month ({closed_month_key}) "
                f"exceeds max monthly coverage ({max_month_key})"
            ),
        }
    else:
        return {
            "version": VERSION,
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
                "closed_month_key": closed_month_key,
                "max_month_key": max_month_key,
                "monthly_entries_total": len(monthly_entries),
                "monthly_entries_parseable_count": parseable_count,
            },
            "explanation": (
                f"F3.B02 ok: closed_date month ({closed_month_key}) "
                f"is within monthly coverage (max: {max_month_key})"
            ),
        }
