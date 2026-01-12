"""FX.B01 — Last Payment Monotonicity (ungated, always-run).

This check runs for every bureau and every account (no R1 gating).
It validates that the reported last_payment truly represents the LAST payment
by enforcing monotonic delinquency behavior after that date.

Core question: "After the month of last_payment, does the monthly delinquency
severity ever IMPROVE?"

Any improvement = CONTRADICTION.

Non-blocking: never modifies root_checks, routing, gate, or payload status.
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any, Mapping, Optional, Tuple

from backend.core.logic.report_analysis.extractors.tokens import parse_date_any

log = logging.getLogger(__name__)

VERSION = "fx_b01_last_payment_monotonicity_v1"

# Severity mapping: lower = better, higher = worse
# OK/current = 0
# Missing ("--") = None (skipped)
# Chargeoff/180 days past due = highest (999)
SEVERITY_MAP = {
    "ok": 0,
    "current": 0,
    "30": 30,
    "60": 60,
    "90": 90,
    "120": 120,
    "150": 150,
    "180": 999,  # Terminal delinquency
    "co": 999,  # Chargeoff (terminal)
    "chargeoff": 999,
}


def _is_missing(val: object, placeholders: set[str]) -> bool:
    """Check if value is missing/placeholder."""
    if val is None:
        return True
    if isinstance(val, str):
        stripped = val.strip().lower()
        if not stripped:
            return True
        if stripped in placeholders:
            return True
        return False
    return False


def _to_date(raw: object, placeholders: set[str]) -> Optional[date]:
    """Parse a raw date string to datetime.date; returns None on failure."""
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


def _parse_month_label(month_label: str) -> Optional[Tuple[int, int]]:
    """Parse month label to (year, month) tuple.
    
    Supports formats:
    - "MM/YYYY" (e.g., "01/2024" → (2024, 1))
    - "YYYY mon" (e.g., "2023 jan" → (2023, 1))  [future extension]
    
    Returns: (year, month) or None
    """
    if not isinstance(month_label, str):
        return None
    
    cleaned = month_label.strip()
    
    # Try MM/YYYY format (e.g., "01/2024")
    if "/" in cleaned:
        parts = cleaned.split("/")
        if len(parts) == 2:
            try:
                month_num = int(parts[0])
                year_num = int(parts[1])
                if 1 <= month_num <= 12 and 2000 <= year_num <= 2100:
                    return (year_num, month_num)
            except ValueError:
                pass
    
    return None


def _get_severity(status: str) -> Optional[int]:
    """Map monthly status to severity score.
    
    Returns None for missing statuses ("--").
    """
    if not isinstance(status, str):
        return None
    
    cleaned = status.strip().lower()
    
    if cleaned == "--":
        return None
    
    return SEVERITY_MAP.get(cleaned)


def evaluate_fx_b01(
    bureau_obj: Mapping[str, object],
    bureaus_data: Mapping[str, object],
    bureau: str,
    placeholders: set[str],
) -> Mapping[str, Any]:
    """Evaluate FX.B01: Validate last_payment monotonicity for a single bureau.

    Parameters
    ----------
    bureau_obj : Mapping[str, object]
        Bureau-local object from bureaus.json[bureau].
    bureaus_data : Mapping[str, object]
        Full bureaus.json mapping (for accessing two_year_payment_history_monthly_tsv_v2).
    bureau : str
        Bureau name (lowercase: "equifax", "experian", "transunion").
    placeholders : set[str]
        Lowercased placeholder tokens.

    Returns
    -------
    dict
        FX.B01 result dict with:
        - version: Check version string
        - status: "ok", "conflict", "skipped_missing_data", "unknown"
        - eligible: Always True (ungated)
        - executed: True if check ran successfully
        - fired: True if conflict detected
        - ungated: Always True (FX branches always run)
        - evidence: Dict with last_payment info, violation details
        - explanation: Human-readable summary
    """

    # ── Skeleton result ───────────────────────────────────────────────────

    result = {
        "version": VERSION,
        "status": "unknown",
        "eligible": True,  # FX branches are always eligible
        "executed": False,
        "fired": False,  # Set to True if conflict detected
        "ungated": True,  # FX branches are always ungated
        "evidence": {
            "last_payment_raw": None,
            "last_payment_month_year": None,
            "monthly_entries_checked": 0,
            "detected_violation": None,
        },
        "explanation": "",
    }

    # ── Precondition 1: last_payment must be parseable ────────────────────

    last_payment_raw = bureau_obj.get("last_payment")
    result["evidence"]["last_payment_raw"] = last_payment_raw if isinstance(last_payment_raw, str) else None

    if _is_missing(last_payment_raw, placeholders):
        result["status"] = "skipped_missing_data"
        result["explanation"] = "FX.B01 skipped: last_payment missing or placeholder"
        return result

    last_payment_dt = _to_date(last_payment_raw, placeholders)
    if last_payment_dt is None:
        result["status"] = "skipped_missing_data"
        result["explanation"] = "FX.B01 skipped: last_payment unparseable"
        return result

    last_payment_year = last_payment_dt.year
    last_payment_month = last_payment_dt.month
    result["evidence"]["last_payment_month_year"] = f"{last_payment_year}-{last_payment_month:02d}"

    # ── Precondition 2: monthly history must exist ────────────────────────

    monthly_entries = None
    if isinstance(bureaus_data, Mapping):
        monthly_block = bureaus_data.get("two_year_payment_history_monthly_tsv_v2")
        if isinstance(monthly_block, Mapping):
            candidate = monthly_block.get(bureau)
            if isinstance(candidate, list):
                monthly_entries = candidate

    if not monthly_entries:
        result["status"] = "skipped_missing_data"
        result["explanation"] = "FX.B01 skipped: two_year_payment_history_monthly_tsv_v2 missing or empty"
        return result

    # ── Precondition 3: Find last_payment month in history ────────────────

    last_payment_index = None
    target_month_year_key = f"{last_payment_year:04d}-{last_payment_month:02d}"

    for i, entry in enumerate(monthly_entries):
        if not isinstance(entry, Mapping):
            continue

        # Use month_year_key directly (format: "YYYY-MM")
        entry_key = entry.get("month_year_key")
        if not isinstance(entry_key, str):
            continue

        # Check if this entry matches last_payment month via month_year_key
        if entry_key == target_month_year_key:
            last_payment_index = i
            break

    if last_payment_index is None:
        result["status"] = "skipped_missing_data"
        result["explanation"] = (
            f"FX.B01 skipped: last_payment month_year_key ({target_month_year_key}) "
            "not found in monthly history"
        )
        return result

    # ── Scan forward after last_payment month for severity improvement ────

    result["executed"] = True

    # Build severity sequence including baseline (last_payment month) if available,
    # then append non-missing severities after last_payment
    severity_sequence: list[tuple[int, str, int]] = []  # (index, month_year_key, severity)

    # Baseline from last_payment month
    baseline_entry = monthly_entries[last_payment_index]
    if isinstance(baseline_entry, Mapping):
        baseline_status = baseline_entry.get("status")
        baseline_month_key = baseline_entry.get("month_year_key")
        if isinstance(baseline_status, str) and isinstance(baseline_month_key, str):
            baseline_severity = _get_severity(baseline_status)
            if baseline_severity is not None:
                severity_sequence.append((last_payment_index, baseline_month_key, baseline_severity))

    for i in range(last_payment_index + 1, len(monthly_entries)):
        entry = monthly_entries[i]
        if not isinstance(entry, Mapping):
            continue

        status = entry.get("status")
        if not isinstance(status, str):
            continue

        month_key = entry.get("month_year_key")
        if not isinstance(month_key, str):
            continue

        severity = _get_severity(status)
        if severity is None:
            # Skip missing ("--") months
            continue

        severity_sequence.append((i, month_key, severity))

    # monthly_entries_checked counts non-missing months AFTER last_payment
    result["evidence"]["monthly_entries_checked"] = max(0, len(severity_sequence) - (1 if severity_sequence and severity_sequence[0][0] == last_payment_index else 0))

    # ── Check for severity improvement (monotonicity violation) ───────────

    violation_detected = None

    for idx in range(1, len(severity_sequence)):
        prev_i, prev_month, prev_severity = severity_sequence[idx - 1]
        curr_i, curr_month, curr_severity = severity_sequence[idx]

        if curr_severity < prev_severity:
            # Severity improved: VIOLATION
            violation_detected = {
                "prev_month": prev_month,
                "curr_month": curr_month,
                "prev_severity": prev_severity,
                "current_severity": curr_severity,
                "message": f"Severity improved from {prev_severity} to {curr_severity} ({prev_month} → {curr_month})",
            }
            break

    if violation_detected is not None:
        result["status"] = "conflict"
        result["fired"] = True
        result["evidence"]["detected_violation"] = violation_detected
        result["explanation"] = (
            f"FX.B01 conflict: delinquency severity improved after last_payment "
            f"({violation_detected['message']})"
        )
    else:
        result["status"] = "ok"
        result["fired"] = False
        result["explanation"] = (
            f"FX.B01 ok: no severity improvement detected after last_payment "
            f"(checked {result['evidence']['monthly_entries_checked']} non-missing months)"
        )

    return result
