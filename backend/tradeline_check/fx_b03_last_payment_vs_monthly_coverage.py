"""FX.B03 — Last Payment vs Monthly Coverage (ungated, always-run).

This check validates that the reported last_payment date is covered by the
monthly history. If last_payment is after the latest month in monthly history,
it's a conflict (temporal impossibility).

Ungated: runs for every bureau and every account (no R1 gating).
Non-blocking: never modifies root_checks, routing, gate, or payload status.

Returns only: "ok", "conflict", or "skipped_missing_data".
No "unknown" status.
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Any, Mapping, Optional

from backend.core.logic.report_analysis.extractors.tokens import parse_date_any

log = logging.getLogger(__name__)

VERSION = "fx_b03_last_payment_vs_monthly_coverage_v1"


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


def evaluate_fx_b03(
    bureau_obj: Mapping[str, object],
    bureaus_data: Mapping[str, object],
    bureau: str,
    placeholders: set[str],
) -> Mapping[str, Any]:
    """Evaluate FX.B03: Validate last_payment is covered by monthly history.

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
        FX.B03 result dict with:
        - version: Check version string
        - status: "ok", "conflict", "skipped_missing_data"
        - eligible: Always True (ungated)
        - executed: True if check ran successfully
        - fired: True if conflict detected
        - ungated: Always True (FX branches always run)
        - evidence: Dict with last_payment info, monthly coverage details
        - explanation: Human-readable summary
    """

    # ── Skeleton result ───────────────────────────────────────────────────

    result = {
        "version": VERSION,
        "status": "skipped_missing_data",
        "eligible": True,  # FX branches are always eligible
        "executed": False,
        "fired": False,  # Set to True if conflict detected
        "ungated": True,  # FX branches are always ungated
        "evidence": {
            "last_payment_raw": None,
            "last_payment_month_key": None,
            "max_month_key": None,
            "monthly_entries_total": 0,
            "monthly_entries_parseable_count": 0,
        },
        "explanation": "",
    }

    # ── Precondition 1: last_payment must be parseable ────────────────────

    last_payment_raw = bureau_obj.get("last_payment")
    result["evidence"]["last_payment_raw"] = last_payment_raw if isinstance(last_payment_raw, str) else None

    if _is_missing(last_payment_raw, placeholders):
        result["status"] = "skipped_missing_data"
        result["explanation"] = "FX.B03 skipped: last_payment missing or placeholder"
        return result

    last_payment_dt = _to_date(last_payment_raw, placeholders)
    if last_payment_dt is None:
        result["status"] = "skipped_missing_data"
        result["explanation"] = "FX.B03 skipped: last_payment unparseable"
        return result

    last_payment_month_key = f"{last_payment_dt.year:04d}-{last_payment_dt.month:02d}"
    result["evidence"]["last_payment_month_key"] = last_payment_month_key

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
        result["explanation"] = "FX.B03 skipped: two_year_payment_history_monthly_tsv_v2 missing or empty"
        return result

    result["evidence"]["monthly_entries_total"] = len(monthly_entries)

    # ── Precondition 3: extract max month_year_key ─────────────────────────

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
    
    result["evidence"]["monthly_entries_parseable_count"] = parseable_count
    result["evidence"]["max_month_key"] = max_month_key

    if max_month_key is None:
        result["status"] = "skipped_missing_data"
        result["explanation"] = "FX.B03 skipped: no parseable month_year_key in monthly history"
        return result

    # ── Check for coverage violation ───────────────────────────────────────

    result["executed"] = True

    if last_payment_month_key > max_month_key:
        result["status"] = "conflict"
        result["fired"] = True
        result["explanation"] = (
            f"FX.B03 conflict: last_payment month ({last_payment_month_key}) "
            f"exceeds max monthly coverage ({max_month_key})"
        )
    else:
        result["status"] = "ok"
        result["fired"] = False
        result["explanation"] = (
            f"FX.B03 ok: last_payment month ({last_payment_month_key}) "
            f"is within monthly coverage (max: {max_month_key})"
        )

    return result
