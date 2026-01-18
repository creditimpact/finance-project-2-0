"""FX.B02 — Seven-Year vs Two-Year Late History Consistency (ungated, always-run).

This branch compares seven_year_history late counters against derived counts from
two_year_payment_history_monthly_tsv_v2 (per bureau). It flags a conflict when
any two-year delinquency bucket (30/60/90+) exceeds the corresponding seven-year
counter. Otherwise, it returns ok. If required inputs are missing/empty, it
returns skipped_missing_data. No "unknown" status is emitted.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, MutableMapping

log = logging.getLogger(__name__)

VERSION = "fx_b02_seven_year_vs_two_year_consistency_v1"

# Severity mapping aligned with FX.B01 semantics
# Returns bucket label or None for missing/ignored statuses
STATUS_BUCKET_MAP = {
    "ok": None,
    "current": None,
    "--": None,  # missing
    "30": "late30",
    "60": "late60",
    "90": "late90",
    "120": "late90",
    "150": "late90",
    "180": "late90",
    "co": "late90",
    "chargeoff": "late90",
}

BUCKET_ORDER = ("late30", "late60", "late90")


def _normalize_seven_year_counts(raw: Mapping[str, Any] | None) -> Dict[str, int] | None:
    """Extract integer late counters from seven_year_history entry.

    Returns dict with late30/late60/late90 or None if malformed/missing.
    Zero is considered valid (not missing).
    """
    if not isinstance(raw, Mapping):
        return None

    out: Dict[str, int] = {}
    for key in BUCKET_ORDER:
        val = raw.get(key)
        try:
            out[key] = int(val)
        except (TypeError, ValueError):
            return None
    return out


def _dedupe_and_count_monthly(entries: Any) -> Dict[str, Any]:
    """Dedupe monthly entries by month_year_key and return bucket counts and sample months.

    Returns dict with:
      counts: {late30, late60, late90}
      months_by_bucket: {bucket: [month_year_key, ...]} (deduped worst-only)
      valid_entries: int (number of months considered after dedupe)
    """
    if not isinstance(entries, list) or len(entries) == 0:
        return {
            "counts": {b: 0 for b in BUCKET_ORDER},
            "months_by_bucket": {b: [] for b in BUCKET_ORDER},
            "valid_entries": 0,
        }

    # Pick worst bucket per month
    month_to_bucket: Dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        mk = entry.get("month_year_key")
        status = entry.get("status")
        if not isinstance(mk, str) or not isinstance(status, str):
            continue
        bucket = STATUS_BUCKET_MAP.get(status.strip().lower())
        if bucket is None:
            # missing/ok/current/-- ignored
            continue
        # Keep worst bucket (late90 > late60 > late30)
        prev = month_to_bucket.get(mk)
        if prev is None:
            month_to_bucket[mk] = bucket
        else:
            if _bucket_worse(bucket, prev):
                month_to_bucket[mk] = bucket

    counts = {b: 0 for b in BUCKET_ORDER}
    months_by_bucket = {b: [] for b in BUCKET_ORDER}

    for mk, bucket in month_to_bucket.items():
        counts[bucket] += 1
        months_by_bucket[bucket].append(mk)

    return {
        "counts": counts,
        "months_by_bucket": months_by_bucket,
        "valid_entries": len(month_to_bucket),
    }


def _bucket_worse(candidate: str, current: str) -> bool:
    """Return True if candidate bucket is worse than current (late90 > late60 > late30)."""
    order = {"late30": 1, "late60": 2, "late90": 3}
    return order.get(candidate, 0) > order.get(current, 0)


def evaluate_fx_b02(
    bureau_obj: Mapping[str, object],
    bureaus_data: Mapping[str, object],
    bureau: str,
    placeholders: set[str],
) -> Mapping[str, Any]:
    """Evaluate FX.B02 consistency between 7-year summary and 2-year monthly history.

    Status space: ok | conflict | skipped_missing_data (no unknown).
    Always eligible, ungated, executed when invoked.
    """
    result: MutableMapping[str, Any] = {
        "version": VERSION,
        "status": "skipped_missing_data",  # default to skipped until proven otherwise
        "eligible": True,
        "executed": True,
        "fired": False,
        "ungated": True,
        "evidence": {
            "seven_year_counts": None,
            "two_year_counts": None,
            "example_months": [],
        },
        "explanation": "",
    }

    # ---- Load seven_year_history ----
    seven_block = None
    if isinstance(bureaus_data, Mapping):
        seven_block = bureaus_data.get("seven_year_history")
    seven_for_bureau = seven_block.get(bureau) if isinstance(seven_block, Mapping) else None
    seven_counts = _normalize_seven_year_counts(seven_for_bureau)
    if seven_counts is None:
        result["explanation"] = "FX.B02 skipped: seven_year_history missing or malformed"
        return result

    # ---- Load two_year_payment_history_monthly_tsv_v2 ----
    monthly_block = bureaus_data.get("two_year_payment_history_monthly_tsv_v2") if isinstance(bureaus_data, Mapping) else None
    monthly_entries = monthly_block.get(bureau) if isinstance(monthly_block, Mapping) else None
    deduped = _dedupe_and_count_monthly(monthly_entries)
    two_counts = deduped["counts"]

    result["evidence"]["seven_year_counts"] = seven_counts
    result["evidence"]["two_year_counts"] = two_counts

    if deduped["valid_entries"] == 0:
        result["explanation"] = "FX.B02 skipped: monthly history missing or all missing entries"
        return result

    # ---- Compare buckets ----
    overages = []
    for bucket in BUCKET_ORDER:
        if two_counts[bucket] > seven_counts[bucket]:
            overages.append(bucket)

    if overages:
        result["status"] = "conflict"
        result["fired"] = True
        # Collect example months from worst buckets first
        example_months: list[str] = []
        for bucket in ("late90", "late60", "late30"):
            for mk in deduped["months_by_bucket"].get(bucket, []):
                if len(example_months) >= 5:
                    break
                example_months.append(mk)
            if len(example_months) >= 5:
                break
        result["evidence"]["example_months"] = example_months
        result["explanation"] = (
            "FX.B02 conflict: two-year delinquency counts exceed seven-year summary in "
            + ",".join(overages)
        )
    else:
        result["status"] = "ok"
        result["explanation"] = "FX.B02 ok: two-year delinquency counts do not exceed seven-year summary"

    return result
