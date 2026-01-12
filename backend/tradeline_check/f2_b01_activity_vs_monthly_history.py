"""F2.B01 — Activity vs Monthly History (Payment Performance).

Evaluates whether the bureau's two_year_payment_history_monthly_tsv_v2 contains
activity evidence (ok or delinquency months) and checks alignment with Q2's
activity output. Eligible only for R1 state 1 (Q1=open, Q2=ok).

Non-blocking: never modifies payload-level status, gate, coverage, findings,
blocked_questions, or root_checks.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

log = logging.getLogger(__name__)


def evaluate_f2_b01(
    payload: Mapping[str, Any],
    bureau_obj: Mapping[str, object],
    bureaus_data: Mapping[str, object],
    bureau: str,
    placeholders: set[str],
) -> dict:
    """Evaluate F2.B01: Activity vs Monthly History for a single bureau.

    Parameters
    ----------
    payload : Mapping[str, Any]
        Current bureau payload (read-only). Contains routing, root_checks, etc.
    bureau_obj : Mapping[str, object]
        Bureau-local object from bureaus.json[bureau].
    bureaus_data : Mapping[str, object]
        Full bureaus.json mapping (for accessing two_year_payment_history_monthly_tsv_v2).
    bureau : str
        Bureau name (lowercase: "equifax", "experian", "transunion").
    placeholders : set[str]
        Lowercased placeholder tokens (not used directly, but for consistency).

    Returns
    -------
    dict
        Result dict to be stored at payload["branch_results"]["results"]["F2.B01"].
        Never raises; logs on recoverable errors and returns safe result.
    """

    # ── Eligibility Gating ────────────────────────────────────────────────
    # Eligible ONLY for R1 state 1 (Q1=open, Q2=ok)

    routing = payload.get("routing", {}) if isinstance(payload, Mapping) else {}
    r1 = routing.get("R1", {}) if isinstance(routing, Mapping) else {}
    r1_state_num = r1.get("state_num")

    try:
        r1_state_num = int(r1_state_num) if r1_state_num is not None else 0
    except (TypeError, ValueError):
        r1_state_num = 0

    eligible = r1_state_num == 1

    # ── Extract Q1 and Q2 inputs ──────────────────────────────────────────

    root_checks = payload.get("root_checks", {}) if isinstance(payload, Mapping) else {}

    q1_result = root_checks.get("Q1", {}) if isinstance(root_checks, Mapping) else {}
    q1_declared_state = q1_result.get("declared_state") if isinstance(q1_result, Mapping) else None

    q2_result = root_checks.get("Q2", {}) if isinstance(root_checks, Mapping) else {}
    q2_status = q2_result.get("status") if isinstance(q2_result, Mapping) else None
    q2_expected_activity = q2_result.get("expected_activity") if isinstance(q2_result, Mapping) else None
    q2_observed_activity = q2_result.get("observed_activity") if isinstance(q2_result, Mapping) else None

    # ── Load monthly history ──────────────────────────────────────────────

    monthly_entries = None
    if isinstance(bureaus_data, Mapping):
        monthly_block = bureaus_data.get("two_year_payment_history_monthly_tsv_v2")
        if isinstance(monthly_block, Mapping):
            candidate = monthly_block.get(bureau)
            if isinstance(candidate, list):
                monthly_entries = candidate

    # ── Compute metrics ───────────────────────────────────────────────────

    if monthly_entries is None or len(monthly_entries) == 0:
        # Monthly history missing or empty
        metrics = {
            "total_months": 0,
            "count_ok": 0,
            "count_missing": 0,
            "count_delinquent": 0,
            "has_any_activity_in_monthly": False,
            "has_only_missing": False,
        }
    else:
        total_months = len(monthly_entries)
        count_ok = 0
        count_missing = 0
        count_delinquent = 0

        for entry in monthly_entries:
            if not isinstance(entry, Mapping):
                continue
            status = entry.get("status")
            if status is None:
                continue
            status_str = str(status).strip()

            if status_str == "--":
                count_missing += 1
            elif status_str == "ok":
                count_ok += 1
            else:
                # Treat any other value as delinquency marker (e.g., "30", "60", "90", "120", etc.)
                # Only count if it looks numeric or is a known delinquency token
                try:
                    int(status_str)
                    count_delinquent += 1
                except ValueError:
                    # If not numeric and not "--" or "ok", skip (conservative)
                    pass

        has_any_activity_in_monthly = (count_ok + count_delinquent) > 0
        has_only_missing = count_missing == total_months

        metrics = {
            "total_months": total_months,
            "count_ok": count_ok,
            "count_missing": count_missing,
            "count_delinquent": count_delinquent,
            "has_any_activity_in_monthly": has_any_activity_in_monthly,
            "has_only_missing": has_only_missing,
        }

    # ── Determine status ──────────────────────────────────────────────────

    if not eligible:
        status = "skipped"
        explanation = "F2.B01 skipped: not eligible (R1.state_num must be 1; requires Q1=open and Q2=ok)"
    elif monthly_entries is None or len(monthly_entries) == 0:
        status = "unknown"
        explanation = "F2.B01 unknown: monthly history missing or empty"
    elif (
        q2_status == "ok"
        and q2_observed_activity is True
        and metrics["has_only_missing"] is True
    ):
        status = "conflict"
        explanation = (
            "F2.B01 conflict: Q2 expects activity (ok) and observed it, "
            "but monthly history is all missing (--)"
        )
    elif (
        q2_status == "skipped_missing_data"
        and metrics["has_any_activity_in_monthly"] is True
    ):
        status = "conflict"
        explanation = (
            "F2.B01 conflict: Q2 skipped (no activity expected due to missing data), "
            "but monthly history shows activity"
        )
    else:
        status = "ok"
        explanation = "F2.B01 ok: monthly history aligns with Q2 activity assessment"

    # ── Extract evidence (first 6 and last 6 months) ───────────────────────

    first_6_months = []
    last_6_months = []

    if monthly_entries is not None and len(monthly_entries) > 0:
        # First 6 months
        for entry in monthly_entries[:6]:
            if isinstance(entry, Mapping):
                first_6_months.append(
                    {
                        "month": entry.get("month"),
                        "status": entry.get("status"),
                    }
                )

        # Last 6 months
        start_idx = max(0, len(monthly_entries) - 6)
        for entry in monthly_entries[start_idx:]:
            if isinstance(entry, Mapping):
                last_6_months.append(
                    {
                        "month": entry.get("month"),
                        "status": entry.get("status"),
                    }
                )

    # ── Build result dict ─────────────────────────────────────────────────

    result = {
        "version": "f2_b01_activity_vs_monthly_history_v1",
        "status": status,
        "eligible": eligible,
        "executed": True,
        "fired": status == "conflict",
        "trigger": {
            "r1_state_num": r1_state_num,
            "q1_declared_state": q1_declared_state,
            "q2_status": q2_status,
            "q2_expected_activity": q2_expected_activity,
            "q2_observed_activity": q2_observed_activity,
        },
        "metrics": metrics,
        "evidence": {
            "first_6_months": first_6_months,
            "last_6_months": last_6_months,
        },
        "explanation": explanation,
    }

    return result
