"""Q1 — Account State Declaration (non-blocking, bureau-isolated).

Evaluates the bureau-declared account state from strictly allowed fields:
- account_status
- account_rating
- payment_status

Presence-only normalization (no semantics). Detects signals OPEN, CLOSED, DEROG.
DEROG never overrides OPEN/CLOSED. Declared state: open|closed|unknown|conflict.

This module does not read any fields beyond the allowed set and does not
modify payload-level status, findings, gate, or coverage.
"""

from __future__ import annotations

from typing import Mapping

ALLOWED_FIELDS = ("account_status", "account_rating", "payment_status")


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


def _normalize_to_signals(raw: str) -> set[str]:
    """Map a raw string to state/behavior signals.

    - OPEN when token contains "open"/"opened"
    - CLOSED when token contains "closed" or exact "paid/closed"
    - DEROG when token contains any of common negative indicators
      (charge-off, collection, delinquent, late, past due, write-off, etc.)

    Deterministic substring checks; no fuzzy inference.
    """
    signals: set[str] = set()
    s = raw.strip().lower()
    if not s:
        return signals

    # State signals
    if "open" in s:
        signals.add("OPEN")
    if "closed" in s or s == "paid/closed":
        signals.add("CLOSED")

    # Behavior-only signals (never override state)
    derog_terms = (
        "charge-off",
        "charge off",
        "collection",
        "collections",
        "delinquent",
        "late",
        "past due",
        "write-off",
        "written off",
        "repossession",
        "foreclosure",
    )
    for term in derog_terms:
        if term in s:
            signals.add("DEROG")
            break

    return signals


def evaluate_q1(bureau_obj: Mapping[str, object], placeholders: set[str]) -> dict:
    """Evaluate Q1 for a single bureau.

    Parameters
    ----------
    bureau_obj: Mapping[str, object]
        Bureau-local object (e.g., bureaus.json["equifax"]) strictly used.
    placeholders: set[str]
        Lowercased placeholder tokens configured in environment.

    Returns
    -------
    dict
        Q1 result dictionary to be stored under payload["root_checks"]["Q1"].
    """
    # Collect raw evidence from allowed fields only
    evidence: dict[str, object] = {}
    non_missing_fields: list[str] = []

    # Aggregate signals from allowed fields
    open_supporters: set[str] = set()
    closed_supporters: set[str] = set()
    derog_present = False

    for field in ALLOWED_FIELDS:
        raw_val = bureau_obj.get(field)
        # Always record raw evidence (allowed fields only)
        if raw_val is not None:
            evidence[field] = raw_val
        else:
            evidence[field] = None

        # Skip missing/placeholder values for signal evaluation
        if _is_missing(raw_val, placeholders):
            continue

        non_missing_fields.append(field)
        if isinstance(raw_val, str):
            sigs = _normalize_to_signals(raw_val)
            if "OPEN" in sigs:
                open_supporters.add(field)
            if "CLOSED" in sigs:
                closed_supporters.add(field)
            if "DEROG" in sigs:
                derog_present = True

    # Decide declared state and status
    declared_state = "unknown"
    status = "ok"

    if open_supporters and not closed_supporters:
        declared_state = "open"
        status = "ok"
    elif closed_supporters and not open_supporters:
        declared_state = "closed"
        status = "ok"
    elif open_supporters and closed_supporters:
        declared_state = "conflict"
        status = "conflict"
    else:
        # Neither OPEN nor CLOSED present
        if derog_present:
            declared_state = "unknown"
            status = "ok"
        else:
            # If all allowed fields are missing -> skipped_missing_data
            if len(non_missing_fields) == 0:
                status = "skipped_missing_data"
            else:
                status = "ok"
            declared_state = "unknown"

    # Compute confidence
    total_non_missing = len(non_missing_fields)
    if total_non_missing == 0:
        confidence = 0.0
    else:
        if declared_state == "open":
            supporters = len(open_supporters)
        elif declared_state == "closed":
            supporters = len(closed_supporters)
        else:
            supporters = 0
        confidence = round(float(supporters) / float(total_non_missing), 4)

    # Build signals list (union of detected state/behavior signals)
    signals: list[str] = []
    if open_supporters:
        signals.append("OPEN")
    if closed_supporters:
        signals.append("CLOSED")
    if derog_present:
        signals.append("DEROG")

    # Contributing fields are the non-missing fields that supported the selected state
    contributing_fields: list[str] = []
    if declared_state == "open":
        contributing_fields = sorted(list(open_supporters))
    elif declared_state == "closed":
        contributing_fields = sorted(list(closed_supporters))
    else:
        contributing_fields = []

    explanation_parts: list[str] = []
    explanation_parts.append(f"declared_state={declared_state}")
    if signals:
        explanation_parts.append(f"signals={','.join(signals)}")
    if contributing_fields:
        explanation_parts.append(f"from={','.join(contributing_fields)}")
    explanation = "; ".join(explanation_parts) if explanation_parts else "no signals found"

    return {
        "version": "q1_state_v1",
        "status": status,
        "declared_state": declared_state,
        "signals": signals,
        "contributing_fields": contributing_fields,
        "evidence": evidence,
        "explanation": explanation,
        "confidence": confidence,
    }
