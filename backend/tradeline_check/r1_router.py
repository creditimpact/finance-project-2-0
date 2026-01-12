"""R1 — Router (7-state classifier; non-blocking, bureau-isolated).

Deterministic state labeler that reads ONLY Q1.declared_state and Q2.status
from root_checks and maps to a single state_id using the canonical truth table.

R1 is NOT a root check. It does not analyze raw bureau fields. It only labels
which combined state we are in based on Q1-Q2 outputs.

This module never modifies payload-level status, gate, coverage, findings, or
blocked_questions.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

log = logging.getLogger(__name__)

# Canonical R1 truth table: 7 legal (Q1, Q2) combinations
# Key: (q1_declared_state, q2_status)
# Value: (state_id, state_num)
R1_TABLE: dict[tuple[str, str], tuple[str, int]] = {
    # Group 1: Q1=open (2 Q2 outcomes)
    ("open", "ok"): ("S_open_ok", 1),
    ("open", "skipped_missing_data"): ("S_open_skipped_missing_data", 2),
    # Group 2: Q1=closed (3 Q2 outcomes)
    ("closed", "ok"): ("S_closed_ok", 3),
    ("closed", "conflict"): ("S_closed_conflict", 4),
    ("closed", "skipped_missing_data"): ("S_closed_skipped_missing_data", 5),
    # Group 3: Q1=unknown (1 Q2 outcome: unknown)
    ("unknown", "unknown"): ("S_unknown_unknown", 6),
    # Group 4: Q1=conflict (1 Q2 outcome: unknown)
    ("conflict", "unknown"): ("S_conflict_unknown", 7),
}


def evaluate_r1(root_checks: Mapping[str, Any]) -> dict:
    """Evaluate R1 router state for a single bureau.

    Reads ONLY Q1.declared_state and Q2.status from root_checks.
    Does NOT read raw bureau fields.

    Parameters
    ----------
    root_checks: Mapping[str, Any]
        payload["root_checks"] dictionary containing Q1, Q2 results.

    Returns
    -------
    dict
        R1 router result to be stored under payload["routing"]["R1"].
    """
    # Extract inputs from root_checks
    q1_result = root_checks.get("Q1", {}) if isinstance(root_checks, Mapping) else {}
    q2_result = root_checks.get("Q2", {}) if isinstance(root_checks, Mapping) else {}

    q1_declared_state = (
        q1_result.get("declared_state") if isinstance(q1_result, Mapping) else None
    )
    q2_status = q2_result.get("status") if isinstance(q2_result, Mapping) else None

    # Build state_key tuple
    state_key_tuple = (q1_declared_state, q2_status)

    # Lookup in canonical truth table
    result = R1_TABLE.get(state_key_tuple)

    if result is None:
        # Illegal combo: not in truth table
        state_id = "S_illegal_combo"
        state_num = 0
        explanation = (
            f"R1 illegal combo: observed (Q1.declared_state={q1_declared_state!r}, "
            f"Q2.status={q2_status!r}) not in canonical 7-state table"
        )
        log.warning(
            "TRADELINE_CHECK_R1_ILLEGAL_COMBO q1=%s q2=%s",
            q1_declared_state,
            q2_status,
        )
    else:
        # Legal combo
        state_id, state_num = result
        explanation = "R1 derived from (Q1.declared_state, Q2.status)"

    return {
        "version": "r1_router_v2",
        "state_id": state_id,
        "state_num": state_num,
        "state_key": {
            "q1_declared_state": q1_declared_state,
            "q2_status": q2_status,
        },
        "inputs": {
            "Q1.declared_state": q1_declared_state,
            "Q2.status": q2_status,
        },
        "explanation": explanation,
    }
