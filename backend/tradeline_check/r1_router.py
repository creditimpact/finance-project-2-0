"""R1 — Router (4-state classifier; non-blocking, bureau-isolated).

Deterministic state labeler that reads ONLY Q1.declared_state from root_checks
and maps to a single state_id using the canonical truth table.

R1 is NOT a root check. It does not analyze raw bureau fields. It only labels
which state we are in based on Q1 output.

This module never modifies payload-level status, gate, coverage, findings, or
blocked_questions.
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

log = logging.getLogger(__name__)

# Canonical R1 truth table: 4 legal Q1-only states
# Key: q1_declared_state
# Value: (state_id, state_num)
R1_TABLE: dict[str, tuple[str, int]] = {
    "open": ("S_open", 1),
    "closed": ("S_closed", 2),
    "unknown": ("S_unknown", 3),
    "conflict": ("S_conflict", 4),
}


def evaluate_r1(root_checks: Mapping[str, Any]) -> dict:
    """Evaluate R1 router state for a single bureau.

    Reads ONLY Q1.declared_state from root_checks.
    Does NOT read raw bureau fields.

    Parameters
    ----------
    root_checks: Mapping[str, Any]
        payload["root_checks"] dictionary containing Q1 results.

    Returns
    -------
    dict
        R1 router result to be stored under payload["routing"]["R1"].
    """
    # Extract inputs from root_checks
    q1_result = root_checks.get("Q1", {}) if isinstance(root_checks, Mapping) else {}

    q1_declared_state = (
        q1_result.get("declared_state") if isinstance(q1_result, Mapping) else None
    )
    # Lookup in canonical truth table
    result = R1_TABLE.get(q1_declared_state)

    if result is None:
        # Illegal combo: not in truth table
        state_id = "S_illegal_combo"
        state_num = 0
        explanation = (
            f"R1 illegal state: observed Q1.declared_state={q1_declared_state!r} not in canonical 4-state table"
        )
        log.warning(
            "TRADELINE_CHECK_R1_ILLEGAL_STATE q1=%s",
            q1_declared_state,
        )
    else:
        # Legal state
        state_id, state_num = result
        explanation = "R1 derived from Q1.declared_state"

    return {
        "version": "r1_router_v3",
        "state_id": state_id,
        "state_num": state_num,
        "state_key": {
            "q1_declared_state": q1_declared_state,
        },
        "inputs": {
            "Q1.declared_state": q1_declared_state,
        },
        "explanation": explanation,
    }
