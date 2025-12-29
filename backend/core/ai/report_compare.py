"""Reporting comparison utilities for deterministic escalation."""
from __future__ import annotations

from typing import Any, Dict

from .eligibility_policy import ALWAYS_ELIGIBLE_FIELDS, CONDITIONAL_FIELDS

from .eligibility_policy import canonicalize_history


def classify_reporting_pattern(values_by_bureau: Dict[str, Any]) -> str:
    """Classify the reporting pattern among bureau values.

    Args:
        values_by_bureau: Mapping of bureau name to the raw reported value.

    Returns:
        One of ``case_1`` through ``case_6`` describing the combination of
        missing and present values along with their equality relationships.
    """

    canonicalized: Dict[str, str | None] = {
        bureau: canonicalize_history(value)
        for bureau, value in values_by_bureau.items()
    }

    present_values = [value for value in canonicalized.values() if value is not None]
    missing_count = len(values_by_bureau) - len(present_values)

    if not present_values:
        return "case_6"

    if len(present_values) == 1:
        return "case_1"

    if missing_count == 1:
        first, second = present_values
        if first == second:
            return "case_2"
        return "case_3"

    # At this point all bureaus have a value (missing_count == 0).
    unique_values = set(present_values)
    if len(unique_values) <= 2:
        return "case_4"
    return "case_5"


def compute_reason_flags(field: str, pattern: str, match_matrix: dict) -> Dict[str, bool]:
    """Compute reason flags describing the escalation rationale for *field*.

    Args:
        field: The bureau field name being evaluated.
        pattern: The reporting pattern classification (``case_1`` - ``case_6``).
        match_matrix: Matrix describing pairwise bureau value matches. Included for
            signature compatibility with upstream callers. Not used by the
            deterministic eligibility logic yet, but retained for future
            expansion.

    Returns:
        A dictionary containing boolean flags for ``missing``, ``mismatch``,
        ``both`` (when the field is both missing and mismatched across bureaus),
        and ``eligible`` (whether the field qualifies for escalation under the
        policy).
    """

    del match_matrix  # Unused in current deterministic computation.

    missing = pattern in {"case_1", "case_2", "case_3", "case_6"}
    mismatch = pattern in {"case_3", "case_4", "case_5"}
    both = missing and mismatch

    if field in ALWAYS_ELIGIBLE_FIELDS:
        eligible = missing or mismatch
    elif field in CONDITIONAL_FIELDS:
        eligible = mismatch
    else:
        eligible = False

    return {
        "missing": missing,
        "mismatch": mismatch,
        "both": both,
        "eligible": eligible,
    }
