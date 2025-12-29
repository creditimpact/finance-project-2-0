"""Utilities for classifying bureau disagreement scenarios.

This module provides helpers that summarize the shape of a field's bureau
responses.  The primary entry point is :func:`classify_reason`, which accepts a
mapping of bureau identifiers to their values (raw or normalized) and
determines the appropriate escalation reason code.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping

__all__ = ["classify_reason", "decide_send_to_ai"]


_REASON_LABELS: Dict[str, str] = {
    "C1_TWO_PRESENT_ONE_MISSING": "two present, one missing",
    "C2_ONE_MISSING": "only one bureau reported a value",
    "C3_TWO_PRESENT_CONFLICT": "conflict with one bureau missing",
    "C4_TWO_MATCH_ONE_DIFF": "two bureaus agree, one differs",
    "C5_ALL_DIFF": "all bureaus reported different values",
    "C6_ALL_MISSING": "all bureaus missing value",
}


def _is_missing(value: Any) -> bool:
    """Return ``True`` when ``value`` should be treated as missing."""

    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in {"", "--"}
    return False


def _freeze(value: Any) -> Any:
    """Convert ``value`` into a hashable representation for comparisons."""

    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(item) for item in value)
    return value


def classify_reason(bureau_values: Mapping[str, Any]) -> Mapping[str, Any]:
    """Classify the disagreement pattern for ``bureau_values``.

    Parameters
    ----------
    bureau_values:
        Mapping of bureau identifiers (e.g., ``"experian"``) to the values they
        reported for a field. Values may be raw or normalized representations.

    Returns
    -------
    Mapping[str, Any]
        A dictionary containing the reason code, metadata about missing vs
        mismatch counts, and helper booleans that downstream callers can use.
    """

    normalized_values: Dict[str, Any] = {}

    for bureau, value in bureau_values.items():
        if isinstance(value, str):
            value = value.strip()
            if value in {"", "--"}:
                value = None
        else:  # pragma: no branch - convert other sentinel markers when possible
            try:
                if value in {"", "--"}:  # type: ignore[operator]
                    value = None
            except TypeError:  # pragma: no cover - unhashable values
                pass

        normalized_values[bureau] = value

    total_bureaus = len(normalized_values)
    missing_count = 0
    present_values: list[Any] = []

    for value in normalized_values.values():
        if _is_missing(value):
            missing_count += 1
        else:
            present_values.append(value)

    present_count = total_bureaus - missing_count
    distinct_values = len({_freeze(value) for value in present_values})

    is_missing = missing_count > 0
    is_mismatch = distinct_values > 1

    if present_count == 0:
        reason_code = "C6_ALL_MISSING"
    elif present_count == 1:
        reason_code = "C2_ONE_MISSING"
    elif missing_count > 0:
        if distinct_values <= 1:
            reason_code = "C1_TWO_PRESENT_ONE_MISSING"
        else:
            reason_code = "C3_TWO_PRESENT_CONFLICT"
    else:
        if distinct_values <= 1:
            # This scenario should already be filtered out before classification,
            # but fall back to C4 for completeness.
            reason_code = "C4_TWO_MATCH_ONE_DIFF"
        elif distinct_values == 2:
            reason_code = "C4_TWO_MATCH_ONE_DIFF"
        else:
            reason_code = "C5_ALL_DIFF"

    return {
        "reason_code": reason_code,
        "reason_label": _REASON_LABELS[reason_code],
        "is_missing": is_missing,
        "is_mismatch": is_mismatch,
        "missing_count": missing_count,
        "present_count": present_count,
        "distinct_values": distinct_values,
    }


_AI_FIELDS = {"account_type", "creditor_type", "account_rating"}
_AI_ALLOWED_REASONS = {"C3", "C4", "C5"}


def decide_send_to_ai(field: Any, reason: Any) -> bool:
    """Return ``True`` when ``field`` should be escalated to AI adjudication.

    Parameters
    ----------
    field:
        The field identifier under review. Only a small subset of free-text
        fields are eligible for AI review.
    reason:
        The escalation reason. This may be a reason code string such as
        ``"C3_TWO_PRESENT_CONFLICT"`` or a mapping containing a ``reason_code``
        entry (for compatibility with :func:`classify_reason` output).
    """

    if not isinstance(field, str):
        return False

    normalized_field = field.strip().lower()
    if normalized_field not in _AI_FIELDS:
        return False

    reason_code: str | None = None
    if isinstance(reason, str):
        reason_code = reason.strip()
    elif isinstance(reason, Mapping):
        for key in ("reason_code", "code", "reason"):
            candidate = reason.get(key)
            if isinstance(candidate, str):
                reason_code = candidate.strip()
                break

    if not reason_code:
        return False

    prefix = reason_code.upper().split("_", 1)[0]
    return prefix in _AI_ALLOWED_REASONS

