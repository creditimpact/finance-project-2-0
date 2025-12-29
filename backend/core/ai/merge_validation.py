"""Guardrails for AI merge decisions."""

from __future__ import annotations

from collections.abc import Mapping

ACCOUNT_STEM_GUARDRAIL_REASON = (
    "Overridden by guardrail: account stems conflict (cannot be same account)."
)

_TRUE_VALUES = {"true", "t", "1", "yes", "y"}
_FALSE_VALUES = {"false", "f", "0", "no", "n"}


def _coerce_bool(value: object) -> bool | None:
    """Best-effort conversion of configuration flags to booleans."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_VALUES:
            return True
        if lowered in _FALSE_VALUES:
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 0:
            return False
        if value == 1:
            return True
    return None


def apply_same_account_guardrail(
    pack: Mapping[str, object], payload: Mapping[str, object]
) -> tuple[dict[str, object], str | None]:
    """Prevent "same_account_*" decisions when account numbers conflict.

    Parameters
    ----------
    pack:
        The pack sent to the adjudicator, expected to contain ``context_flags``.
    payload:
        The adjudicator response payload to validate.

    Returns
    -------
    tuple[dict[str, object], str | None]
        A tuple ``(updated_payload, reason)``. ``reason`` is ``None`` when no
        guardrail was applied; otherwise it contains the override reason string.
    """

    decision_raw = payload.get("decision")
    decision = str(decision_raw).strip().lower() if decision_raw is not None else ""
    if not decision.startswith("same_account_"):
        return dict(payload), None

    context_flags = pack.get("context_flags")
    if not isinstance(context_flags, Mapping):
        return dict(payload), None

    acctnum_conflict = _coerce_bool(context_flags.get("acctnum_conflict"))
    acct_stem_equal = _coerce_bool(context_flags.get("acct_stem_equal"))

    if not ((acctnum_conflict is True) or (acct_stem_equal is False)):
        return dict(payload), None

    updated_payload = dict(payload)
    updated_payload["decision"] = "different"
    updated_payload["reason"] = ACCOUNT_STEM_GUARDRAIL_REASON

    flags_raw = updated_payload.get("flags")
    flags: dict[str, object]
    if isinstance(flags_raw, Mapping):
        flags = dict(flags_raw)
    else:
        flags = {}
    flags["account_match"] = False
    flags["debt_match"] = False
    updated_payload["flags"] = flags
    updated_payload["normalized"] = True

    return updated_payload, ACCOUNT_STEM_GUARDRAIL_REASON


__all__ = ["apply_same_account_guardrail", "ACCOUNT_STEM_GUARDRAIL_REASON"]
