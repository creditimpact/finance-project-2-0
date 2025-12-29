"""Validation helpers for AI adjudication payloads."""

from __future__ import annotations

DECISIONS = {
    "same_account_same_debt",
    "same_account_diff_debt",
    "same_account_debt_unknown",
    "same_debt_diff_account",
    "same_debt_account_unknown",
    "different",
    "duplicate",
    "not_duplicate",
}


def validate_ai_result(obj: dict) -> tuple[bool, str | None]:
    """Return ``(True, None)`` when ``obj`` is a valid AI result payload."""

    try:
        d = obj["decision"]
        flags = obj["flags"]
        if d in {"duplicate", "not_duplicate"} or "duplicate" in flags:
            dup_raw = flags.get("duplicate")
            if isinstance(dup_raw, bool):
                dup_val = dup_raw
            elif isinstance(dup_raw, str):
                lowered = dup_raw.strip().lower()
                if lowered in {"true", "1", "yes", "duplicate"}:
                    dup_val = True
                elif lowered in {"false", "0", "no", "not_duplicate"}:
                    dup_val = False
                else:
                    return False, "flags_duplicate_invalid"
            else:
                return False, "flags_duplicate_invalid"
            if d == "duplicate" and not dup_val:
                return False, "flags_inconsistent"
            if d == "not_duplicate" and dup_val:
                return False, "flags_inconsistent"
            return True, None

        am = flags.get("account_match")
        dm = flags.get("debt_match")
        ok = (
            (d == "same_account_same_debt" and am is True and dm is True)
            or (d == "same_account_diff_debt" and am is True and dm is False)
            or (d == "same_account_debt_unknown" and am is True and dm == "unknown")
            or (d == "same_debt_diff_account" and am is False and dm is True)
            or (d == "same_debt_account_unknown" and am == "unknown" and dm is True)
            or (d == "different" and am is False and dm is False)
        )
        if d not in DECISIONS:
            return False, "decision_outside_contract"
        if not ok:
            return False, "flags_inconsistent"
        return True, None
    except Exception as exc:  # pragma: no cover - defensive
        return False, f"schema_error:{exc}"

