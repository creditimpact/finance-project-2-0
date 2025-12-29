from __future__ import annotations

import re
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict

PII_REPLACEMENTS = {
    "NAME": "REDACTED_NAME",
    "SSN": "REDACTED_SSN",
    "ADDRESS": "REDACTED_ADDRESS",
    "PHONE": "REDACTED_PHONE",
    "EMAIL": "REDACTED_EMAIL",
}

ACCOUNT_REPLACEMENT = "REDACTED_ACCOUNT"

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b")
ADDRESS_RE = re.compile(
    r"\b(street|st\.|avenue|ave|road|rd\.|blvd|boulevard|apt|suite)\b", re.I
)
ACCOUNT_NUMBER_RE = re.compile(r"\d{8,}")

DATE_FORMATS = (
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%m-%d-%Y",
    "%Y.%m.%d",
    "%Y%m%d",
)


def _normalize_date(value: str) -> str:
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value).date().isoformat()
    except ValueError:
        return value


def _mask_dob(value: str) -> str:
    normalized = _normalize_date(value)
    m = re.match(r"(\d{4})", normalized)
    if m:
        return f"{m.group(1)}-**-**"
    return normalized


def _mask_account_number(value: str) -> str:
    if value.startswith("****") or value == ACCOUNT_REPLACEMENT:
        return value
    digits = re.sub(r"\D", "", value)
    if len(digits) < 4:
        return ACCOUNT_REPLACEMENT
    return f"****{digits[-4:]}"


def _redact_string(key: str | None, value: str) -> str:
    if value.startswith("REDACTED_"):
        return value

    lower_key = (key or "").lower()
    if lower_key == "account_number":
        return _mask_account_number(value)
    if lower_key in {"name", "full_name", "first_name", "last_name"}:
        return PII_REPLACEMENTS["NAME"]
    if "address" in lower_key:
        return PII_REPLACEMENTS["ADDRESS"]
    if "email" in lower_key:
        return PII_REPLACEMENTS["EMAIL"]
    if "phone" in lower_key:
        return PII_REPLACEMENTS["PHONE"]
    if "ssn" in lower_key:
        return PII_REPLACEMENTS["SSN"]
    if lower_key in {"dob", "date_of_birth"}:
        return _mask_dob(value)

    if EMAIL_RE.search(value):
        return PII_REPLACEMENTS["EMAIL"]
    if PHONE_RE.search(value):
        return PII_REPLACEMENTS["PHONE"]
    if SSN_RE.search(value):
        return PII_REPLACEMENTS["SSN"]
    if ADDRESS_RE.search(value):
        return PII_REPLACEMENTS["ADDRESS"]

    def _acc_repl(match: re.Match[str]) -> str:
        digits = match.group()
        return f"****{digits[-4:]}"

    value = ACCOUNT_NUMBER_RE.sub(_acc_repl, value)

    if any(
        token in lower_key
        for token in (
            "date",
            "last_activity",
            "last_payment",
            "last_verified",
            "opened",
            "closed",
        )
    ):
        return _normalize_date(value)

    return value


def _redact(obj: Any, key: str | None = None) -> Any:
    if isinstance(obj, dict):
        return {k: _redact(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    if isinstance(obj, str):
        return _redact_string(key, obj)
    return obj


def redact_account_fields(fields: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep-copied dict where PII is removed/masked and analytic fields preserved."""
    return _redact(deepcopy(fields))


AI_FIELD_WHITELIST = [
    "account_status",
    "payment_status",
    "creditor_remarks",
    "account_description",
    "account_rating",
    "past_due_amount",
    "balance_owed",
    "credit_limit",
    "account_type",
    "creditor_type",
    "dispute_status",
    "two_year_payment_history",
    "days_late_7y",
]


def _extract_last4(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    digits = re.sub(r"\D", "", value)
    if len(digits) >= 4:
        return digits[-4:]
    return None


def redact_for_ai(account: Dict[str, Any]) -> Dict[str, Any]:
    """Build a minimized, PII-safe AI payload."""
    fields = account.get("fields", {}) or {}
    redacted = redact_account_fields(fields)

    ai_fields = {k: redacted.get(k) for k in AI_FIELD_WHITELIST if k in redacted}
    presence_map = {k: k in fields for k in AI_FIELD_WHITELIST}
    last4 = _extract_last4(fields.get("account_number"))

    return {
        "fields": ai_fields,
        "field_presence_map": presence_map,
        "account_last4": last4,
    }


__all__ = [
    "redact_account_fields",
    "redact_for_ai",
    "PII_REPLACEMENTS",
]
