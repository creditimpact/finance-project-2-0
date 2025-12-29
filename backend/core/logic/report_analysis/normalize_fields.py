from __future__ import annotations

from typing import Dict, List
import re

_DASH_CHARS = "-\u2010\u2011\u2012\u2013\u2014\u2015"
_DASH_PLACEHOLDER_RE = re.compile(rf"^[{_DASH_CHARS}\s]+$")


# Canonical label list in visual order (22)
LABELS: List[str] = [
    "Account #",
    "High Balance",
    "Last Verified",
    "Date of Last Activity",
    "Date Reported",
    "Date Opened",
    "Balance Owed",
    "Closed Date",
    "Account Rating",
    "Account Description",
    "Dispute Status",
    "Creditor Type",
    "Original Creditor",
    "Account Status",
    "Payment Status",
    "Creditor Remarks",
    "Payment Amount",
    "Last Payment",
    "Term Length",
    "Past Due Amount",
    "Account Type",
    "Payment Frequency",
    "Credit Limit",
]

LABEL_TO_KEY: Dict[str, str] = {
    "Account #": "account_number_display",
    "High Balance": "high_balance",
    "Last Verified": "last_verified",
    "Date of Last Activity": "date_of_last_activity",
    "Date Reported": "date_reported",
    "Date Opened": "date_opened",
    "Balance Owed": "balance_owed",
    "Closed Date": "closed_date",
    "Account Rating": "account_rating",
    "Account Description": "account_description",
    "Dispute Status": "dispute_status",
    "Creditor Type": "creditor_type",
    "Original Creditor": "original_creditor",
    "Original Creditor 01": "original_creditor",
    "Original Creditor 02": "original_creditor",
    "Orig. Creditor": "original_creditor",
    "Account Status": "account_status",
    "Payment Status": "payment_status",
    "Creditor Remarks": "creditor_remarks",
    "Payment Amount": "payment_amount",
    "Last Payment": "last_payment",
    "Term Length": "term_length",
    "Past Due Amount": "past_due_amount",
    "Account Type": "account_type",
    "Payment Frequency": "payment_frequency",
    "Credit Limit": "credit_limit",
}

CANONICAL_KEYS: List[str] = [LABEL_TO_KEY[lbl] for lbl in LABELS]


def is_dash_placeholder(val: str | None) -> bool:
    if not isinstance(val, str):
        return False
    stripped = val.strip()
    if not stripped:
        return False
    return bool(_DASH_PLACEHOLDER_RE.fullmatch(stripped))


def is_effectively_blank(val: str | None) -> bool:
    if val is None:
        return True
    if isinstance(val, str):
        stripped = val.strip()
    else:
        stripped = str(val).strip()
    if not stripped:
        return True
    return is_dash_placeholder(stripped)


def clean_value(val: str | None) -> str:
    raw = "" if val is None else str(val)
    if not raw:
        return ""

    if "*" in raw:
        # Preserve masked account patterns verbatim aside from whitespace cleanup
        return re.sub(r"\s+", " ", raw).strip()

    s = raw.strip()
    if not s:
        return ""
    if is_dash_placeholder(s):
        return "--"
    # Keep dates/amount formats as found; just normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def join_parts(parts: List[str]) -> str:
    # Join with a single space and normalize whitespace
    joined = " ".join(p for p in parts if p)
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


def ensure_all_keys(d: Dict[str, str]) -> Dict[str, str]:
    out = dict(d or {})
    for k in CANONICAL_KEYS:
        out.setdefault(k, "")
    return out


__all__ = [
    "LABELS",
    "LABEL_TO_KEY",
    "CANONICAL_KEYS",
    "is_dash_placeholder",
    "is_effectively_blank",
    "clean_value",
    "join_parts",
    "ensure_all_keys",
]

