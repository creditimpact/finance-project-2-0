"""Shared regex tokens and helpers for deterministic extractors."""

from __future__ import annotations

import re
from typing import Optional

AMOUNT_RE = re.compile(r"[-+]?\$?\d{1,3}(?:,\d{3})*(?:\.\d+)?")
# Original ISO-only matcher retained for compatibility
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
ACCOUNT_RE = re.compile(r"ac(?:count|ct)\s*(?:#|number)?[:\s]*([A-Za-z0-9]+)", re.I)

# Canonical field mappings for account level extraction
ACCOUNT_FIELD_MAP = {
    "high balance": "high_balance",
    "last verified": "last_verified",
    "date of last activity": "date_of_last_activity",
    "date reported": "date_reported",
    "date opened": "date_opened",
    "balance owed": "balance_owed",
    "closed date": "closed_date",
    "account rating": "account_rating",
    "account description": "account_description",
    "dispute status": "dispute_status",
    "creditor type": "creditor_type",
    "original creditor": "original_creditor",
    "original creditor 01": "original_creditor",
    "original creditor 02": "original_creditor",
    "orig. creditor": "original_creditor",
    "orig creditor": "original_creditor",
    "account status": "account_status",
    "payment status": "payment_status",
    "creditor remarks": "creditor_remarks",
    "payment amount": "payment_amount",
    "last payment": "last_payment",
    "term length": "term_length",
    "past due amount": "past_due_amount",
    "account type": "account_type",
    "payment frequency": "payment_frequency",
    "credit limit": "credit_limit",
    "two-year payment history": "two_year_payment_history",
    "days late": "days_late_7y",
}

SUMMARY_FIELD_MAP = {
    "total accounts": "total_accounts",
    "open accounts": "open_accounts",
    "closed accounts": "closed_accounts",
    "delinquent": "delinquent",
    "derogatory": "derogatory",
    "balances": "balances",
    "payments": "payments",
    "public records": "public_records",
    "inquiries": "inquiries_2y",
}

META_FIELD_MAP = {
    "credit report date": "credit_report_date",
    "name": "name",
    "also known as": "also_known_as",
    "date of birth": "dob",
    "current address": "current_address",
    "previous address": "previous_address",
    "employer": "employer",
}


def parse_amount(text: str) -> Optional[float | int]:
    m = AMOUNT_RE.search(text)
    if not m:
        return None
    val = m.group().replace("$", "").replace(",", "")
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return None


DATE_PATTERNS = [
    # ISO: YYYY-MM-DD
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), ("Y", "M", "D")),
    # Dots: DD.MM.YYYY or D.M.YYYY
    (re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b"), ("D", "M", "Y")),
    # Slashes: M/D/YYYY or MM/DD/YYYY
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"), ("M", "D", "Y")),
    # Hyphens: M-D-YYYY or MM-DD-YYYY
    (re.compile(r"\b(\d{1,2})-(\d{1,2})-(\d{4})\b"), ("M", "D", "Y")),
    # Spaces: M D YYYY or MM DD YYYY
    (re.compile(r"\b(\d{1,2})\s+(\d{1,2})\s+(\d{4})\b"), ("M", "D", "Y")),
]


def _to_iso(parts: dict) -> str:
    Y = int(parts["Y"])
    M = int(parts["M"])
    D = int(parts["D"])
    return f"{Y:04d}-{M:02d}-{D:02d}"


def parse_date_any(text: str) -> Optional[str]:
    """Parse various date formats and normalize to YYYY-MM-DD."""
    if not text:
        return None
    for rx, order in DATE_PATTERNS:
        m = rx.search(text)
        if m:
            parts = {k: m.group(i + 1) for i, k in enumerate(order)}
            try:
                return _to_iso(parts)
            except Exception:
                continue
    return None


def parse_date(text: str) -> Optional[str]:
    """Backward compatible helper for ISO dates."""
    return parse_date_any(text)


def normalize_issuer(text: str) -> str:
    # strip punctuation at ends, collapse whitespace, uppercase
    t = re.sub(r"\s+", " ", text.strip())
    t = t.strip(",:;-–—.()[]{}")
    return t.upper()
