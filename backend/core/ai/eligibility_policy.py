"""Eligibility policy helpers for deterministic field escalation."""
from __future__ import annotations

from typing import Any, Iterable
import re
import unicodedata

ALWAYS_ELIGIBLE_FIELDS: set[str] = {
    "date_opened",
    "closed_date",
    "account_type",
    "creditor_type",
    "high_balance",
    "credit_limit",
    "term_length",
    "payment_amount",
    "payment_frequency",
    "balance_owed",
    "last_payment",
    "past_due_amount",
    "date_of_last_activity",
    "account_status",
    "payment_status",
    "date_reported",
    "two_year_payment_history",
    "seven_year_history",
}

CONDITIONAL_FIELDS: set[str] = {
    "account_rating",
    "account_number_display",
}

ALL_POLICY_FIELDS: set[str] = ALWAYS_ELIGIBLE_FIELDS | CONDITIONAL_FIELDS

_PUNCTUATION_NOISE = ",.;:!?"
_WHITESPACE_RE = re.compile(r"\s+")


def is_missing(value: Any) -> bool:
    """Return True when *value* should be considered missing."""

    if value is None:
        return True

    if isinstance(value, str):
        if value.strip() == "":
            return True
        if value.strip() == "--":
            return True
        return False

    return False


def _normalize_string(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    normalized = normalized.strip()
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    if not normalized:
        return normalized
    # Replace punctuation noise with spaces and collapse again.
    normalized = normalized.translate({ord(char): " " for char in _PUNCTUATION_NOISE})
    normalized = _WHITESPACE_RE.sub(" ", normalized)
    return normalized.strip()


def canonicalize_scalar(value: Any) -> str | None:
    """Normalize scalar values for comparison."""

    if is_missing(value):
        return None

    if isinstance(value, (int, float)):
        text = str(value)
    else:
        text = str(value)

    normalized = _normalize_string(text)
    return normalized or None


def _canonicalize_iterable(values: Iterable[Any]) -> list[str]:
    normalized_values: list[str] = []
    for value in values:
        normalized = canonicalize_scalar(value)
        normalized_values.append(normalized or "")
    return normalized_values


def canonicalize_history(value: Any) -> str | None:
    """Canonicalize history fields for deterministic comparison."""

    if value is None:
        return None

    # Handle two-year payment history formats (list or dict)
    if isinstance(value, list):
        normalized_values = _canonicalize_iterable(value)
        combined = "|".join(normalized_values)
        return combined or None

    if isinstance(value, dict):
        items = []
        for key in sorted(value):
            normalized_key = canonicalize_scalar(key)
            normalized_value = canonicalize_scalar(value[key])
            if normalized_key is None:
                continue
            items.append(f"{normalized_key}={normalized_value or ''}")
        combined = ";".join(items)
        return combined or None

    # Fallback: treat as scalar string representation
    normalized_scalar = canonicalize_scalar(value)
    return normalized_scalar

