"""Helpers for redacting personally identifiable information from text.

This module provides simple pattern based masking for common PII. Emails and
phone numbers are fully redacted, while SSNs and account numbers keep their
last four digits for audit/debugging purposes.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
_SSN_RE = re.compile(r"\b(?:\d{3}[- ]\d{2}[- ]\d{4}|\d{9})\b")
# Match common "last-4" SSN patterns such as "ssn 1234" or "ssn:1234"
_SSN_LAST4_RE = re.compile(r"(?i)(ssn[\s:#-]*)(\d{4})")
# Allow account numbers with spaces or hyphens between groups of digits
_ACCOUNT_RE = re.compile(r"\b(?:\d{4}[ -]?){2,4}\d{4}\b")


def redact_pii(text: str) -> str:
    """Return ``text`` with common PII patterns masked."""
    if not text:
        return ""
    redacted = _EMAIL_RE.sub("[REDACTED]", text)
    redacted = _PHONE_RE.sub("[REDACTED]", redacted)
    redacted = _SSN_RE.sub(
        lambda m: "***-**-" + re.sub(r"\D", "", m.group())[-4:], redacted
    )
    redacted = _SSN_LAST4_RE.sub(
        lambda m: m.group(1) + "***-**-" + m.group(2), redacted
    )
    redacted = _ACCOUNT_RE.sub(
        lambda m: "****" + re.sub(r"\D", "", m.group())[-4:], redacted
    )
    return redacted


def mask_account(account_number: str) -> str:
    """Return ``account_number`` masked except for the last four digits."""

    digits = re.sub(r"\D", "", account_number)
    return "****" + digits[-4:]


def mask_account_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``data`` with account numbers and SSNs masked."""

    return json.loads(redact_pii(json.dumps(data)))
