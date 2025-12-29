"""Account number normalization utilities for merge scoring."""

from __future__ import annotations

import re


def normalize_acctnum(raw: str | None) -> str | None:
    """Normalize an account number by removing separators and mask characters."""

    if not raw:
        return None
    s = re.sub(r"[\s\-]", "", raw)
    s = s.replace("*", "")
    return s or None


def last4(s: str | None) -> str | None:
    """Return the last four digits from the provided string, if available."""

    if not s:
        return None
    digits = re.sub(r"\D", "", s)
    return digits[-4:] if len(digits) >= 4 else None


__all__ = ["normalize_acctnum", "last4"]
