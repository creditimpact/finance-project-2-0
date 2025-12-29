"""Utilities for preparing account data for AI adjudication."""

from __future__ import annotations

import hashlib
from typing import Any, Dict

from backend.config import AI_REDACT_STRATEGY
from backend.core.logic.validation_field_sets import ALL_VALIDATION_FIELDS

_ALLOWED_FIELDS = {"normalized_name", *ALL_VALIDATION_FIELDS}


def _mask_last4(last4: str) -> str:
    if AI_REDACT_STRATEGY == "hash_last4":
        return hashlib.sha256(str(last4).encode()).hexdigest()[:8]
    if AI_REDACT_STRATEGY == "keep_last4":
        return str(last4)
    return "xxxx"


def redact_account_for_ai(account: Dict[str, Any]) -> Dict[str, Any]:
    """Return a PII-safe subset of ``account`` for AI consumption."""

    redacted: Dict[str, Any] = {}
    for field in _ALLOWED_FIELDS:
        val = account.get(field)
        if val is not None:
            redacted[field] = val
    if "normalized_name" in account:
        redacted["normalized_name"] = account["normalized_name"]
    last4 = account.get("account_number_last4")
    if last4:
        redacted["account_number_last4"] = _mask_last4(str(last4))
    presence = {
        f: account.get(f) is not None for f in _ALLOWED_FIELDS if f != "normalized_name"
    }
    redacted["field_presence_map"] = presence
    return redacted
