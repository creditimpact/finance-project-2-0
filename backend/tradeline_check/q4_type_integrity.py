"""Q4 — Account Type Integrity (non-blocking, bureau-isolated).

Determines a conservative, bureau-local broad account type category using
high-precision token/phrase matching from allowed declaration fields only:
- account_type (primary)
- creditor_type (fallback)

Structural hints are context-only and never override declarations in v1.
This module does not modify payload-level status, gate, coverage, findings,
or any other root checks.
"""
from __future__ import annotations

import re
from typing import Mapping, Optional


# Allowed inputs (strict)
ALLOWED_FIELDS_CORE = ("account_type", "creditor_type")
ALLOWED_FIELDS_STRUCTURAL = (
    "term_length",
    "payment_frequency",
    "credit_limit",
    "high_balance",
    "payment_amount",
)


def _is_missing(value: object, placeholders: set[str]) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return True
        if s.lower() in placeholders:
            return True
        return False
    return False


_SEP_RE = re.compile(r"[\s\-_/.,;:]+")


def _normalize_tokens(raw: str) -> list[str]:
    """Lowercase, replace separators with spaces, collapse and tokenize."""
    s = raw.lower().strip()
    if not s:
        return []
    # Replace separators with single space
    s = _SEP_RE.sub(" ", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s.split(" ")


def _tokens_to_bucket(tokens: list[str]) -> Optional[str]:
    """Conservative token/phrase mapping to broad buckets.

    Returns one of buckets or None when no strong signal exists.
    Buckets: revolving | installment | student_loan | mortgage | collection | other
    """
    if not tokens:
        return None

    token_set = set(tokens)

    # Helper: phrase presence
    def has_phrase(words: list[str]) -> bool:
        # Require all words to be present (order-insensitive for safety)
        return all(w in token_set for w in words)

    # student_loan
    if (
        "student" in token_set
        or "education" in token_set
        or "educational" in token_set
        or has_phrase(["student", "loan"]) 
        or has_phrase(["student", "loans"]) 
    ):
        return "student_loan"

    # revolving
    brand_tokens = {"visa", "mastercard", "amex", "american", "express", "discover"}
    has_card_word = "card" in token_set
    if (
        "revolving" in token_set
        or has_phrase(["credit", "card"])
        or has_phrase(["charge", "card"])
        or has_phrase(["store", "card"]) 
        or has_phrase(["retail", "card"]) 
        or (has_card_word and (token_set & brand_tokens))
    ):
        return "revolving"

    # installment
    if (
        "installment" in token_set
        or has_phrase(["auto", "loan"]) 
        or has_phrase(["car", "loan"]) 
        or has_phrase(["vehicle", "loan"]) 
    ):
        return "installment"

    # mortgage
    if (
        "mortgage" in token_set
        or has_phrase(["home", "loan"]) 
        or "heloc" in token_set
    ):
        return "mortgage"

    # collection
    if (
        "collection" in token_set
        or "collections" in token_set
        or has_phrase(["debt", "collector"]) 
        or has_phrase(["collection", "agency"]) 
    ):
        return "collection"

    # other (intentionally conservative)
    if (
        "utility" in token_set
        or "telecom" in token_set
        or "lease" in token_set
    ):
        return "other"

    return None


def _map_bucket(raw: object, placeholders: set[str]) -> Optional[str]:
    if _is_missing(raw, placeholders):
        return None
    if not isinstance(raw, str):
        return None
    tokens = _normalize_tokens(raw)
    return _tokens_to_bucket(tokens)


def evaluate_q4(bureau_obj: Mapping[str, object], placeholders: set[str]) -> dict:
    """Evaluate Q4 for a single bureau (non-blocking)."""
    result = {
        "version": "q4_type_v1",
        "status": "unknown",
        "declared_type": "unknown",
        "signals": [],
        "conflicts": [],
        "structure_flags": [],
        "evidence_fields": [],
        "evidence": {},
        "explanation": "",
        "confidence": None,
    }

    # Collect evidence from allowed fields only (non-missing fields)
    evidence: dict[str, object] = {}
    evidence_fields: list[str] = []
    ordered_fields = list(ALLOWED_FIELDS_CORE) + list(ALLOWED_FIELDS_STRUCTURAL)
    for field in ordered_fields:
        raw_val = bureau_obj.get(field)
        if _is_missing(raw_val, placeholders):
            continue
        evidence[field] = raw_val
        evidence_fields.append(field)

    account_type_raw = bureau_obj.get("account_type")
    creditor_type_raw = bureau_obj.get("creditor_type")

    account_bucket = _map_bucket(account_type_raw, placeholders)
    creditor_bucket = _map_bucket(creditor_type_raw, placeholders)

    # Signals for transparency
    signals: list[str] = []
    if account_bucket:
        signals.append(f"ACCOUNT_TYPE:{account_bucket.upper()}")
    if creditor_bucket:
        signals.append(f"CREDITOR_TYPE:{creditor_bucket.upper()}")

    # Decide status
    core_missing = (
        _is_missing(account_type_raw, placeholders) and _is_missing(creditor_type_raw, placeholders)
    )

    if core_missing:
        status = "skipped_missing_data"
        declared_type = "unknown"
        confidence = 0.0
        explanation = "Q4 skipped: both declaration fields missing"
    elif account_bucket and creditor_bucket and account_bucket != creditor_bucket:
        status = "conflict"
        declared_type = "conflict"
        confidence = 1.0
        explanation = "type mismatch between account_type and creditor_type"
    elif account_bucket:
        status = "ok"
        declared_type = account_bucket
        confidence = 1.0
        explanation = "declared from account_type"
    elif creditor_bucket:
        status = "ok"
        declared_type = creditor_bucket
        confidence = 1.0
        explanation = "declared from creditor_type"
    else:
        status = "unknown"
        declared_type = "unknown"
        confidence = 0.5
        explanation = "no strong type signals detected"

    # Conflicts list
    conflicts: list[str] = []
    if status == "conflict":
        conflicts.append("type_mismatch_account_vs_creditor")

    # Structure flags (context-only)
    structure_flags: list[str] = []
    declared = declared_type
    # Helper missing checks for structural fields
    def is_miss(field: str) -> bool:
        return _is_missing(bureau_obj.get(field), placeholders)

    if declared == "revolving":
        if is_miss("credit_limit") and is_miss("high_balance"):
            structure_flags.append("revolving_missing_limit_and_balance")

    if declared in {"installment", "student_loan", "mortgage"}:
        if is_miss("term_length") and is_miss("payment_amount") and is_miss("payment_frequency"):
            structure_flags.append("installment_missing_terms_and_payment")

    result.update(
        {
            "status": status,
            "declared_type": declared_type,
            "signals": signals,
            "conflicts": conflicts,
            "structure_flags": structure_flags,
            "evidence_fields": evidence_fields,
            "evidence": evidence,
            "explanation": explanation,
            "confidence": confidence,
        }
    )

    return result
