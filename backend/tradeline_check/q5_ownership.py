"""Q5: Ownership & Responsibility Declaration (non-blocking, bureau-isolated).

This root check answers ONLY:
  "What does the bureau explicitly declare the ownership / responsibility to be?"

Allowed Input (STRICT):
  - account_description (only this field; no other fields)

Output:
  - declared_responsibility: individual | joint | authorized_user | unknown
  - status: ok | unknown | skipped_missing_data
  - signals: list of evidence tokens (e.g., "ACCOUNT_DESCRIPTION:INDIVIDUAL")
  - evidence: raw field values
  - explanation: deterministic short string
  - confidence: 1.0 | 0.9 | 0.5

Constraints:
  - Non-blocking (never sets findings or blocked_questions)
  - Bureau-isolated (reads only from current bureau object)
  - Deterministic mapping only (no semantic inference)
"""

from __future__ import annotations

from typing import Mapping

# Allowed fields for Q5 v1
ALLOWED_FIELDS = {"account_description"}


def _is_missing(value: object, placeholders: set[str]) -> bool:
    """Return True if value is considered missing per presence-only rules.

    Parameters
    ----------
    value
        Field value to check
    placeholders
        Set of placeholder tokens (lowercase, e.g., {"--", "n/a", "unknown"})

    Returns
    -------
    bool
        True if missing, False otherwise
    """
    if value is None:
        return True
    if not isinstance(value, str):
        return False
    s = value.strip()
    if not s:
        return True
    if s.lower() in placeholders:
        return True
    return False


def _normalize_token(value: str) -> str:
    """Normalize account_description token for mapping.

    Parameters
    ----------
    value
        Raw field value

    Returns
    -------
    str
        Normalized lowercase stripped token
    """
    return value.strip().lower()


def _map_responsibility(token: str) -> tuple[str, float]:
    """Map normalized token to declared_responsibility and confidence.

    Parameters
    ----------
    token
        Normalized lowercase token

    Returns
    -------
    tuple[str, float]
        (declared_responsibility, confidence)
        - declared_responsibility: "individual" | "joint" | "authorized_user" | "unknown"
        - confidence: 1.0 (exact match) | 0.9 (conservative) | 0.5 (unknown/ambiguous)
    """
    # Exact matches (high confidence)
    if token == "individual":
        return ("individual", 1.0)
    if token == "joint":
        return ("joint", 1.0)
    if token == "authorized user":
        return ("authorized_user", 1.0)
    if token == "authorized":
        return ("authorized_user", 1.0)
    if token == "authorizeduser":
        return ("authorized_user", 1.0)

    # Conservative mapping (co-signer → joint)
    if token in ("co-signer", "cosigner"):
        return ("joint", 0.9)

    # Ambiguous cases
    if token == "terminated":
        return ("unknown", 0.5)

    # Default: unknown
    return ("unknown", 0.5)


def evaluate_q5(bureau_obj: Mapping, placeholders: set[str]) -> dict:
    """Evaluate Q5 (Ownership & Responsibility Declaration) for a single bureau.

    Parameters
    ----------
    bureau_obj
        Per-bureau object from bureaus.json (e.g., bureaus_data["equifax"])
    placeholders
        Set of placeholder tokens (lowercase, e.g., {"--", "n/a", "unknown"})

    Returns
    -------
    dict
        Q5 output block with keys:
        - version: "q5_ownership_v1"
        - status: "ok" | "unknown" | "skipped_missing_data"
        - declared_responsibility: "individual" | "joint" | "authorized_user" | "unknown"
        - signals: list of evidence tokens
        - evidence_fields: list of field names used
        - evidence: dict of raw field values
        - explanation: deterministic short string
        - confidence: float (1.0 | 0.9 | 0.5)
    """
    # Extract account_description
    account_desc = bureau_obj.get("account_description")

    # Check if missing
    if _is_missing(account_desc, placeholders):
        return {
            "version": "q5_ownership_v1",
            "status": "skipped_missing_data",
            "declared_responsibility": "unknown",
            "signals": [],
            "evidence_fields": ["account_description"],
            "evidence": {"account_description": account_desc},
            "explanation": "account_description missing",
            "confidence": 0.5,
        }

    # Normalize and map
    normalized = _normalize_token(account_desc)
    declared_responsibility, confidence = _map_responsibility(normalized)

    # Determine status
    if declared_responsibility in ("individual", "joint", "authorized_user"):
        status = "ok"
    else:
        status = "unknown"

    # Build signals
    signals = []
    if status == "ok":
        # Construct signal token
        signal_suffix = declared_responsibility.upper()
        signals.append(f"ACCOUNT_DESCRIPTION:{signal_suffix}")

    # Build explanation
    if status == "ok":
        explanation = "declared from account_description"
    else:
        explanation = "account_description ambiguous or unrecognized"

    return {
        "version": "q5_ownership_v1",
        "status": status,
        "declared_responsibility": declared_responsibility,
        "signals": signals,
        "evidence_fields": ["account_description"],
        "evidence": {"account_description": account_desc},
        "explanation": explanation,
        "confidence": confidence,
    }
