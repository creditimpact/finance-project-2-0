"""Output schema and constants for tradeline_check findings."""

from __future__ import annotations

SCHEMA_VERSION = 1
SUPPORTED_BUREAUS = {"equifax", "experian", "transunion"}


def bureau_output_template(
    account_key: str,
    bureau: str,
    generated_at: str,
) -> dict:
    """Return a minimal per-bureau output template."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "account_key": account_key,
        "bureau": bureau,
        "status": "ok",
        "findings": [],
        "blocked_questions": [],
        "notes": "tradeline_check scaffold v1 (no analysis yet)",
    }
