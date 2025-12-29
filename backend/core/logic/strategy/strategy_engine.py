from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Dict, List, Mapping

from backend.api.session_manager import get_session, update_session
from backend.core.logic.guardrails.summary_validator import (
    validate_structured_summaries,
)
from backend.core.logic.letters.outcomes_store import get_outcomes

# Simple mapping of dispute types to relevant statutes and tone guidance.
# In a production system these would likely come from a dedicated rules
# service or configuration file.  For the purposes of unit testing we keep a
# lightweight, documented mapping here.
LEGAL_BASIS = {
    "late": "FCRA 611(a)",
    "identity_theft": "FCRA 605B",
    "collections": "FCRA 623(a)(3)",
    "inaccurate_reporting": "FCRA 611(a)",
}

TONE_MAP = {
    "identity_theft": "urgent and firm",
    "collections": "assertive",
    "late": "professional",
    "inaccurate_reporting": "professional",
}


def _lookup_account(account_id: str, bureau_data: Dict[str, Any]) -> Dict[str, Any]:
    """Return account details and originating bureau for ``account_id``."""

    for bureau, payload in bureau_data.items():
        for section in payload.values():
            if isinstance(section, list):
                for entry in section:
                    if str(entry.get("account_id")) == str(account_id):
                        result = dict(entry)
                        result["bureau"] = bureau
                        return result
    return {"bureau": None}


def _build_dispute_items(
    structured: Dict[str, Any], bureau_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Construct per-account dispute strategies."""

    items: List[Dict[str, Any]] = []
    for item_id, summary in structured.items():
        account_id = summary.get("account_id", item_id)
        account_info = _lookup_account(account_id, bureau_data)
        dispute_type = summary.get("dispute_type", "inaccurate_reporting")
        legal_basis = LEGAL_BASIS.get(dispute_type, "FCRA 611")
        tone = TONE_MAP.get(dispute_type, "professional")

        item = {
            "account_id": account_id,
            "bureau": account_info.get("bureau"),
            "account_name": account_info.get("name"),
            "dispute_type": dispute_type,
            "rationale": summary.get("facts_summary"),
            "legal_basis": legal_basis,
            "tone": tone,
            "next_steps": [
                "Send dispute letter to bureau",
                "Monitor response and follow up",
            ],
        }
        items.append(item)
    return items


def generate_strategy(
    session_id: str, bureau_data: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Build a comprehensive strategy document for a given session.

    The strategy combines sanitized client summaries (``structured_summaries``),
    a snapshot of the provided credit report data and recent outcome telemetry.
    Raw client explanations are intentionally excluded to prevent accidental
    leakage into any generated letters.
    """

    session = get_session(session_id) or {}
    structured = validate_structured_summaries(session.get("structured_summaries", {}))

    items = _build_dispute_items(structured, bureau_data)

    strategy: Dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dispute_items": structured,
        "bureau_data": bureau_data,
        "historical_outcomes": get_outcomes(),
        "items": items,
        "tone_guidelines": "Maintain a firm but respectful tone and avoid admissions of liability.",
        "follow_up": [
            "Review bureau responses and escalate unresolved items",
        ],
    }

    update_session(session_id, strategy=strategy)
    return strategy
