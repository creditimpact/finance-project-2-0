"""Centralized compliance pipeline for rendered documents."""

from __future__ import annotations

import re
from typing import Optional

from backend.core.logic.guardrails import fix_draft_with_guardrails
from backend.core.models.letter import LetterArtifact
from backend.core.services.ai_client import AIClient

# Re-export existing compliance helpers for compatibility
from .compliance_adapter import (
    DEFAULT_DISPUTE_REASON,
    ESCALATION_NOTE,
    adapt_gpt_output,
    sanitize_client_info,
    sanitize_disputes,
)


def run_compliance_pipeline(
    letter: LetterArtifact | str,
    state: Optional[str],
    session_id: str,
    doc_type: str,
    *,
    ai_client: AIClient,
) -> LetterArtifact | str:
    """Apply shared compliance checks to rendered HTML or artifacts."""

    if isinstance(letter, LetterArtifact):
        html = letter.html
    else:
        html = letter

    plain_text = re.sub(r"<[^>]+>", " ", html)
    fix_draft_with_guardrails(
        plain_text,
        state,
        {},
        session_id,
        doc_type,
        ai_client=ai_client,
    )
    return letter


# Backwards compatible alias
apply_text_compliance = run_compliance_pipeline

__all__ = [
    "run_compliance_pipeline",
    "apply_text_compliance",
    "adapt_gpt_output",
    "sanitize_client_info",
    "sanitize_disputes",
    "DEFAULT_DISPUTE_REASON",
    "ESCALATION_NOTE",
]
