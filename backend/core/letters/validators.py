from __future__ import annotations

import re
from typing import Dict, List

from backend.core.logic.utils.pii import redact_pii

CHECKLIST: Dict[str, List[str]] = {
    "dispute_letter_template.html": ["bureau"],
    "goodwill_letter_template.html": ["creditor"],
    "general_letter_template.html": ["recipient"],
    "instruction_template.html": [
        "client_name",
        "date",
        "accounts_summary",
        "per_account_actions",
    ],
    "bureau_dispute_letter_template.html": [
        "creditor_name",
        "account_number_masked",
        "bureau",
        "legal_safe_summary",
    ],
    "mov_letter_template.html": [
        "creditor_name",
        "account_number_masked",
        "legal_safe_summary",
        "cra_last_result",
        "days_since_cra_result",
    ],
    "personal_info_correction_letter_template.html": [
        "client_name",
        "client_address_lines",
        "date_of_birth",
        "ssn_last4",
        "legal_safe_summary",
    ],
}


SUBSTANCE_CHECKLIST: Dict[str, Dict[str, str | None]] = {
    "debt_validation_letter_template.html": {
        "fdcpa_1692g": r"1692g",
        "validation_window_30_day": r"30\s*day",
    },
    "fraud_dispute_letter_template.html": {
        "fcra_605b": r"605b",
        "ftc_report": r"ftc",
        "block_or_remove_request": r"block|remove",
        "response_window": r"30\s*day",
    },
    "pay_for_delete_letter_template.html": {
        "deletion_clause": r"delete|remove",
        "payment_clause": r"pay",
    },
    "mov_letter_template.html": {
        "reinvestigation_request": r"reinvestigat",
        "method_of_verification": r"method\s+of\s+verif",
        "cra_last_result": None,
        "days_since_cra_result": None,
    },
    "bureau_dispute_letter_template.html": {
        "fcra_611": r"611",
        "reinvestigation_request": r"reinvestigat",
        "account_number_masked": None,
    },
    "dispute_letter_template.html": {
        "fcra_611": r"611",
        "investigation_request": r"investigat",
        "account_number_masked": None,
        "response_window": r"30\s*day",
    },
    "goodwill_letter_template.html": {
        "non_promissory_tone": r"goodwill",
        "positive_history_reference": r"positive|good",
        "discretionary_request": r"request",
        "no_admission": r"without\s+admit",
    },
    "cease_and_desist_letter_template.html": {
        "stop_contact": r"stop\s*contact|cease|desist",
        "collector_name": None,
    },
    "personal_info_correction_letter_template.html": {
        "update_request": r"update|correct",
        "ssn_last4": None,
        "date_of_birth": None,
    },
}


def validate_required_fields(
    template_path: str | None,
    ctx: dict,
    required: List[str],
    checklist: Dict[str, List[str]],
) -> List[str]:
    """Return missing required fields for ``template_path``."""

    expected = required or checklist.get(template_path or "", [])
    missing = [field for field in expected if ctx.get(field) is None]

    sentence = ctx.get("client_context_sentence")
    if sentence:
        if len(sentence) > 150:
            missing.append("client_context_sentence.length")
        if sentence != redact_pii(sentence):
            missing.append("client_context_sentence.pii")
        if re.search(r"promise to pay", sentence, re.IGNORECASE):
            missing.append("client_context_sentence.banned")

    if template_path == "instruction_template.html":
        actions = ctx.get("per_account_actions") or []
        if not actions:
            missing.append("per_account_actions")
        else:
            for action in actions:
                if not action.get("account_ref"):
                    missing.append("per_account_actions.account_ref")
                    break
                sentence = action.get("action_sentence", "")
                if not sentence:
                    missing.append("per_account_actions.action_sentence")
                    break
                if not re.search(
                    r"\b(pay|send|contact|review|dispute|call|update|keep|monitor|mail)\b",
                    sentence,
                    re.IGNORECASE,
                ):
                    missing.append("per_account_actions.action_verb")
                    break

    return missing


def validate_substance(template_path: str, ctx: dict) -> List[str]:
    """Return missing substantive markers for ``template_path``."""

    requirements = SUBSTANCE_CHECKLIST.get(template_path, {})
    text = " ".join(str(v).lower() for v in ctx.values() if isinstance(v, str))
    missing: List[str] = []
    for key, pattern in requirements.items():
        if pattern is None:
            if not ctx.get(key):
                missing.append(key)
        else:
            if not re.search(pattern, text, re.IGNORECASE):
                missing.append(key)
    return missing


__all__ = [
    "validate_required_fields",
    "validate_substance",
    "CHECKLIST",
    "SUBSTANCE_CHECKLIST",
]
