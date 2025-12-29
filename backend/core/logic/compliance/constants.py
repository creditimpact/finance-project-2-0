# Constants and helpers used across the logic package.
from enum import Enum

# Allowed action tags used for account recommendations.
VALID_ACTION_TAGS = {
    "dispute",
    "goodwill",
    "custom_letter",
    "instruction",
    "fraud_dispute",
    "debt_validation",
    "pay_for_delete",
    "mov",
    "cease_and_desist",
    "direct_dispute",
    "ignore",
    "obsolescence",
    "bureau_dispute",
    "inquiry_dispute",
    "medical_dispute",
    "paydown_first",
}

# Map common strategist phrases to canonical action tags.
_ACTION_ALIAS_MAP = {
    "dispute": "dispute",
    "delete": "dispute",
    "remove": "dispute",
    "dispute_for_verification": "dispute",
    "challenge_the_debt": "dispute",
    "request_deletion": "dispute",
    "dispute_the_accuracy": "dispute",
    "verify_this_record": "dispute",
    "initiate_a_dispute": "dispute",
    "initiate_dispute": "dispute",
    "goodwill": "goodwill",
    "good_will": "goodwill",
    "goodwill_letter": "goodwill",
    "goodwill adjustment": "goodwill",
    "custom": "custom_letter",
    "custom_letter": "custom_letter",
    "custom letter": "custom_letter",
    "ignore": "ignore",
    "none": "ignore",
    "no_action": "ignore",
    "no action": "ignore",
    "monitor": "ignore",
    "instruction": "instruction",
    "instruction_letter": "instruction",
    "instruction letter": "instruction",
    "instructions": "instruction",
    "fraud_dispute": "fraud_dispute",
    "fraud dispute": "fraud_dispute",
    "fraud": "fraud_dispute",
    "fraudulent": "fraud_dispute",
    "identity_theft": "fraud_dispute",
    "identity theft": "fraud_dispute",
    "debt_validation": "debt_validation",
    "debt validation": "debt_validation",
    "debt_verification": "debt_validation",
    "debt verification": "debt_validation",
    "pay_for_delete": "pay_for_delete",
    "pay for delete": "pay_for_delete",
    "p4d": "pay_for_delete",
    "pay_to_delete": "pay_for_delete",
    "pay to delete": "pay_for_delete",
    "mov": "mov",
    "method_of_verification": "mov",
    "method of verification": "mov",
    "mov_letter": "mov",
    "mov letter": "mov",
    "cease_and_desist": "cease_and_desist",
    "cease and desist": "cease_and_desist",
    "stop_contact": "cease_and_desist",
    "stop contact": "cease_and_desist",
    "direct_dispute": "direct_dispute",
    "direct dispute": "direct_dispute",
    "direct_to_furnisher": "direct_dispute",
    "direct to furnisher": "direct_dispute",
    "obsolescence": "obsolescence",
    "obsolete": "obsolescence",
    "bureau_dispute": "bureau_dispute",
    "bureau dispute": "bureau_dispute",
    "cra_dispute": "bureau_dispute",
    "cra dispute": "bureau_dispute",
    "inquiry_dispute": "inquiry_dispute",
    "inquiry dispute": "inquiry_dispute",
    "medical_dispute": "medical_dispute",
    "medical dispute": "medical_dispute",
    "paydown_first": "paydown_first",
    "paydown first": "paydown_first",
    "pay_down_first": "paydown_first",
}

_DISPLAY_NAME = {
    "dispute": "Dispute",
    "goodwill": "Goodwill",
    "custom_letter": "Custom Letter",
    "instruction": "Instruction",
    "fraud_dispute": "Fraud Dispute",
    "debt_validation": "Debt Validation",
    "pay_for_delete": "Pay for Delete",
    "mov": "Method of Verification",
    "cease_and_desist": "Cease and Desist",
    "direct_dispute": "Direct Dispute",
    "ignore": "Ignore",
    "obsolescence": "Obsolescence",
    "bureau_dispute": "Bureau Dispute",
    "inquiry_dispute": "Inquiry Dispute",
    "medical_dispute": "Medical Dispute",
    "paydown_first": "Pay Down First",
}


class FallbackReason(str, Enum):
    """Reasons why a fallback dispute tag was applied."""

    KEYWORD_MATCH = "keyword_match"
    UNRECOGNIZED_TAG = "unrecognized_tag"
    NO_RECOMMENDATION = "no_recommendation"


class StrategistFailureReason(str, Enum):
    """Reasons why a strategist run failed."""

    MISSING_INPUT = "missing_input"
    SCHEMA_ERROR = "schema_error"
    EMPTY_OUTPUT = "empty_output"
    UNRECOGNIZED_FORMAT = "unrecognized_format"
    PROMPT_MISMATCH = "prompt_mismatch"


def normalize_action_tag(raw: str | None) -> tuple[str, str]:
    """Return (action_tag, recommended_action) for a strategist value.

    The returned action_tag will be one of ``VALID_ACTION_TAGS`` or an
    empty string if the value is unrecognised. ``recommended_action`` is a
    human friendly label for display purposes.
    """
    if not raw:
        return "", ""
    key = str(raw).strip().lower().replace(" ", "_")
    tag = _ACTION_ALIAS_MAP.get(key)
    if not tag:
        return "", str(raw).strip()
    return tag, _DISPLAY_NAME.get(tag, tag.title())


__all__ = [
    "VALID_ACTION_TAGS",
    "FallbackReason",
    "StrategistFailureReason",
    "normalize_action_tag",
]
