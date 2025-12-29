import re
from typing import Any, Mapping

from backend.core.logic.utils.json_utils import parse_json
from backend.core.services.ai_client import AIClient

_PROFANITY = [
    "damn",
    "shit",
    "fuck",
    "bitch",
]


def _redact(pattern: str, text: str) -> str:
    return re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)


def sanitize(text: str) -> str:
    """
    Remove unsafe content (PII, profanity, emojis, HTML) and normalize whitespace.
    """
    if not isinstance(text, str):
        return ""

    cleaned = text
    # Remove HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    # Emails, phone numbers, SSN
    cleaned = _redact(r"\b[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}\b", cleaned)
    cleaned = _redact(r"\b\d{3}[-\s]?\d{3}[-\s]?\d{4}\b", cleaned)
    cleaned = _redact(r"\b\d{3}-\d{2}-\d{4}\b", cleaned)
    # Profanity
    for word in _PROFANITY:
        cleaned = re.sub(
            rf"\b{re.escape(word)}\b", "[REDACTED]", cleaned, flags=re.IGNORECASE
        )
    # Emojis and other symbols beyond BMP
    cleaned = re.sub(r"[\U00010000-\U0010FFFF]", "", cleaned)
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


_SCHEMA = {
    "type": "object",
    "properties": {
        "account_id": {"type": "string"},
        "dispute_type": {"type": "string"},
        "facts_summary": {"type": "string"},
        "claimed_errors": {"type": "array", "items": {"type": "string"}},
        "dates": {"type": "object", "additionalProperties": {"type": "string"}},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "risk_flags": {"type": "object", "additionalProperties": {"type": "boolean"}},
    },
    "required": [
        "account_id",
        "dispute_type",
        "facts_summary",
        "claimed_errors",
        "dates",
        "evidence",
        "risk_flags",
    ],
    "additionalProperties": False,
}


def extract_structured(
    safe_text: str,
    account_ctx: Mapping[str, Any],
    ai_client: AIClient,
) -> Mapping[str, Any]:
    """
    Returns structured summary using LLM.
    """
    prompt = (
        "Paraphrase the explanation in a neutral, factual tone. "
        "Focus solely on objective details such as account names, dates, or verifiable "
        "hardship reasons. Exclude any admissions of fault, personal details, or "
        "emotional language. Do NOT quote the user's words. Return only JSON matching "
        "the provided schema."
        f"\nExplanation: {safe_text}"
        f"\nAccount context: {account_ctx}"
    )

    try:
        response = ai_client.response_json(
            prompt=prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "structured_summary", "schema": _SCHEMA},
            },
        )
        content = response.output[0].content[0].text
        data, _ = parse_json(content)
    except Exception:
        # Fallback to empty structure
        data = {}

    # Ensure defaults and include context
    result = {
        "account_id": str(account_ctx.get("account_id", "")),
        "dispute_type": account_ctx.get("dispute_type", ""),
        "facts_summary": "",
        "claimed_errors": [],
        "dates": {},
        "evidence": [],
        "risk_flags": {},
    }
    if isinstance(data, dict):
        result.update({k: data.get(k, result[k]) for k in result})
    return result
