"""AI adjudicator client for calling OpenAI's chat completion API."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Sequence

import httpx

from backend.core.ai import PROJECT_HEADER_NAME, auth_probe, build_openai_headers


log = logging.getLogger(__name__)

_AUTH_READY = False
_AUTH_LOCK = threading.Lock()


def _key_prefix(value: str) -> str:
    if not value:
        return "<empty>"
    if value.startswith("sk-proj-"):
        return "sk-proj..."
    if value.startswith("sk-"):
        return "sk-..."
    if len(value) <= 4:
        return f"{value}..."
    return f"{value[:4]}..."

_SYSTEM_PROMPT = """You are a meticulous adjudicator for credit-report account pairing.
Decide if two account entries (A,B) refer to the SAME underlying account.

Allowed decisions (exact strings, choose one):
- same_account_same_debt        # accounts align and refer to the same debt
- same_account_diff_debt        # same account, but debt details clearly differ
- same_account_debt_unknown     # same account, debt status cannot be confirmed
- same_debt_diff_account        # same debt, but reported under a different account
- same_debt_account_unknown     # same debt, account identity cannot be confirmed
- different                     # neither the account nor the debt matches
Legacy labels (merge, same_debt, same_debt_account_diff, same_account,
same_account_debt_diff, different) may appear in reference material, but
you MUST respond using only the six decisions above.

Always output strict JSON matching the contract below (no prose around it):
{
  "decision": "one of the six allowed decisions above",
  "flags": {"account_match": true|false|"unknown", "debt_match": true|false|"unknown"},
  "reason": "short natural language"
}

Decision guidance:
- Flags.account_match=true when normalized account numbers (exact or last4 corroborated by lender+dates) align.
- Flags.debt_match=true when balances/high_balance/past_due + timing align within tolerance; false when they conflict.
- If account_match=true and debt_match=true → same_account_same_debt.
- If account_match=true and debt_match=false → same_account_diff_debt.
- If account_match=false and debt_match=true → same_debt_diff_account when tradelines clearly differ; otherwise lean conservative.
- If account_match=true and debt_match="unknown" → same_account_debt_unknown.
- If debt_match=true and account_match="unknown" → same_debt_account_unknown.
- If either flag is "unknown" be conservative and avoid *_diff decisions unless evidence is explicit.
- If both flags are false → different.

When BOTH sides are collection agencies:
• If amounts_equal_within_tol is true for positive debt and dates_plausible_chain is true, prefer "same_debt_diff_account" over "different".
• If amounts match but timing is ambiguous, respond with "same_debt_account_unknown".
• Do NOT choose "different" solely because account numbers or lender names differ.

Allowed outputs: same_account_same_debt, same_account_diff_debt, same_account_debt_unknown, same_debt_diff_account, same_debt_account_unknown, different.

Consider:
• High-precision cues: account-number (last4/exact), balance owed equality within tolerances, date alignments.
• Lender names/brands and free-text descriptors from the raw “context” lines.
• The numeric 0–100 match summary as a hint, but override if raw context contradicts it.
Be conservative: if critical fields conflict without plausible explanation → "different".
Do NOT mention these rules in the output."""

SYSTEM_PROMPT_SHA256 = hashlib.sha256(_SYSTEM_PROMPT.encode("utf-8")).hexdigest()

REQUEST_PARAMS: Dict[str, object] = {
    "temperature": 0,
    "top_p": 1,
}

RESPONSE_FORMAT: Dict[str, object] = {"type": "json_object"}

_ALLOWED_USER_PAYLOAD_KEYS: Iterable[str] = (
    "sid",
    "pair",
    "numeric_match_summary",
    "tolerances_hint",
    "ids",
    "context",
)


ALLOWED_DECISIONS: set[str] = {
    "same_account_same_debt",
    "same_account_diff_debt",
    "same_account_debt_unknown",
    "same_debt_diff_account",
    "same_debt_account_unknown",
    "different",
    "duplicate",
    "not_duplicate",
}

ALLOWED_FLAGS_ACCOUNT: set[str] = {"true", "false", "unknown"}
ALLOWED_FLAGS_DEBT: set[str] = {"true", "false", "unknown"}


def _coerce_flag(value: Any) -> str:
    """Return a lowercase flag string from a boolean or string value."""

    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).strip().lower()


def validate_ai_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate and normalize a raw AI payload against the decision contract."""

    decision = str(payload.get("decision", "")).strip()
    flags_raw = payload.get("flags", {})
    if not isinstance(flags_raw, Mapping):
        raise AdjudicatorError("Flags outside contract")

    duplicate_flag: bool | None = None
    if "duplicate" in flags_raw:
        duplicate_flag = _normalize_duplicate_flag(flags_raw.get("duplicate"))

    if decision in {"duplicate", "not_duplicate"} or duplicate_flag is not None:
        inferred_duplicate = decision == "duplicate"
        if duplicate_flag is None:
            duplicate_flag = inferred_duplicate
        elif decision in {"duplicate", "not_duplicate"} and duplicate_flag != inferred_duplicate:
            raise AdjudicatorError("flags.duplicate disagrees with decision")

        reason_value = payload.get("reason")
        if not isinstance(reason_value, str) or not reason_value.strip():
            raise AdjudicatorError("AI adjudicator response must include a reason string")
        normalized_reason = reason_value.strip()
        normalized_decision = "duplicate" if duplicate_flag else "not_duplicate"
        normalized_flags: Dict[str, Any] = {"duplicate": duplicate_flag}
        return {
            "decision": normalized_decision,
            "flags": normalized_flags,
            "reason": normalized_reason,
        }

    if decision not in ALLOWED_DECISIONS:
        raise AdjudicatorError(f"Decision outside contract: {decision!r}")

    account_flag = _coerce_flag(flags_raw.get("account_match", "unknown"))
    debt_flag = _coerce_flag(flags_raw.get("debt_match", "unknown"))

    if account_flag not in ALLOWED_FLAGS_ACCOUNT or debt_flag not in ALLOWED_FLAGS_DEBT:
        raise AdjudicatorError("Flags outside contract")

    normalized_flags: Dict[str, Any] = {
        "account_match": account_flag,
        "debt_match": debt_flag,
    }
    if duplicate_flag is not None:
        normalized_flags["duplicate"] = "true" if duplicate_flag else "false"

    return {
        "decision": decision,
        "flags": normalized_flags,
        "reason": payload.get("reason"),
    }


class AdjudicatorError(ValueError):
    """Raised when the AI adjudicator response is malformed."""


def _normalize_match_flag(value: object, *, field: str) -> bool | str:
    """Return a normalized boolean/"unknown" flag from ``value``."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        allowed_values = (
            ALLOWED_FLAGS_ACCOUNT if field == "account_match" else ALLOWED_FLAGS_DEBT
        )
        if lowered not in allowed_values:
            raise AdjudicatorError(
                f"AI adjudicator flags must set {field} to true, false, or \"unknown\""
            )
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "unknown":
            return "unknown"
    raise AdjudicatorError(
        f"AI adjudicator flags must set {field} to true, false, or \"unknown\""
    )


def _normalize_duplicate_flag(value: object) -> bool:
    """Return a normalized boolean from ``flags.duplicate``."""

    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "duplicate"}:
            return True
        if lowered in {"false", "0", "no", "not_duplicate"}:
            return False
    raise AdjudicatorError("AI adjudicator flags.duplicate must be true or false")


def _decision_for_flags(
    account_flag: bool | str,
    debt_flag: bool | str,
    *,
    requested: str,
) -> str:
    """Return the contract-compliant decision for the provided flags."""

    normalized_requested = requested.strip().lower()
    legacy_map = {
        "merge": "same_account_same_debt",
        "same_account": "same_account_debt_unknown",
        "same_account_debt_different": "same_account_diff_debt",
        "same_account_debt_diff": "same_account_diff_debt",
        "same_debt": "same_debt_account_unknown",
        "same_debt_account_different": "same_debt_diff_account",
        "same_debt_account_diff": "same_debt_diff_account",
    }
    requested_normalized = legacy_map.get(normalized_requested, normalized_requested)

    if account_flag is True and debt_flag is True:
        return "same_account_same_debt"
    if account_flag is True and debt_flag is False:
        return "same_account_diff_debt"
    if account_flag is False and debt_flag is True:
        return "same_debt_diff_account"
    if account_flag is True and debt_flag == "unknown":
        return "same_account_debt_unknown"
    if account_flag == "unknown" and debt_flag is True:
        return "same_debt_account_unknown"
    if account_flag is False and debt_flag == "unknown":
        return "different"
    if account_flag == "unknown" and debt_flag is False:
        return "different"
    if account_flag == "unknown" and debt_flag == "unknown":
        return "different"
    if account_flag is False and debt_flag is False:
        return "different"
    if requested_normalized in ALLOWED_DECISIONS:
        return requested_normalized
    return "different"


def _normalize_and_validate_decision(
    resp: Mapping[str, Any]
) -> tuple[Dict[str, Any], bool]:
    """Return the normalized decision payload and whether normalization occurred."""

    if not isinstance(resp, Mapping):
        raise AdjudicatorError("AI adjudicator response payload must be an object")

    decision_raw = resp.get("decision")
    if not isinstance(decision_raw, str) or not decision_raw.strip():
        raise AdjudicatorError("AI adjudicator decision must be a non-empty string")
    decision_value = decision_raw.strip().lower()

    reason_raw = resp.get("reason")
    if not isinstance(reason_raw, str) or not reason_raw.strip():
        raise AdjudicatorError("AI adjudicator response must include a reason string")
    reason_value = reason_raw.strip()

    flags_raw = resp.get("flags")
    if not isinstance(flags_raw, Mapping):
        raise AdjudicatorError(
            "AI adjudicator response must include flags.account_match/debt_match"
        )

    if "duplicate" in flags_raw:
        duplicate_flag = _normalize_duplicate_flag(flags_raw.get("duplicate"))
        decision_primitive = resp.get("decision")
        if isinstance(decision_primitive, str) and decision_primitive.strip().lower() in {
            "duplicate",
            "not_duplicate",
        }:
            decision_value = decision_primitive.strip().lower()
        else:
            decision_value = "duplicate" if duplicate_flag else "not_duplicate"

        normalized_payload: Dict[str, Any] = {
            "decision": decision_value,
            "reason": reason_value,
            "flags": {"duplicate": duplicate_flag},
        }
        return normalized_payload, False

    duplicate_flag: bool | None = None
    if "duplicate" in flags_raw:
        duplicate_flag = _normalize_duplicate_flag(flags_raw.get("duplicate"))

    if decision_value in {"duplicate", "not_duplicate"}:
        inferred_duplicate = decision_value == "duplicate"
        if duplicate_flag is None:
            duplicate_flag = inferred_duplicate
        elif duplicate_flag != inferred_duplicate:
            raise AdjudicatorError("flags.duplicate disagrees with decision")

        normalized_payload: Dict[str, Any] = dict(resp)
        normalized_payload["decision"] = (
            "same_account_same_debt" if duplicate_flag else "different"
        )
        normalized_payload["reason"] = reason_value
        normalized_payload["flags"] = {
            "account_match": True if duplicate_flag else False,
            "debt_match": True if duplicate_flag else False,
            "duplicate": duplicate_flag,
        }
        normalized_payload["normalized"] = True
        return normalized_payload, True

    account_flag = _normalize_match_flag(flags_raw.get("account_match"), field="account_match")
    debt_flag = _normalize_match_flag(flags_raw.get("debt_match"), field="debt_match")

    normalized_decision = _decision_for_flags(account_flag, debt_flag, requested=decision_value)
    if normalized_decision not in ALLOWED_DECISIONS:
        raise AdjudicatorError("AI adjudicator decision was outside the allowed set")

    normalized_flags: Dict[str, Any] = {
        "account_match": account_flag,
        "debt_match": debt_flag,
    }
    if duplicate_flag is not None:
        normalized_flags["duplicate"] = duplicate_flag

    normalized = decision_value not in ALLOWED_DECISIONS or decision_value != normalized_decision

    normalized_payload: Dict[str, Any] = dict(resp)
    normalized_payload["decision"] = normalized_decision
    normalized_payload["reason"] = reason_value
    normalized_payload["flags"] = normalized_flags
    if normalized:
        normalized_payload["normalized"] = True
    else:
        normalized_payload.pop("normalized", None)

    return normalized_payload, normalized


def _coerce_positive_int(value: str | None, *, default: int) -> int:
    """Return a positive integer parsed from ``value`` or ``default``."""

    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _prepare_user_payload(pack: Dict[str, Any]) -> Dict[str, Any]:
    """Extract only the allowed keys from ``pack`` for the user message."""

    return {key: pack[key] for key in _ALLOWED_USER_PAYLOAD_KEYS if key in pack}


def decide_merge_or_different(
    pack: dict,
    *,
    timeout: int,
    messages: Sequence[Mapping[str, Any]] | None = None,
) -> dict:
    """Return the adjudicator response with decision, reason, and flags.

    May raise transport/HTTP errors; caller handles retries and ai_error tags.
    """

    base_url_raw = os.getenv("OPENAI_BASE_URL")
    base_url_clean = (base_url_raw or "").strip() or "https://api.openai.com/v1"
    base_url = base_url_clean.rstrip("/") or "https://api.openai.com/v1"
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set for adjudicator calls")

    model = os.getenv("AI_MODEL")
    if not model:
        raise RuntimeError("AI_MODEL must be set for adjudicator calls")

    request_timeout = _coerce_positive_int(os.getenv("AI_REQUEST_TIMEOUT"), default=timeout)

    project_header_enabled = os.getenv("OPENAI_SEND_PROJECT_HEADER", "0") == "1"
    global _AUTH_READY
    if not _AUTH_READY:
        with _AUTH_LOCK:
            if not _AUTH_READY:
                auth_probe()
                log.info(
                    "OPENAI_AUTH ready: key_prefix=%s project_header_enabled=%s base_url=%s",
                    _key_prefix(api_key),
                    project_header_enabled,
                    base_url_clean,
                )
                _AUTH_READY = True
    headers = build_openai_headers(api_key=api_key)

    org_id = os.getenv("OPENAI_ORG_ID")
    if org_id:
        headers["OpenAI-Organization"] = org_id

    if messages is None:
        user_payload = _prepare_user_payload(pack)
        request_messages: List[Dict[str, Any]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload)},
        ]
    else:
        request_messages = []
        for message in messages:
            if not isinstance(message, Mapping):
                raise AdjudicatorError("AI adjudicator messages must be objects")
            role_raw = message.get("role")
            if not isinstance(role_raw, str) or not role_raw.strip():
                raise AdjudicatorError("AI adjudicator messages require a role")
            cloned: Dict[str, Any] = {"role": role_raw.strip()}
            if "content" in message:
                cloned["content"] = message["content"]
            if "name" in message:
                cloned["name"] = message["name"]
            request_messages.append(cloned)
        if not request_messages:
            raise AdjudicatorError("AI adjudicator messages cannot be empty")

    request_body: Dict[str, object] = {
        "model": model,
        "messages": request_messages,
        "response_format": dict(RESPONSE_FORMAT),
    }
    for key, value in REQUEST_PARAMS.items():
        request_body[key] = value

    url = f"{base_url}/chat/completions"
    log.debug(
        "OPENAI_CALL endpoint=%s has_project_header=%s",
        "/v1/chat/completions",
        PROJECT_HEADER_NAME in headers,
    )
    response = httpx.post(url, headers=headers, json=request_body, timeout=request_timeout)
    response.raise_for_status()
    data = response.json()

    try:
        choice = data["choices"][0]
        message = choice["message"]
        content = message["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AdjudicatorError("Unexpected response structure from AI adjudicator") from exc

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise AdjudicatorError("AI adjudicator response was not valid JSON") from exc

    decision = parsed.get("decision")
    reason = parsed.get("reason")
    if not isinstance(decision, str) or not decision.strip():
        raise AdjudicatorError("AI adjudicator decision must be a non-empty string")
    if not isinstance(reason, str) or not reason.strip():
        raise AdjudicatorError("AI adjudicator response must include a reason string")

    flags = parsed.get("flags")
    if flags is not None and not isinstance(flags, dict):
        raise AdjudicatorError("AI adjudicator flags must be an object when provided")

    if flags is None:
        return {"decision": decision, "reason": reason}

    result = dict(parsed)
    result["decision"] = decision
    result["reason"] = reason
    result["flags"] = dict(flags)
    return result
