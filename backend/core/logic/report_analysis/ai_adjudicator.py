from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

import backend.config as config
from backend.core.ai import PROJECT_HEADER_NAME, build_openai_headers
from backend.core.io.tags import upsert_tag

from . import config as merge_config
from .ai_pack import DEFAULT_MAX_LINES


logger = logging.getLogger(__name__)


HIGHLIGHT_KEYS: tuple[str, ...] = (
    "total",
    "identity_score",
    "debt_score",
    "triggers",
    "parts",
    "matched_fields",
    "conflicts",
    "acctnum_level",
)

ALLOWED_DECISIONS: tuple[str, ...] = (
    "same_account_same_debt",
    "same_account_diff_debt",
    "same_account_debt_unknown",
    "same_debt_diff_account",
    "same_debt_account_unknown",
    "different",
)

SYSTEM_MESSAGE = (
    "You are an expert credit tradeline merge adjudicator. Review the provided "
    "highlights and short context snippets to decide if the two accounts refer to "
    "the same underlying obligation. Treat the token '--' as missing or unknown "
    "data. Prioritize strong triggers over mid triggers; mid triggers offer "
    "supporting evidence but cannot override conflicts backed by strong signals. "
    "Creditor names may appear with aliases, abbreviations, or formatting "
    "differences—treat reasonable variants as referring to the same source when "
    "supported by other evidence.\n"
    "Allowed decisions (exact strings, choose one):\n"
    "- same_account_same_debt        # accounts align and refer to the same debt\n"
    "- same_account_diff_debt        # same account, but debt details clearly differ\n"
    "- same_account_debt_unknown     # same account, debt status cannot be confirmed\n"
    "- same_debt_diff_account        # same debt, but reported under a different account\n"
    "- same_debt_account_unknown     # same debt, account identity cannot be confirmed\n"
    "- different                     # neither the account nor the debt matches\n"
    "Legacy labels such as merge, same_debt, same_debt_account_different, "
    "same_account, same_account_debt_different, and different may appear in "
    "reference material, but you MUST respond with only the six decisions above.\n"
    "If only last four digits match but stems differ, never choose any same_account_*.\n"
    "If account identifiers DO NOT match, but:\n"
    "- amounts_equal_within_tol is true for positive debt (balance and/or past due), AND\n"
    "- one side is a collection agent (is_collection_agency_*) while the other is an original creditor (is_original_creditor_*), AND\n"
    "- dates_plausible_chain is true (collection reported/opened after the original),\n"
    "then prefer \"same_debt_diff_account\" over \"different\".\n"
    "When BOTH sides are collection agencies:\n"
    "- If amounts_equal_within_tol is true for positive debt and dates_plausible_chain is true, prefer \"same_debt_diff_account\" over \"different\".\n"
    "- If amounts match but timing is ambiguous, use \"same_debt_account_unknown\".\n"
    "- Do NOT pick \"different\" solely because account numbers or lender names differ.\n"
    "Allowed outputs: same_account_same_debt, same_account_diff_debt, same_account_debt_unknown, same_debt_diff_account, same_debt_account_unknown, different.\n"
    "Return strict JSON only: {\"decision\":\"<one above>\", "
    "\"reason\":\"short natural language\", \"flags\":{\"account_match\":true|false|"
    "\"unknown\",\"debt_match\":true|false|\"unknown\"}}. Do not add commentary or "
    "extra keys."
)


def _coerce_positive_int(value: Any, default: int) -> int:
    try:
        number = int(str(value))
    except Exception:
        return default
    return number if number > 0 else default


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(str(value))
    except Exception:
        return default


def _limit_context(lines: list[Any], limit: int) -> list[str]:
    if not lines:
        return []
    coerced = [str(item) if item is not None else "" for item in lines]
    if limit <= 0:
        return coerced
    return coerced[:limit]


def _extract_highlights(source: dict[str, Any] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if not isinstance(source, dict):
        return payload
    for key in HIGHLIGHT_KEYS:
        if key in source:
            payload[key] = source[key]
    return payload


def _build_user_message(pack: dict, max_lines: int) -> str:
    pair = pack.get("pair") or {}
    ids = pack.get("ids") or {}
    highlights = _extract_highlights(pack.get("highlights"))

    context = pack.get("context") or {}
    context_a = _limit_context(list(context.get("a") or []), max_lines)
    context_b = _limit_context(list(context.get("b") or []), max_lines)

    summary = {
        "sid": pack.get("sid", ""),
        "pair": {"a": pair.get("a"), "b": pair.get("b")},
        "account_numbers": {
            "a": ids.get("account_number_a", "--"),
            "b": ids.get("account_number_b", "--"),
        },
        "account_numbers_normalized": {
            "a": ids.get("account_number_a_normalized", "--"),
            "b": ids.get("account_number_b_normalized", "--"),
        },
        "account_numbers_last4": {
            "a": ids.get("account_number_a_last4", "--"),
            "b": ids.get("account_number_b_last4", "--"),
        },
        "highlights": highlights,
        "context": {
            "a": context_a,
            "b": context_b,
        },
    }

    return json.dumps(summary, ensure_ascii=False, sort_keys=True)


def build_prompt_from_pack(pack: dict) -> dict[str, str]:
    limits = pack.get("limits") or {}
    default_limit = merge_config.get_ai_pack_max_lines_per_side()
    if default_limit <= 0:
        default_limit = DEFAULT_MAX_LINES
    pack_limit = _coerce_positive_int(limits.get("max_lines_per_side"), default_limit)
    max_lines = min(default_limit, pack_limit)

    user_message = _build_user_message(pack, max_lines)

    return {"system": SYSTEM_MESSAGE, "user": user_message}


def _strip_code_fences(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_reasons(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _normalize_flag(value: Any) -> bool | str | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "unknown":
            return "unknown"
    return None


def _expected_decision_for_flags(account_flag: bool | str, debt_flag: bool | str) -> str:
    signature = (
        "true" if account_flag is True else "false" if account_flag is False else "unknown",
        "true" if debt_flag is True else "false" if debt_flag is False else "unknown",
    )

    mapping = {
        ("true", "true"): "same_account_same_debt",
        ("true", "false"): "same_account_diff_debt",
        ("true", "unknown"): "same_account_debt_unknown",
        ("false", "true"): "same_debt_diff_account",
        ("false", "false"): "different",
        ("false", "unknown"): "different",
        ("unknown", "true"): "same_debt_account_unknown",
        ("unknown", "false"): "different",
        ("unknown", "unknown"): "different",
    }

    expected = mapping.get(signature)
    if expected is None:
        raise ValueError("Invalid flag combination in AI decision")
    return expected


def _sanitize_ai_decision(resp: Mapping[str, Any] | None, *, allow_disabled: bool) -> dict[str, Any]:
    if not isinstance(resp, Mapping):
        raise ValueError("AI decision payload must be a mapping")

    decision = str(resp.get("decision", "")).strip().lower()
    if decision == "ai_disabled":
        if not allow_disabled:
            raise ValueError("ai_disabled decision not permitted in this context")
        reason = str(resp.get("reason", "")) or "AI adjudication disabled"
        return {
            "decision": "ai_disabled",
            "confidence": 0.0,
            "reason": reason,
            "reasons": [reason] if reason else [],
            "flags": {"account_match": "unknown", "debt_match": "unknown"},
        }

    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"Unsupported AI decision: {decision!r}")

    reason_raw = resp.get("reason")
    if not isinstance(reason_raw, str) or not reason_raw.strip():
        raise ValueError("AI decision payload missing reason")
    reason = reason_raw.strip()

    confidence_raw = resp.get("confidence", 0.0)
    confidence = _coerce_float(confidence_raw, 0.0)
    if confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"Confidence must be between 0 and 1: {confidence_raw!r}")

    reasons = _normalize_reasons(resp.get("reasons"))
    if not reasons:
        reasons = [reason]

    flags_raw = resp.get("flags")
    if not isinstance(flags_raw, Mapping):
        raise ValueError("AI decision payload missing flags")

    account_flag = _normalize_flag(flags_raw.get("account_match"))
    debt_flag = _normalize_flag(flags_raw.get("debt_match"))
    if account_flag is None or debt_flag is None:
        raise ValueError("AI decision flags must include account_match and debt_match")

    expected = _expected_decision_for_flags(account_flag, debt_flag)
    if decision != expected:
        raise ValueError("AI decision does not align with flags.account_match/debt_match")

    return {
        "decision": decision,
        "confidence": float(confidence),
        "reason": reason,
        "reasons": reasons,
        "flags": {"account_match": account_flag, "debt_match": debt_flag},
    }


def _parse_ai_response(content: str) -> dict[str, Any]:
    trimmed = _strip_code_fences(content)
    data = json.loads(trimmed)
    if not isinstance(data, Mapping):
        raise ValueError("AI response JSON must be an object")
    return dict(data)


def _estimate_token_count(messages: Sequence[Mapping[str, Any]] | None) -> int:
    if not messages:
        return 0

    total_chars = 0
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if isinstance(content, str):
            total_chars += len(content)

    if total_chars <= 0:
        return 0

    return max(1, (total_chars + 3) // 4)


def _build_request_payload(pack: dict) -> tuple[str, dict[str, Any], dict[str, str], dict[str, Any]]:
    prompt = build_prompt_from_pack(pack)
    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]

    base_url_raw = os.getenv("OPENAI_BASE_URL")
    base_url_clean = (base_url_raw or "").strip() or "https://api.openai.com/v1"
    base_url = base_url_clean.rstrip("/") or "https://api.openai.com/v1"
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when AI adjudication is enabled")

    model = merge_config.get_ai_model()
    # Adjudicator requests must be deterministic to ensure parity with the
    # manual workflow. Force zero temperature and unity top_p rather than
    # permitting environment overrides that might introduce randomness.
    temperature = 0.0
    top_p = 1.0
    max_tokens = _coerce_positive_int(
        os.getenv("AI_MAX_TOKENS"), getattr(config, "AI_MAX_TOKENS", 600)
    )

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    project_header_enabled = os.getenv("OPENAI_SEND_PROJECT_HEADER", "0") == "1"
    logger.info(
        "OPENAI_SETUP key_prefix=%s project_header_enabled=%s base_url=%s",
        (api_key[:7] + "...") if api_key else "<empty>",
        project_header_enabled,
        base_url_clean,
    )
    headers = build_openai_headers(api_key=api_key)

    metadata = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    return f"{base_url}/chat/completions", payload, headers, metadata


def adjudicate_pair(pack: dict) -> dict[str, Any]:
    pair = pack.get("pair") or {}
    sid = str(pack.get("sid") or "")
    a_idx = pair.get("a")
    b_idx = pair.get("b")

    context = pack.get("context") or {}
    context_sizes = {
        "a": len(context.get("a") or []),
        "b": len(context.get("b") or []),
    }

    if not getattr(config, "ENABLE_AI_ADJUDICATOR", False):
        log_payload = {
            "sid": sid,
            "pair": {"a": a_idx, "b": b_idx},
            "reason": "disabled",
        }
        logger.info("AI_ADJUDICATOR_SKIPPED %s", json.dumps(log_payload, sort_keys=True))
        reason = "AI adjudication disabled"
        return {
            "decision": "ai_disabled",
            "confidence": 0.0,
            "reason": reason,
            "reasons": [reason],
            "flags": {"account_match": "unknown", "debt_match": "unknown"},
        }

    url, payload, headers, metadata = _build_request_payload(pack)
    prompt_tokens_est = _estimate_token_count(payload.get("messages"))
    request_log = {
        "sid": sid,
        "pair": {"a": a_idx, "b": b_idx},
        "context_sizes": context_sizes,
        "model": metadata.get("model"),
        "temperature": metadata.get("temperature"),
        "max_tokens": metadata.get("max_tokens"),
        "prompt_tokens_est": prompt_tokens_est,
    }
    logger.info("AI_ADJUDICATOR_REQUEST %s", json.dumps(request_log, sort_keys=True))

    timeout_s = float(merge_config.get_ai_request_timeout())

    started = time.perf_counter()
    try:
        logger.debug(
            "OPENAI_CALL endpoint=%s has_project_header=%s",
            "/v1/chat/completions",
            PROJECT_HEADER_NAME in headers,
        )
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout_s)
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("OpenAI response missing choices")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("OpenAI response missing textual content")

        parsed = _parse_ai_response(content)
        sanitized = _sanitize_ai_decision(parsed, allow_disabled=False)

        duration_ms = (time.perf_counter() - started) * 1000
        response_log = {
            "sid": sid,
            "pair": {"a": a_idx, "b": b_idx},
            "decision": sanitized["decision"],
            "reason": sanitized.get("reason"),
            "flags": sanitized.get("flags", {}),
            "confidence": sanitized.get("confidence", 0.0),
            "reasons_count": len(sanitized.get("reasons", [])),
            "latency_ms": round(duration_ms, 3),
        }
        logger.info("AI_ADJUDICATOR_RESPONSE %s", json.dumps(response_log, sort_keys=True))
        return sanitized
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        error_log = {
            "sid": sid,
            "pair": {"a": a_idx, "b": b_idx},
            "error": exc.__class__.__name__,
            "latency_ms": round(duration_ms, 3),
        }
        logger.error("AI_ADJUDICATOR_ERROR %s", json.dumps(error_log, sort_keys=True))
        raise


def _remove_legacy_ai_artifacts(base: Path, account_idx: int) -> None:
    ai_dir = base / str(account_idx) / "ai"
    if not ai_dir.exists():
        return
    try:
        entries = list(ai_dir.iterdir())
    except FileNotFoundError:
        return
    except NotADirectoryError:
        return
    except OSError:
        logger.debug(
            "AI_ADJUDICATOR_PRUNE_AI_DIR_LIST_FAILED path=%s", ai_dir, exc_info=True
        )
        return

    for entry in entries:
        if not entry.is_file():
            continue
        try:
            entry.unlink()
        except OSError:
            logger.debug(
                "AI_ADJUDICATOR_PRUNE_AI_ENTRY_FAILED path=%s", entry, exc_info=True
            )

    try:
        ai_dir.rmdir()
    except OSError:
        # Directory may remain when other non-file entries exist or concurrent writes happen.
        logger.debug(
            "AI_ADJUDICATOR_PRUNE_AI_DIR_RMDIR_FAILED path=%s", ai_dir, exc_info=True
        )


def persist_ai_decision(
    sid: str,
    runs_root: str | os.PathLike[str],
    a_idx: int,
    b_idx: int,
    resp: Mapping[str, Any],
) -> None:
    sanitized = _sanitize_ai_decision(resp, allow_disabled=True)

    try:
        account_a = int(a_idx)
        account_b = int(b_idx)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError("Account indices must be integers") from exc

    sid_str = str(sid)
    base = Path(runs_root) / sid_str / "cases" / "accounts"

    _remove_legacy_ai_artifacts(base, account_a)
    _remove_legacy_ai_artifacts(base, account_b)

    tag_a = {
        "kind": "merge_result",
        "with": account_b,
        "decision": sanitized["decision"],
        "confidence": sanitized.get("confidence", 0.0),
        "reason": sanitized.get("reason"),
        "reasons": list(sanitized.get("reasons", [])),
        "flags": dict(sanitized.get("flags", {})),
        "source": "ai_adjudicator",
    }
    tag_b = {
        "kind": "merge_result",
        "with": account_a,
        "decision": sanitized["decision"],
        "confidence": sanitized.get("confidence", 0.0),
        "reason": sanitized.get("reason"),
        "reasons": list(sanitized.get("reasons", [])),
        "flags": dict(sanitized.get("flags", {})),
        "source": "ai_adjudicator",
    }

    tag_path_a = base / str(account_a) / "tags.json"
    tag_path_b = base / str(account_b) / "tags.json"

    upsert_tag(tag_path_a, tag_a, unique_keys=("kind", "with", "source"))
    upsert_tag(tag_path_b, tag_b, unique_keys=("kind", "with", "source"))

    tag_log = {
        "sid": sid_str,
        "pair": {"a": account_a, "b": account_b},
        "decision": sanitized["decision"],
        "confidence": sanitized.get("confidence", 0.0),
        "reason": sanitized.get("reason"),
        "reasons": list(sanitized.get("reasons", [])),
        "flags": dict(sanitized.get("flags", {})),
    }
    logger.info("MERGE_V2_TAG_UPDATE %s", json.dumps(tag_log, sort_keys=True))

