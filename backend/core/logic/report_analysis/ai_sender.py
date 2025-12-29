"""Send merge adjudication packs to the AI adjudicator service."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import httpx

from backend.core.ai import PROJECT_HEADER_NAME, build_openai_headers

from backend.core.io.tags import read_tags, upsert_tag


log = logging.getLogger(__name__)


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 30.0

# Retry configuration – one initial attempt plus three retries using this schedule.
RETRY_BACKOFF_SECONDS: Sequence[float] = (1.0, 3.0, 7.0)
MAX_RETRIES = len(RETRY_BACKOFF_SECONDS)

ALLOWED_DECISIONS = {
    "same_account_same_debt",
    "same_account_diff_debt",
    "same_account_debt_unknown",
    "same_debt_diff_account",
    "same_debt_account_unknown",
    "different",
}

PAIR_TAG_BY_DECISION: dict[str, str] = {
    "same_account_same_debt": "same_account_pair",
    "same_account_diff_debt": "same_account_pair",
    "same_account_debt_unknown": "same_account_pair",
    "same_debt_diff_account": "same_debt_pair",
    "same_debt_account_unknown": "same_debt_pair",
}

_EXPECTED_DECISION_BY_FLAGS = {
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


@dataclass(frozen=True)
class AISenderConfig:
    """Configuration required to contact the chat completion API."""

    base_url: str
    api_key: str
    model: str
    timeout: float


@dataclass
class SendOutcome:
    """Result of attempting to adjudicate a single pack."""

    success: bool
    attempts: int
    decision: str | None = None
    reason: str | None = None
    flags: dict[str, bool | str] | None = None
    error_kind: str | None = None
    error_message: str | None = None


def _bool_from_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def is_enabled() -> bool:
    """Return whether AI adjudication is enabled via configuration."""

    env_value = os.getenv("ENABLE_AI_ADJUDICATOR")
    enabled = _bool_from_env(env_value, default=False)
    if env_value is not None:
        return enabled

    try:  # pragma: no cover - defensive fallback when module is absent
        import backend.config as backend_config  # type: ignore

        return bool(getattr(backend_config, "ENABLE_AI_ADJUDICATOR", False))
    except Exception:  # pragma: no cover - optional dependency
        return False


def _coerce_positive_float(raw: str | None, default: float) -> float:
    if raw is None:
        return default
    try:
        value = float(str(raw).strip())
    except Exception:
        return default
    if value <= 0:
        return default
    return value


def load_config_from_env() -> AISenderConfig:
    """Build :class:`AISenderConfig` from environment variables."""

    base_url_raw = os.getenv("OPENAI_BASE_URL")
    base_url_clean = (base_url_raw or "").strip() or DEFAULT_BASE_URL
    base_url = base_url_clean.rstrip("/") or DEFAULT_BASE_URL
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when sending AI merge packs")

    project_header_enabled = os.getenv("OPENAI_SEND_PROJECT_HEADER", "0") == "1"
    log.info(
        "OPENAI_SETUP key_prefix=%s project_header_enabled=%s base_url=%s",
        (api_key[:7] + "...") if api_key else "<empty>",
        project_header_enabled,
        base_url_clean,
    )

    model = os.getenv("AI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    timeout = _coerce_positive_float(os.getenv("AI_REQUEST_TIMEOUT"), DEFAULT_TIMEOUT)

    return AISenderConfig(base_url=base_url, api_key=api_key, model=model, timeout=timeout)


def _format_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _default_http_request(
    url: str,
    payload: Mapping[str, Any],
    headers: Mapping[str, str],
    timeout: float,
) -> httpx.Response:
    log.debug(
        "OPENAI_CALL endpoint=%s has_project_header=%s",
        "/v1/chat/completions",
        PROJECT_HEADER_NAME in headers,
    )
    return httpx.post(url, json=dict(payload), headers=dict(headers), timeout=timeout)


def _strip_code_fences(text: str) -> str:
    trimmed = text.strip()
    if not trimmed.startswith("```"):
        return trimmed

    lines = [line for line in trimmed.splitlines()]
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _parse_model_payload(content: str) -> MutableMapping[str, Any]:
    try:
        data = json.loads(_strip_code_fences(content))
    except json.JSONDecodeError as exc:
        raise ValueError("Model response must be valid JSON") from exc
    if not isinstance(data, MutableMapping):
        raise ValueError("Model response JSON must be an object")
    return data


def _normalize_match_flag(value: Any) -> bool | str | None:
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


def _flag_signature(value: bool | str) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return "unknown"


def _normalize_flags(flags: object) -> dict[str, bool | str]:
    if not isinstance(flags, Mapping):
        raise ValueError("Model response missing flags")
    account_flag = _normalize_match_flag(flags.get("account_match"))
    debt_flag = _normalize_match_flag(flags.get("debt_match"))
    if account_flag is None or debt_flag is None:
        raise ValueError("Model response must include account_match and debt_match flags")
    return {"account_match": account_flag, "debt_match": debt_flag}


def _sanitize_decision(payload: Mapping[str, Any]) -> tuple[str, str, dict[str, bool | str]]:
    raw_decision = payload.get("decision")
    decision = str(raw_decision).strip().lower()
    if decision not in ALLOWED_DECISIONS:
        raise ValueError(f"Unsupported decision: {raw_decision!r}")

    reason_raw = payload.get("reason")
    if reason_raw is None:
        raise ValueError("Model response missing reason")
    reason = str(reason_raw).strip()
    if not reason:
        raise ValueError("Model response reason must be non-empty")

    flags = _normalize_flags(payload.get("flags"))
    signature = (_flag_signature(flags["account_match"]), _flag_signature(flags["debt_match"]))
    expected = _EXPECTED_DECISION_BY_FLAGS.get(signature)
    if expected is None:
        raise ValueError("Model response flags contained an invalid combination")
    if decision != expected:
        raise ValueError("Model decision does not align with flags.account_match/debt_match")

    return decision, reason, flags


def send_single_attempt(
    pack: Mapping[str, Any],
    config: AISenderConfig,
    *,
    request: Callable[[str, Mapping[str, Any], Mapping[str, str], float], httpx.Response] | None = None,
) -> tuple[str, str, dict[str, bool | str]]:
    """Send ``pack`` once and return the decision, reason, and flags."""

    messages = pack.get("messages")
    if not isinstance(messages, Sequence):
        raise ValueError("Pack is missing messages payload")

    url = _format_url(config.base_url)
    payload = {
        "model": config.model,
        "messages": list(messages),
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    headers = build_openai_headers(api_key=config.api_key)

    sender = request or _default_http_request
    response = sender(url, payload, headers, config.timeout)
    response.raise_for_status()
    data = response.json()
    choices = data.get("choices") if isinstance(data, Mapping) else None
    if not choices:
        raise ValueError("Model response missing choices")
    message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
    if not isinstance(message, Mapping):
        raise ValueError("Model response missing message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Model response missing textual content")

    parsed = _parse_model_payload(content)
    return _sanitize_decision(parsed)


LogCallback = Callable[[str, Mapping[str, Any]], None]


def process_pack(
    pack: Mapping[str, Any],
    config: AISenderConfig,
    *,
    request: Callable[[str, Mapping[str, Any], Mapping[str, str], float], httpx.Response] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    log: LogCallback | None = None,
) -> SendOutcome:
    """Attempt to adjudicate ``pack`` using retry logic."""

    attempts = 0
    last_error: Exception | None = None
    max_attempts = 1 + MAX_RETRIES

    while attempts < max_attempts:
        attempts += 1
        if log is not None:
            log(
                "REQUEST",
                {
                    "attempt": attempts,
                    "max_attempts": max_attempts,
                },
            )

        try:
            decision, reason, flags = send_single_attempt(pack, config, request=request)
            if log is not None:
                log(
                    "RESPONSE",
                    {
                        "attempt": attempts,
                        "decision": decision,
                        "flags": flags,
                    },
                )
            return SendOutcome(
                success=True,
                attempts=attempts,
                decision=decision,
                reason=reason,
                flags=flags,
            )
        except Exception as exc:  # pragma: no cover - diverse error sources
            last_error = exc
            will_retry = attempts <= MAX_RETRIES
            if log is not None and will_retry:
                payload = {
                    "attempt": attempts,
                    "error": exc.__class__.__name__,
                    "will_retry": True,
                }
                log("ERROR", payload)

            if not will_retry:
                break

            delay = RETRY_BACKOFF_SECONDS[min(attempts - 1, len(RETRY_BACKOFF_SECONDS) - 1)]
            if log is not None:
                log(
                    "RETRY",
                    {
                        "attempt": attempts,
                        "delay_seconds": delay,
                    },
                )
            sleep(delay)

    error_kind = last_error.__class__.__name__ if last_error is not None else "UnknownError"
    error_message = str(last_error) if last_error is not None else "unknown"
    if log is not None and last_error is not None:
        log(
            "ERROR",
            {
                "attempt": attempts,
                "error": error_kind,
                "will_retry": False,
                "final": True,
            },
        )
    return SendOutcome(
        success=False,
        attempts=attempts,
        decision=None,
        reason=None,
        error_kind=error_kind,
        error_message=error_message,
    )


def isoformat_timestamp(dt: datetime | None = None) -> str:
    """Return a UTC ISO-8601 timestamp without fractional seconds."""

    if dt is None:
        dt = datetime.now(timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def _account_tags_dir(runs_root: os.PathLike[str] | str, sid: str) -> str:
    base = os.fspath(runs_root)
    return os.path.join(base, sid, "cases", "accounts")


def _ensure_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{label} must be an integer") from exc


def write_decision_tags(
    runs_root: os.PathLike[str] | str,
    sid: str,
    a_idx: Any,
    b_idx: Any,
    decision: str,
    reason: str,
    at: str,
    flags: Mapping[str, Any] | None = None,
) -> None:
    """Write symmetric ai_decision and supplemental pair tags for the pair."""

    account_a = _ensure_int(a_idx, "a_idx")
    account_b = _ensure_int(b_idx, "b_idx")

    base = _account_tags_dir(runs_root, sid)

    normalized_flags: dict[str, bool | str] | None = None
    if flags is not None:
        try:
            normalized_flags = _normalize_flags(flags)
        except ValueError:
            normalized_flags = None

    pair_tag_kind = PAIR_TAG_BY_DECISION.get(decision)

    for source_idx, other_idx in ((account_a, account_b), (account_b, account_a)):
        tag_path = os.path.join(base, str(source_idx), "tags.json")
        decision_tag = {
            "kind": "ai_decision",
            "tag": "ai_decision",
            "source": "ai_adjudicator",
            "with": other_idx,
            "decision": decision,
            "reason": reason,
            "at": at,
        }
        if normalized_flags is not None:
            decision_tag["flags"] = dict(normalized_flags)
        upsert_tag(tag_path, decision_tag, unique_keys=("kind", "with", "source"))

        if pair_tag_kind is not None:
            pair_tag = {
                "kind": pair_tag_kind,
                "with": other_idx,
                "source": "ai_adjudicator",
                "at": at,
                "reason": reason,
            }
            upsert_tag(tag_path, pair_tag, unique_keys=("kind", "with", "source"))
            _prune_pair_tags(tag_path, other_idx, keep_kind=pair_tag_kind)
        else:
            _prune_pair_tags(tag_path, other_idx, keep_kind=None)


def _prune_pair_tags(tag_path: str, other_idx: int, *, keep_kind: str | None) -> None:
    try:
        tags = read_tags(tag_path)
    except FileNotFoundError:
        return

    filtered: list[dict[str, Any]] = []
    modified = False
    for entry in tags:
        kind = str(entry.get("kind", "")).lower()
        if kind not in {"same_account_pair", "same_debt_pair"}:
            filtered.append(dict(entry))
            continue
        source = str(entry.get("source", ""))
        if source != "ai_adjudicator":
            filtered.append(dict(entry))
            continue
        partner_raw = entry.get("with")
        try:
            partner = int(partner_raw) if partner_raw is not None else None
        except (TypeError, ValueError):
            partner = None
        if partner != other_idx:
            filtered.append(dict(entry))
            continue
        if keep_kind is not None and kind == keep_kind:
            filtered.append(dict(entry))
            continue
        modified = True

    if not modified:
        return

    serialized = json.dumps(filtered, ensure_ascii=False, indent=2)
    Path(tag_path).write_text(f"{serialized}\n", encoding="utf-8")


def write_error_tags(
    runs_root: os.PathLike[str] | str,
    sid: str,
    a_idx: Any,
    b_idx: Any,
    error_kind: str,
    message: str,
    at: str,
) -> None:
    """Write symmetric ai_error tags for the pair."""

    account_a = _ensure_int(a_idx, "a_idx")
    account_b = _ensure_int(b_idx, "b_idx")

    base = _account_tags_dir(runs_root, sid)

    def _payload(other: int) -> dict[str, Any]:
        return {
            "kind": "ai_error",
            "with": other,
            "source": "ai_adjudicator",
            "error_kind": error_kind,
            "message": message,
            "at": at,
        }

    for source_idx, other_idx in ((account_a, account_b), (account_b, account_a)):
        tag_path = os.path.join(base, str(source_idx), "tags.json")
        upsert_tag(
            tag_path,
            _payload(other_idx),
            unique_keys=("kind", "with", "source"),
        )


__all__ = [
    "AISenderConfig",
    "SendOutcome",
    "ALLOWED_DECISIONS",
    "RETRY_BACKOFF_SECONDS",
    "MAX_RETRIES",
    "is_enabled",
    "load_config_from_env",
    "send_single_attempt",
    "process_pack",
    "isoformat_timestamp",
    "write_decision_tags",
    "write_error_tags",
]

