"""Environment-backed configuration for merge adjudication helpers."""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable


logger = logging.getLogger(__name__)


_DEFAULT_AI_PACK_MAX_LINES_PER_SIDE = 20
_MIN_AI_PACK_MAX_LINES_PER_SIDE = 5
_DEFAULT_AI_MODEL = "gpt-4o-mini"
_DEFAULT_AI_REQUEST_TIMEOUT = 30
_DEFAULT_MERGE_V2_ONLY = True

_WARNED_KEYS: set[str] = set()


def _warn_once(key: str, raw: object, default: object, reason: str) -> None:
    """Emit a structured warning for an invalid environment value once."""

    if key in _WARNED_KEYS:
        return

    _WARNED_KEYS.add(key)
    payload = {
        "key": key,
        "value": "" if raw is None else str(raw),
        "default": default,
        "reason": reason,
    }
    logger.warning("MERGE_V2_CONFIG_DEFAULT %s", json.dumps(payload, sort_keys=True))


def _parse_int(key: str, raw: object, default: int, *, min_value: int | None = None) -> int:
    try:
        value = int(str(raw).strip())
    except Exception:
        _warn_once(key, raw, default, "invalid_int")
        return default

    if min_value is not None and value < min_value:
        _warn_once(key, raw, default, f"min_{min_value}")
        return default

    return value


def _read_int(
    key: str,
    default: int,
    *,
    min_value: int | None = None,
    fallback_keys: Iterable[str] = (),
) -> int:
    raw = os.getenv(key)
    if raw is not None:
        return _parse_int(key, raw, default, min_value=min_value)

    for fallback in fallback_keys:
        fallback_raw = os.getenv(fallback)
        if fallback_raw is not None:
            return _parse_int(fallback, fallback_raw, default, min_value=min_value)

    return default


def _read_str(key: str, default: str, *, fallback_keys: Iterable[str] = ()) -> str:
    raw = os.getenv(key)
    if raw is not None:
        value = str(raw).strip()
        if value:
            return value
        _warn_once(key, raw, default, "empty")
        return default

    for fallback in fallback_keys:
        fallback_raw = os.getenv(fallback)
        if fallback_raw is None:
            continue
        value = str(fallback_raw).strip()
        if value:
            return value
        _warn_once(fallback, fallback_raw, default, "empty")
        return default

    return default


def _read_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return bool(default)

    lowered = str(raw).strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False

    _warn_once(key, raw, bool(default), "invalid_bool")
    return bool(default)


def get_ai_pack_max_lines_per_side() -> int:
    """Return the per-side context limit for AI packs."""

    return _read_int(
        "AI_PACK_MAX_LINES_PER_SIDE",
        _DEFAULT_AI_PACK_MAX_LINES_PER_SIDE,
        min_value=_MIN_AI_PACK_MAX_LINES_PER_SIDE,
    )


def get_ai_model() -> str:
    """Return the chat completion model identifier."""

    return _read_str("AI_MODEL", _DEFAULT_AI_MODEL, fallback_keys=("AI_MODEL_ID",))


def get_ai_request_timeout() -> int:
    """Return the HTTP request timeout (seconds) for AI adjudication."""

    return _read_int(
        "AI_REQUEST_TIMEOUT",
        _DEFAULT_AI_REQUEST_TIMEOUT,
        min_value=1,
        fallback_keys=("AI_REQUEST_TIMEOUT_S",),
    )


def get_merge_v2_only() -> bool:
    """Return whether legacy merge artifacts should be skipped."""

    return _read_bool("MERGE_V2_ONLY", _DEFAULT_MERGE_V2_ONLY)


__all__ = [
    "get_ai_pack_max_lines_per_side",
    "get_ai_model",
    "get_ai_request_timeout",
    "get_merge_v2_only",
]
