"""Schema validation helpers for note_style analysis payloads."""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence

__all__ = ["validate_analysis_payload", "coerce_text"]

_MAX_EMPHASIS_ITEMS = 6
_RISK_FLAG_SANITIZE_PATTERN = re.compile(r"[^a-z0-9]+")
_MAX_RISK_FLAGS = 6


def coerce_text(value: Any, *, preserve_case: bool = False) -> str:
    """Return ``value`` coerced to a trimmed string."""

    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        try:
            text = value.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    else:
        text = str(value)
    text = text.strip()
    return text if preserve_case else text.lower()


def _require_string(value: Any, *, field: str) -> str:
    text = coerce_text(value, preserve_case=True)
    if not text:
        raise ValueError(f"{field} must be a non-empty string")
    return text


def _sanitize_emphasis(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    seen: set[str] = set()
    sanitized: list[str] = []
    for entry in values:
        text = coerce_text(entry)
        if not text:
            continue
        if text in seen:
            continue
        sanitized.append(text)
        seen.add(text)
        if len(sanitized) >= _MAX_EMPHASIS_ITEMS:
            break
    return sanitized


def _normalize_risk_flag(value: Any) -> str:
    text = coerce_text(value)
    if not text:
        return ""
    sanitized = _RISK_FLAG_SANITIZE_PATTERN.sub("_", text)
    sanitized = sanitized.strip("_")
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized


def _sanitize_risk_flags(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []
    seen: set[str] = set()
    flags: list[str] = []
    for entry in values:
        normalized = _normalize_risk_flag(entry)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        flags.append(normalized)
        if len(flags) >= _MAX_RISK_FLAGS:
            break
    return flags


def _sanitize_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("confidence must be a number between 0 and 1") from exc
    if math.isnan(numeric) or math.isinf(numeric):
        raise ValueError("confidence must be a finite number")
    if numeric < 0 or numeric > 1:
        raise ValueError("confidence must be within [0, 1]")
    return round(numeric, 3)


_amount_pattern = re.compile(r"-?\d+(?:\.\d+)?")


def _sanitize_amount(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            raise ValueError("context_hints.entities.amount must be finite")
        return numeric
    text = coerce_text(value, preserve_case=True)
    if not text:
        return None
    cleaned = text.replace(",", "")
    match = _amount_pattern.search(cleaned)
    if not match:
        raise ValueError("context_hints.entities.amount must be a number or null")
    try:
        numeric = float(match.group(0))
    except ValueError as exc:
        raise ValueError("context_hints.entities.amount must be a number or null") from exc
    if math.isnan(numeric) or math.isinf(numeric):
        raise ValueError("context_hints.entities.amount must be finite")
    return numeric


def _sanitize_context(context: Any) -> dict[str, Any]:
    if not isinstance(context, Mapping):
        raise ValueError("context_hints must be an object")

    timeframe_payload = context.get("timeframe")
    timeframe: dict[str, Any] = {"month": None, "relative": None}
    if timeframe_payload is not None:
        if not isinstance(timeframe_payload, Mapping):
            raise ValueError("context_hints.timeframe must be an object")
        month = coerce_text(timeframe_payload.get("month"), preserve_case=True)
        relative = coerce_text(timeframe_payload.get("relative"), preserve_case=True)
        timeframe["month"] = month or None
        timeframe["relative"] = relative or None

    topic = _require_string(context.get("topic"), field="context_hints.topic")

    entities_payload = context.get("entities")
    entities: dict[str, Any] = {"creditor": None, "amount": None}
    if entities_payload is not None:
        if not isinstance(entities_payload, Mapping):
            raise ValueError("context_hints.entities must be an object")
        creditor = coerce_text(entities_payload.get("creditor"), preserve_case=True)
        entities["creditor"] = creditor or None
        entities["amount"] = _sanitize_amount(entities_payload.get("amount"))

    return {
        "timeframe": timeframe,
        "topic": topic,
        "entities": entities,
    }


def validate_analysis_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a note_style analysis payload."""

    if not isinstance(payload, Mapping):
        raise ValueError("analysis payload must be an object")

    tone = _require_string(payload.get("tone"), field="tone")
    context = _sanitize_context(payload.get("context_hints"))
    emphasis = _sanitize_emphasis(payload.get("emphasis"))

    if "confidence" not in payload:
        raise ValueError("confidence is required")
    confidence = _sanitize_confidence(payload.get("confidence"))
    risk_flags = _sanitize_risk_flags(payload.get("risk_flags"))

    return {
        "tone": tone,
        "context_hints": context,
        "emphasis": emphasis,
        "confidence": confidence,
        "risk_flags": risk_flags,
    }
