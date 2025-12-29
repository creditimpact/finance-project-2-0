"""Utilities for redacting PII from validation packs and logs."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from backend.core.logic.utils.pii import redact_pii

_NAME_PLACEHOLDER = "[REDACTED_NAME]"
_PHONE_PLACEHOLDER = "[REDACTED_PHONE]"


def _mask_account_number_display(text: str) -> str:
    """Mask all but the last four digits in ``text`` while preserving formatting."""

    digits = [ch for ch in text if ch.isdigit()]
    if not digits:
        return text

    keep = digits[-4:] if len(digits) >= 4 else digits
    masked_digits = ["*"] * (len(digits) - len(keep)) + keep

    result: list[str] = []
    index = 0
    for ch in text:
        if ch.isdigit():
            result.append(masked_digits[index])
            index += 1
        else:
            result.append(ch)
    return "".join(result)


def _contains_name_key(path: Sequence[str]) -> bool:
    for part in path:
        lowered = part.lower()
        if "name" in lowered:
            return True
    return False


def _contains_phone_key(path: Sequence[str]) -> bool:
    for part in path:
        if "phone" in part.lower():
            return True
    return False


def _contains_account_display(
    path: Sequence[str], account_field: str | None
) -> bool:
    if any(part.lower() == "account_number_display" for part in path):
        return True
    if account_field == "account_number_display":
        return any(part.lower() == "bureaus" for part in path)
    return False


def _sanitize_string(
    value: str, path: Sequence[str], account_field: str | None
) -> str:
    if _contains_account_display(path, account_field):
        return _mask_account_number_display(value)

    if _contains_phone_key(path):
        return _PHONE_PLACEHOLDER

    sanitized = redact_pii(value)

    if _contains_name_key(path):
        return _NAME_PLACEHOLDER

    return sanitized


def _sanitize(value: Any, path: tuple[str, ...], account_field: str | None) -> Any:
    if isinstance(value, Mapping):
        field_context = account_field
        field_value = value.get("field") if isinstance(value, Mapping) else None
        if isinstance(field_value, str) and field_value.strip():
            field_context = field_value.strip().lower()

        sanitized_map: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            sanitized_map[key_str] = _sanitize(item, path + (key_str,), field_context)
        return sanitized_map

    if isinstance(value, list):
        return [_sanitize(item, path, account_field) for item in value]

    if isinstance(value, tuple):  # pragma: no cover - defensive
        return tuple(_sanitize(item, path, account_field) for item in value)

    if isinstance(value, str):
        return _sanitize_string(value, path, account_field)

    return value


def sanitize_validation_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-sanitized copy of ``payload`` safe for persistence."""

    return _sanitize(payload, tuple(), None)


def sanitize_validation_log_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a sanitized copy of ``payload`` safe for structured logging."""

    return _sanitize(payload, tuple(), None)


__all__ = [
    "sanitize_validation_payload",
    "sanitize_validation_log_payload",
]
