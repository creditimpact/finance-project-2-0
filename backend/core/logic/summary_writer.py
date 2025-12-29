"""Utilities for persisting validation summaries with feature toggles."""

from __future__ import annotations

import os
from typing import Any, Mapping, MutableMapping

__all__ = [
    "include_field_consistency",
    "include_legacy_requirements",
    "should_write_empty_requirements",
    "sanitize_validation_payload",
    "strip_disallowed_sections",
]

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def _coerce_flag(raw_value: str | None, *, default: bool) -> bool:
    """Normalize ``raw_value`` into a boolean feature flag."""

    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    return default


def include_field_consistency() -> bool:
    """Return ``True`` when ``field_consistency`` should be written."""

    return _coerce_flag(os.getenv("SUMMARY_INCLUDE_FIELD_CONSISTENCY", "1"), default=True)


def include_legacy_requirements() -> bool:
    """Return ``True`` when legacy ``requirements`` arrays are enabled."""

    return _coerce_flag(os.getenv("VALIDATION_SUMMARY_INCLUDE_REQUIREMENTS", "0"), default=False)


def should_write_empty_requirements() -> bool:
    """Return ``True`` when empty validation requirement blocks should persist."""

    return _coerce_flag(os.getenv("REQUIREMENTS_WRITE_EMPTY", "1"), default=True)


def sanitize_validation_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``payload`` with disabled sections removed."""

    sanitized = dict(payload)
    if not include_legacy_requirements():
        sanitized.pop("requirements", None)
    if not include_field_consistency():
        sanitized.pop("field_consistency", None)
    return sanitized


def strip_disallowed_sections(summary: MutableMapping[str, Any]) -> bool:
    """Strip toggled-off sections from ``summary`` in-place.

    Parameters
    ----------
    summary:
        Parsed ``summary.json`` contents that will be mutated in-place.

    Returns
    -------
    bool
        ``True`` when the summary was modified.
    """

    changed = False

    if not include_field_consistency():
        if summary.pop("field_consistency", None) is not None:
            changed = True

    block = summary.get("validation_requirements")
    if isinstance(block, MutableMapping):
        if not include_field_consistency():
            if block.pop("field_consistency", None) is not None:
                changed = True
        if not include_legacy_requirements():
            if block.pop("requirements", None) is not None:
                changed = True

    return changed

