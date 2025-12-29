"""Runtime context helpers for validation normalization."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ValidationContext:
    """Container for contextual signals used during validation."""

    date_convention: Mapping[str, str]


_DEFAULT_DATE_CONVENTION: Mapping[str, str] = {
    "convention": "MDY",
    "month_language": "en",
}


def _normalize_date_convention(
    convention: Mapping[str, Any] | None,
) -> Mapping[str, str]:
    normalized = dict(_DEFAULT_DATE_CONVENTION)
    if not isinstance(convention, Mapping):
        return normalized

    raw_convention = str(convention.get("convention") or "").upper()
    if raw_convention in {"MDY", "DMY"}:
        normalized["convention"] = raw_convention

    raw_language = str(convention.get("month_language") or "").lower()
    if raw_language in {"en", "he"}:
        normalized["month_language"] = raw_language

    return normalized


def _build_context(convention: Mapping[str, Any] | None) -> ValidationContext:
    return ValidationContext(date_convention=_normalize_date_convention(convention))


_VALIDATION_CONTEXT: ContextVar[ValidationContext] = ContextVar(
    "validation_context", default=_build_context(None)
)


def get_validation_context() -> ValidationContext:
    """Return the current validation context."""

    return _VALIDATION_CONTEXT.get()


def set_validation_context(convention: Mapping[str, Any] | None) -> None:
    """Set the active validation context."""

    _VALIDATION_CONTEXT.set(_build_context(convention))


@contextmanager
def override_validation_context(convention: Mapping[str, Any] | None):
    """Temporarily override the validation context within a scope."""

    token = _VALIDATION_CONTEXT.set(_build_context(convention))
    try:
        yield get_validation_context()
    finally:
        _VALIDATION_CONTEXT.reset(token)
