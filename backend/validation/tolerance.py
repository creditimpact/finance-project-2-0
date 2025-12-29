"""Tolerance helpers for validation comparisons."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Mapping

from backend.validation import config as validation_config
from backend.validation.utils_amounts import are_amounts_within_tolerance
from backend.validation.utils_dates import (
    are_dates_within_tolerance,
    load_date_convention_for_sid,
)

_DEBUG_ENV = "VALIDATION_DEBUG"
_BUREAU_KEYS = ("equifax", "experian", "transunion")

DATE_FIELDS = {
    "date_opened",
    "closed_date",
    "last_payment",
    "date_reported",
    "date_of_last_activity",
    "last_verified",
}

AMOUNT_FIELDS = {
    "balance_owed",
    "high_balance",
    "credit_limit",
    "past_due_amount",
    "payment_amount",
}


@lru_cache(maxsize=32)
def _cached_date_convention(runs_root: str, sid: str) -> str:
    """Return the cached date convention for ``sid``."""

    return load_date_convention_for_sid(runs_root, sid)


def _normalized_field(field: str | None) -> str:
    return str(field or "").strip().lower()


def _extract_bureau_values(bureau_values: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(bureau_values, Mapping):
        return {key: None for key in _BUREAU_KEYS}
    return {key: bureau_values.get(key) for key in _BUREAU_KEYS}


def _resolve_date_convention(runs_root: str | None, sid: str | None) -> str | None:
    if not runs_root or not sid:
        return None
    try:
        return _cached_date_convention(runs_root, sid)
    except Exception:
        return None


def _is_debug_enabled() -> bool:
    value = os.getenv(_DEBUG_ENV)
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized not in {"", "0", "false", "no"}


def _with_log_payload(
    result: Mapping[str, Any],
    log_payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not log_payload:
        return dict(result)

    enriched = dict(result)
    enriched["log_payload"] = dict(log_payload)
    return enriched


def evaluate_field_with_tolerance(
    sid: str | None,
    runs_root: str | os.PathLike[str] | None,
    field: str,
    bureau_values: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate whether ``field`` is a mismatch after tolerance is applied."""

    normalized_field = _normalized_field(field)
    values = _extract_bureau_values(bureau_values)

    result = {
        "is_mismatch": True,
        "metric": None,
        "tolerance_applied": False,
        "note": None,
    }

    debug_enabled = _is_debug_enabled()

    log_payload: dict[str, Any] | None = None

    if normalized_field in DATE_FIELDS:
        conv = _resolve_date_convention(
            os.fspath(runs_root) if runs_root is not None else None,
            sid,
        )
        tol_days = validation_config.get_date_tolerance_days()
        used_conv = conv if conv is not None else "MDY"
        within, span = are_dates_within_tolerance(values.values(), used_conv, tol_days)
        if debug_enabled:
            log_payload = {
                "kind": "date",
                "sid": sid or "",
                "field": normalized_field,
                "conv": used_conv,
                "tol_days": tol_days,
                "span": span,
                "within": within,
            }
        if within:
            note = {
                "field": normalized_field,
                "status": "within",
                "tol_days": tol_days,
            }
            if span is not None:
                note["span_days"] = span
            return _with_log_payload(
                {
                    "is_mismatch": False,
                    "metric": span,
                    "tolerance_applied": True,
                    "note": note,
                },
                log_payload,
            )
        return _with_log_payload(result, log_payload)

    if normalized_field in AMOUNT_FIELDS:
        abs_tol = validation_config.get_amount_tolerance_abs()
        ratio_tol = validation_config.get_amount_tolerance_ratio()
        within, diff, max_value = are_amounts_within_tolerance(
            values.values(), abs_tol, ratio_tol
        )
        if debug_enabled:
            log_payload = {
                "kind": "amount",
                "sid": sid or "",
                "field": normalized_field,
                "abs": abs_tol,
                "ratio": ratio_tol,
                "diff": diff,
                "maxv": max_value,
                "within": within,
            }
        if within:
            ratio_cap = abs(max_value) * ratio_tol if max_value is not None else 0.0
            threshold = max(abs_tol, ratio_cap)
            note = {
                "field": normalized_field,
                "status": "within",
                "ceil": threshold,
            }
            if diff is not None:
                note["diff"] = diff
            return _with_log_payload(
                {
                    "is_mismatch": False,
                    "metric": diff,
                    "tolerance_applied": True,
                    "note": note,
                },
                log_payload,
            )
        return _with_log_payload(result, log_payload)

    return dict(result)


def clear_cached_conventions() -> None:
    """Clear cached date conventions (primarily for tests)."""

    _cached_date_convention.cache_clear()


__all__ = [
    "AMOUNT_FIELDS",
    "DATE_FIELDS",
    "clear_cached_conventions",
    "evaluate_field_with_tolerance",
]
