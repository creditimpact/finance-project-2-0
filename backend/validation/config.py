"""Configuration helpers for validation pipeline.

These helpers provide typed accessors for environment-driven switches used by
the validation step. Values are read on demand so that tests can override the
environment without needing to reload modules.
"""

from __future__ import annotations

import os
from typing import Optional


_DATE_TOL_DAYS_ENV = "DATE_TOL_DAYS"
_AMOUNT_TOL_ABS_ENV = "AMOUNT_TOL_ABS"
_AMOUNT_TOL_RATIO_ENV = "AMOUNT_TOL_RATIO"
_PREVALIDATION_OUT_REL_ENV = "PREVALIDATION_OUT_PATH_REL"


def _read_int(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback."""

    raw: Optional[str] = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default

    try:
        return int(raw)
    except ValueError:
        return default


def _read_float(name: str, default: float) -> float:
    """Read a floating-point environment variable with a fallback."""

    raw: Optional[str] = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default

    try:
        return float(raw)
    except ValueError:
        return default


def get_date_tolerance_days() -> int:
    """Return the number of days allowed for date comparisons."""

    return _read_int(_DATE_TOL_DAYS_ENV, 5)


def get_amount_tolerance_abs() -> float:
    """Return the absolute tolerance allowed for amount comparisons."""

    return _read_float(_AMOUNT_TOL_ABS_ENV, 50.0)


def get_amount_tolerance_ratio() -> float:
    """Return the ratio-based tolerance allowed for amount comparisons."""

    return _read_float(_AMOUNT_TOL_RATIO_ENV, 0.01)


def get_prevalidation_trace_relpath() -> str:
    """Return the relative path to the prevalidation convention trace file."""

    raw = os.getenv(_PREVALIDATION_OUT_REL_ENV)
    if raw is None or raw.strip() == "":
        return "trace/date_convention.json"
    return raw


__all__ = [
    "get_amount_tolerance_abs",
    "get_amount_tolerance_ratio",
    "get_date_tolerance_days",
    "get_prevalidation_trace_relpath",
]

