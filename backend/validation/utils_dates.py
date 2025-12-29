"""Utilities for working with date conventions in validation."""

from __future__ import annotations

import json
import logging
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple

from .config import get_prevalidation_trace_relpath


def business_to_calendar(
    start_date_or_weekday: date | datetime | int,
    business_days: int,
    *,
    holidays: Iterable[date | datetime] | None = None,
    weekend: Iterable[int] = (5, 6),
) -> int:
    """Return calendar days needed to satisfy a ``business_days`` SLA.

    Parameters
    ----------
    start_date_or_weekday:
        Either a :class:`datetime.date`, :class:`datetime.datetime`, or the
        integer weekday (``0`` = Monday … ``6`` = Sunday) representing the
        starting point of the SLA window.
    business_days:
        The number of business days that must elapse. Non-positive values
        return ``0``.
    holidays:
        Optional iterable of holiday dates that should be skipped in addition to
        weekends.
    weekend:
        Iterable of weekday integers that represent non-working days. Defaults
        to Saturday/Sunday (``5``/``6``).
    """

    try:
        required_days = int(business_days)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("business_days must be an integer") from exc

    if required_days <= 0:
        return 0

    weekend_days = {int(day) for day in weekend}

    current_date: date | None
    current_weekday: int

    if isinstance(start_date_or_weekday, datetime):
        current_date = start_date_or_weekday.date()
        current_weekday = current_date.weekday()
    elif isinstance(start_date_or_weekday, date):
        current_date = start_date_or_weekday
        current_weekday = current_date.weekday()
    elif isinstance(start_date_or_weekday, int):
        current_weekday = int(start_date_or_weekday)
        if current_weekday < 0 or current_weekday > 6:
            raise ValueError("Weekday must be between 0 (Mon) and 6 (Sun)")
        current_date = None
    else:  # pragma: no cover - defensive
        raise TypeError(
            "start_date_or_weekday must be a date, datetime, or weekday integer"
        )

    holiday_set: set[date] = set()
    if holidays is not None:
        for entry in holidays:
            if isinstance(entry, datetime):
                holiday_set.add(entry.date())
            elif isinstance(entry, date):
                holiday_set.add(entry)
            else:  # pragma: no cover - defensive
                raise TypeError("holiday entries must be date or datetime")

    calendar_days = 0
    accrued_business = 0

    while accrued_business < required_days:
        if current_date is not None:
            weekday = current_date.weekday()
            is_holiday = current_date in holiday_set
        else:
            weekday = current_weekday
            is_holiday = False

        calendar_days += 1

        if weekday not in weekend_days and not is_holiday:
            accrued_business += 1
            if accrued_business >= required_days:
                break

        if current_date is not None:
            current_date += timedelta(days=1)
        else:
            current_weekday = (current_weekday + 1) % 7

    return calendar_days


def business_to_calendar_days(business_days: Any) -> int:
    """Backward compatible wrapper assuming a Monday starting point."""

    try:
        days = int(business_days)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError("business_days must be an integer") from exc

    return business_to_calendar(0, days)


_LOGGER = logging.getLogger(__name__)

_VALID_CONVENTIONS = {"DMY", "MDY", "YMD"}


def parse_date_with_convention(s: Optional[str], conv: str) -> Optional[datetime]:
    """Parse ``s`` according to ``conv`` returning a :class:`datetime` or ``None``.

    The function first attempts to parse the string as an ISO-8601 formatted
    date (``YYYY-MM-DD``). When that fails the supplied convention is applied
    using the typical ``/``, ``-`` or ``.`` separators. Any invalid or empty
    value results in ``None``.
    """

    if s is None:
        return None

    text = s.strip()
    if not text:
        return None

    # ISO-8601 parsing
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass

    parts = [part for part in re.split(r"[\s./-]", text) if part]
    if len(parts) != 3:
        return None

    try:
        if conv == "YMD":
            year, month, day = (int(parts[0]), int(parts[1]), int(parts[2]))
        elif conv == "DMY":
            day, month, year = (int(parts[0]), int(parts[1]), int(parts[2]))
        else:  # Default to MDY
            month, day, year = (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None

    try:
        return datetime(year, month, day)
    except ValueError:
        return None


DEFAULT_DATE_TOLERANCE_DAYS = 5


def are_dates_within_tolerance(
    values: Iterable[Optional[str]], conv: str, tol_days: int
) -> Tuple[bool, Optional[int]]:
    """Return whether ``values`` fall within a ``tol_days`` window.

    Parameters
    ----------
    values:
        Iterable of up to three date strings in EQ/EX/TU order. ``None``
        entries are ignored when computing the span.
    conv:
        Date convention used for parsing non-ISO formatted values.
    tol_days:
        Maximum allowed span in days between the earliest and latest parsed
        date for the values to be considered matching.
    """

    try:
        effective_tol = int(tol_days)
    except (TypeError, ValueError):
        effective_tol = DEFAULT_DATE_TOLERANCE_DAYS
    else:
        if effective_tol < 0:
            effective_tol = DEFAULT_DATE_TOLERANCE_DAYS

    parsed = [parse_date_with_convention(value, conv) for value in values]
    valid_values = [dt for dt in parsed if dt is not None]

    if len(valid_values) < 2:
        return True, None

    min_date = min(valid_values)
    max_date = max(valid_values)
    span_days = (max_date - min_date).days
    return span_days <= effective_tol, span_days


def _normalize_convention(value: Any) -> Optional[str]:
    """Return a normalised convention value if valid, otherwise ``None``."""

    if isinstance(value, str):
        candidate = value.strip().upper()
        if candidate in _VALID_CONVENTIONS:
            return candidate
    return None


def _extract_convention(payload: Any) -> Optional[str]:
    """Extract the convention value from a parsed JSON payload."""

    if not isinstance(payload, Mapping):
        return None

    for key in ("convention", "conv"):
        if key in payload:
            convention = _normalize_convention(payload[key])
            if convention is not None:
                return convention
    return None


def _normalize_rel_path(value: Any) -> Optional[str]:
    """Return a normalised relative path string if ``value`` is usable."""

    if isinstance(value, str):
        text = value.strip()
        if text:
            return text.replace("\\", "/")
    return None


def _extract_rel_path_from_manifest(manifest: Mapping[str, Any]) -> Optional[str]:
    """Return the date convention relative path from ``manifest`` if present."""

    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, Mapping):
        traces = artifacts.get("traces")
        if isinstance(traces, Mapping):
            rel_candidate = _normalize_rel_path(traces.get("date_convention_rel"))
            if rel_candidate:
                return rel_candidate

    prevalidation = manifest.get("prevalidation")
    if isinstance(prevalidation, Mapping):
        block = prevalidation.get("date_convention")
        if isinstance(block, Mapping):
            for key in ("file_rel", "relative_path", "path"):
                rel_candidate = _normalize_rel_path(block.get(key))
                if rel_candidate:
                    return rel_candidate

    return None


def _manifest_rel_path(runs_root: Path, sid: str) -> Optional[str]:
    """Return the relative path recorded in the manifest for ``sid`` if any."""

    manifest_path = runs_root / sid / "manifest.json"
    try:
        raw_text = manifest_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        _LOGGER.debug(
            "Failed to read manifest for date convention path at %s: %s",
            manifest_path,
            exc,
        )
        return None

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        _LOGGER.debug(
            "Manifest JSON invalid while resolving date convention path at %s: %s",
            manifest_path,
            exc,
        )
        return None

    if not isinstance(payload, Mapping):
        return None

    rel_path = _extract_rel_path_from_manifest(payload)
    if rel_path:
        _LOGGER.debug("Using manifest date convention path for sid=%s: %s", sid, rel_path)
        return rel_path

    return None


def load_date_convention_for_sid(runs_root: str, sid: str, rel_path: str | None = None) -> str:
    """Load the date parsing convention for a given run.

    Parameters
    ----------
    runs_root:
        Absolute path to the directory containing run folders.
    sid:
        Identifier for the specific run whose convention we want to read.
    rel_path:
        Optional relative path to the convention file. When ``None`` the
        ``PREVALIDATION_OUT_PATH_REL`` environment variable (via
        :func:`get_prevalidation_trace_relpath`) determines the location.

    Returns
    -------
    str
        The date convention value (``"DMY"``, ``"MDY"`` or ``"YMD"``). If the
        file is missing or contains an invalid payload the default ``"MDY"`` is
        returned.
    """

    runs_root_path = Path(runs_root)
    sid_root = runs_root_path / sid

    resolved_rel_path = rel_path
    if resolved_rel_path is None:
        manifest_rel = _manifest_rel_path(runs_root_path, sid)
        if manifest_rel is not None:
            resolved_rel_path = manifest_rel

    if resolved_rel_path is None:
        resolved_rel_path = get_prevalidation_trace_relpath()

    normalized_rel = _normalize_rel_path(resolved_rel_path) or get_prevalidation_trace_relpath()

    rel_path_obj = Path(normalized_rel)
    if rel_path_obj.is_absolute():
        convention_path = rel_path_obj
    else:
        convention_path = sid_root / rel_path_obj

    try:
        data = json.loads(convention_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _LOGGER.debug("Date convention file not found at %s", convention_path)
        return "MDY"
    except (OSError, json.JSONDecodeError) as exc:
        _LOGGER.debug("Failed to read date convention file at %s: %s", convention_path, exc)
        return "MDY"

    convention = _extract_convention(data)
    if convention is None:
        _LOGGER.debug("Date convention not present or invalid in %s", convention_path)
        return "MDY"

    return convention


__all__ = [
    "load_date_convention_for_sid",
    "parse_date_with_convention",
    "are_dates_within_tolerance",
]
