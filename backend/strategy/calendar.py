"""Business-day to calendar-day conversion helpers for the planner."""

from __future__ import annotations

import calendar as _calendar
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
from zoneinfo import ZoneInfo


def advance_business_days(
    start_weekday: int,
    business_days: int,
    weekend: Set[int],
    *,
    holidays: Optional[Set[date]] = None,
) -> tuple[int, int]:
    """Advance ``business_days`` from ``start_weekday`` respecting ``weekend``.

    Returns a tuple of ``(end_weekday, calendar_days)`` where ``calendar_days`` is the
    total elapsed number of calendar days required to satisfy ``business_days``.
    ``start_weekday`` and ``end_weekday`` use ISO numbering (0=Mon..6=Sun).

    ``holidays`` are accepted for forward compatibility but currently ignored because the
    planner operates on relative weekdays without an absolute start date.
    """

    business_days = max(business_days, 0)
    if business_days == 0:
        return start_weekday % 7, 0

    weekend = {day % 7 for day in weekend}

    current = start_weekday % 7
    remaining = business_days
    elapsed = 0

    while remaining > 0:
        current = (current + 1) % 7
        elapsed += 1
        if current in weekend:
            continue
        remaining -= 1

    return current, elapsed


def sum_calendar_span(
    order_fields: Sequence[str],
    field_to_min_days: Dict[str, int],
    start_weekday: int,
    weekend: Iterable[int],
) -> int:
    """Return total calendar span for executing ``order_fields``.

    After completing each segment the clock advances to the next business day if the
    computed end falls on a weekend, ensuring submissions only occur on weekdays.
    """

    weekend_set = {day % 7 for day in weekend}
    total_calendar_days = 0
    current_weekday = start_weekday % 7

    total_fields = len(order_fields)
    for index, field in enumerate(order_fields):
        min_days = max(int(field_to_min_days.get(field, 0)), 0)
        current_weekday, elapsed = advance_business_days(current_weekday, min_days, weekend_set)
        total_calendar_days += elapsed

        if index == total_fields - 1:
            continue
        while current_weekday in weekend_set:
            current_weekday = (current_weekday + 1) % 7
            total_calendar_days += 1

    return total_calendar_days


def weekend_from_env(csv: str | None) -> Set[int]:
    """Parse a comma-separated list of weekend days into a set of integers."""

    if not csv or not csv.strip():
        return {5, 6}

    weekend: Set[int] = set()
    for token in csv.split(","):
        part = token.strip()
        if not part:
            continue
        try:
            value = int(part)
        except ValueError as exc:
            raise ValueError(f"Invalid weekend day '{part}' in PLANNER_WEEKEND") from exc
        if value < 0 or value > 6:
            raise ValueError("Weekend day must be between 0 and 6 inclusive")
        weekend.add(value % 7)

    return weekend or {5, 6}


def holidays_from_env(
    source: str | None,
    static_list: str | None,
    region: str | None,
) -> Optional[Set[date]]:
    """Return a set of static holiday dates when configured, otherwise ``None``."""

    normalized_source = (source or "").strip().lower()
    if not normalized_source or normalized_source == "none":
        return None

    if normalized_source == "static":
        holidays: Set[date] = set()
        for token in (static_list or "").split(","):
            value = token.strip()
            if not value:
                continue
            try:
                holidays.add(datetime.strptime(value, "%Y-%m-%d").date())
            except ValueError as exc:
                raise ValueError(f"Invalid holiday date '{value}'") from exc
        return holidays or None

    # Region-based holiday sources are not yet implemented.
    return None


def next_occurrence_of_weekday(from_dt: datetime, target_weekday: int, tz: ZoneInfo) -> date:
    """Return the next date (inclusive) matching ``target_weekday`` in ``tz``."""

    localized = from_dt.astimezone(tz)
    current_weekday = localized.weekday()
    delta = (target_weekday - current_weekday) % 7
    candidate = localized.date() + timedelta(days=delta)
    return candidate


def _is_business_day(day: date, weekend: Set[int], holidays: Set[date]) -> bool:
    return day.weekday() not in weekend and day not in holidays


def _advance_to_business_day(day: date, weekend: Set[int], holidays: Set[date]) -> date:
    current = day
    while not _is_business_day(current, weekend, holidays):
        current += timedelta(days=1)
    return current


def advance_business_days_date(
    start: date,
    business_days: int,
    weekend: Set[int],
    holidays: Optional[Set[date]] = None,
) -> date:
    """Advance ``business_days`` from ``start`` over real calendar dates."""

    holidays_set = holidays or set()
    current = start
    remaining = max(int(business_days or 0), 0)
    while remaining > 0:
        current += timedelta(days=1)
        if not _is_business_day(current, weekend, holidays_set):
            continue
        remaining -= 1
    return current


def add_business_days_date(
    start: date,
    business_days: int,
    weekend: Set[int],
    holidays: Optional[Set[date]] = None,
) -> date:
    """Alias of :func:`advance_business_days_date` for readability."""

    return advance_business_days_date(start, business_days, weekend, holidays)


def subtract_business_days_date(
    start: date,
    business_days: int,
    weekend: Set[int],
    holidays: Optional[Set[date]] = None,
) -> date:
    """Move ``business_days`` backwards from ``start`` honoring weekend/holidays."""

    holidays_set = holidays or set()
    current = start
    remaining = max(int(business_days or 0), 0)
    while remaining > 0:
        current -= timedelta(days=1)
        if not _is_business_day(current, weekend, holidays_set):
            continue
        remaining -= 1
    return current


def business_days_between(
    start: date,
    end: date,
    weekend: Set[int],
    holidays: Optional[Set[date]] = None,
) -> int:
    """Return the number of business days from ``start`` (exclusive) to ``end`` (inclusive)."""

    if start >= end:
        return 0
    holidays_set = holidays or set()
    current = start
    count = 0
    while current < end:
        current += timedelta(days=1)
        if _is_business_day(current, weekend, holidays_set):
            count += 1
    return count


def roll_if_weekend(
    candidate: date,
    weekend: Set[int],
    holidays: Optional[Set[date]] = None,
) -> tuple[date, Optional[str], List[tuple[date, str]]]:
    """Roll ``candidate`` forward to the next business day if blocked."""

    holidays_set = holidays or set()
    blocked: List[tuple[date, str]] = []
    current = candidate
    reason: Optional[str] = None

    while current.weekday() in weekend or current in holidays_set:
        if current.weekday() in weekend:
            blocked.append((current, "weekend"))
            if reason is None:
                reason = "shifted_to_avoid_weekend"
        else:
            blocked.append((current, "holiday"))
            if reason is None:
                reason = "shifted_due_to_holiday"
        current += timedelta(days=1)

    return current, reason, blocked


def find_business_day_in_window(
    anchor: date,
    window: Tuple[int, int],
    weekend: Set[int],
    holidays: Optional[Set[date]] = None,
) -> tuple[date, int, Dict[str, object]]:
    """Select a business day near the preferred window for the planner."""

    start, end = window
    if start < 0:
        start = 0
    if end < start:
        start, end = end, start

    weekend_set = {day % 7 for day in weekend}
    holidays_set = holidays or set()

    preference_offsets = list(range(end, start - 1, -1)) if end >= start else [start]
    attempted_index = preference_offsets[0]
    blocked: List[Dict[str, object]] = []

    for offset in preference_offsets:
        candidate = anchor + timedelta(days=offset)
        if candidate.weekday() in weekend_set or candidate in holidays_set:
            reason = "weekend" if candidate.weekday() in weekend_set else "holiday"
            blocked.append({"offset": offset, "reason": reason, "date": candidate.isoformat()})
            continue
        return candidate, offset, {
            "within_window": True,
            "window_available": True,
            "fallback_lower": False,
            "blocked_offsets": tuple(entry["offset"] for entry in blocked),
            "blocked_details": tuple(blocked),
            "preferred_index": attempted_index,
            "attempted_index": attempted_index,
        }

    for offset in range(end, -1, -1):
        candidate = anchor + timedelta(days=offset)
        if candidate.weekday() in weekend_set or candidate in holidays_set:
            reason = "weekend" if candidate.weekday() in weekend_set else "holiday"
            entry = {"offset": offset, "reason": reason, "date": candidate.isoformat()}
            if start <= offset <= end:
                blocked.append(entry)
            continue
        return candidate, offset, {
            "within_window": offset >= start,
            "window_available": False,
            "fallback_lower": offset < start,
            "blocked_offsets": tuple(entry["offset"] for entry in blocked),
            "blocked_details": tuple(blocked),
            "preferred_index": attempted_index,
            "attempted_index": attempted_index,
        }

    rolled, _, blocked_dates = roll_if_weekend(anchor, weekend_set, holidays_set)
    index = (rolled - anchor).days
    return rolled, index, {
        "within_window": index >= start,
        "window_available": False,
        "fallback_lower": True,
        "blocked_offsets": tuple(entry["offset"] for entry in blocked),
        "blocked_details": tuple(blocked),
        "blocked_dates": tuple(blocked_dates),
        "preferred_index": attempted_index,
        "attempted_index": attempted_index,
    }


def anchor_and_expand_sequence(
    anchor_weekday: int,
    run_dt: datetime,
    tz: ZoneInfo,
    weekend: Iterable[int],
    holidays: Optional[Set[date]],
    items: Sequence[Mapping[str, object]],
    *,
    handoff_offset: int = 1,
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    """Anchor the canonical sequence to real dates and return plan + logs."""

    weekend_set = {int(day) % 7 for day in weekend}
    if not weekend_set:
        weekend_set = {5, 6}
    holidays_set = set(holidays or set())

    anchor_date = next_occurrence_of_weekday(run_dt, anchor_weekday % 7, tz)
    anchor_date = _advance_to_business_day(anchor_date, weekend_set, holidays_set)
    anchor_reason = (
        f"next occurrence of weekday {anchor_weekday % 7} ({_calendar.day_name[anchor_weekday % 7][:3]}) at/after run time"
    )

    sequence: List[Dict[str, object]] = []
    schedule_logs: List[Dict[str, object]] = []

    first_submit: Optional[date] = None
    last_sla_end: Optional[date] = None

    previous_submit: Optional[date] = None
    previous_min_days: int = 0

    def _serialize_day(day: date) -> Dict[str, object]:
        weekday = day.weekday()
        return {
            "date": day.isoformat(),
            "weekday": weekday,
            "weekday_name": _calendar.day_abbr[weekday],
        }

    for idx, item in enumerate(items, start=1):
        min_days = max(int(item.get("min_days", 0)), 0)
        role = str(item.get("role", ""))
        field = str(item.get("field", ""))
        score = item.get("score")
        why_here = str(item.get("why_here", ""))

        if idx == 1:
            submit_date = anchor_date
            handoff_reason = "anchor submission"
        else:
            delta = max(previous_min_days - handoff_offset, 0)
            submit_candidate = advance_business_days_date(previous_submit or anchor_date, delta, weekend_set, holidays_set)
            submit_date = _advance_to_business_day(submit_candidate, weekend_set, holidays_set)
            if previous_min_days <= handoff_offset:
                handoff_reason = "previous SLA already due; chained immediately"
            else:
                handoff_reason = f"handoff -{handoff_offset} business day from previous SLA"

        sla_end = advance_business_days_date(submit_date, min_days, weekend_set, holidays_set)

        sequence.append(
            {
                "idx": idx,
                "field": field,
                "role": role,
                "min_days": min_days,
                "submit_on": _serialize_day(submit_date),
                "sla_window": {
                    "start": _serialize_day(submit_date),
                    "end": _serialize_day(sla_end),
                },
                "explainer": {
                    "why_here": why_here,
                    "handoff_rule": "N/A (first item)" if idx == 1 else f"Next starts at ({previous_min_days} - {handoff_offset}) business days from previous submit",
                    "score": score or {},
                },
            }
        )

        schedule_logs.append(
            {
                "event": "schedule_item",
                "idx": idx,
                "field": field,
                "submit_on": submit_date.isoformat(),
                "weekday": submit_date.weekday(),
                "reason": handoff_reason,
            }
        )

        previous_submit = submit_date
        previous_min_days = min_days

        if first_submit is None:
            first_submit = submit_date
        last_sla_end = sla_end

    if first_submit and last_sla_end:
        span_days = max((last_sla_end - first_submit).days, 0)
    else:
        span_days = 0

    plan_payload: Dict[str, object] = {
        "schema_version": 1,
        "anchor": {
            "weekday": anchor_weekday % 7,
            "date": anchor_date.isoformat(),
            "reason": anchor_reason,
        },
        "timezone": tz.key,
        "sequence": sequence,
        "calendar_span_days": span_days,
    }

    return plan_payload, schedule_logs
