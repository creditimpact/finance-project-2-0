from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
from typing import List, Tuple

from backend.telemetry.metrics import emit_counter
from backend.core.models import AccountState, AccountStatus, StateTransition
from backend.outcomes.models import OutcomeEvent

"""Simple finite-state machine for account planning.

This module defines allowed transitions for ``AccountState`` records and
imposes SLA gates between steps. ``evaluate_state`` returns a tuple of
``(allowed_tags, next_eligible_at)`` describing the actions that may be
performed now and, if no action is currently permitted, the next time the
account becomes eligible.
"""

# Number of days that must elapse before the next action is allowed after
# letters are sent to the bureaus.
_DEFAULT_SLA_DAYS = 30


def _serialize_dt(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _deserialize_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def record_wait_time(state: AccountState, now: datetime, eligible_at: datetime) -> None:
    """Record time until the next eligible action for ``state``."""

    delta_ms = (eligible_at - now).total_seconds() * 1000
    emit_counter("planner.time_to_next_step_ms", delta_ms)
    emit_counter(
        f"planner.time_to_next_step_ms.cycle.{state.current_cycle}", delta_ms
    )
    emit_counter(
        f"planner.time_to_next_step_ms.step.{state.current_step}", delta_ms
    )


def evaluate_state(
    state: AccountState, now: datetime | None = None
) -> Tuple[List[str], datetime | None]:
    """Evaluate the state machine for a single account.

    Args:
        state: The account state to evaluate.
        now:   Optional override for the current time.

    Returns:
        A tuple ``(allowed_tags, next_eligible_at)``. ``allowed_tags`` is a
        list of planner tags that may be executed immediately.  If the list is
        empty, ``next_eligible_at`` contains the datetime when the account will
        become eligible for the next step.
    """

    now = now or datetime.utcnow()
    if state.status == AccountStatus.PLANNED:
        # Initial cycle – we can send the first dispute letter immediately.
        return ["dispute"], None

    if state.status == AccountStatus.SENT:
        # We must wait for the SLA period before allowing a follow-up.
        if state.last_sent_at:
            eligible_at = state.last_sent_at + timedelta(days=_DEFAULT_SLA_DAYS)
            if now >= eligible_at:
                return ["followup"], None
            record_wait_time(state, now, eligible_at)
            return [], eligible_at
        return ["followup"], None

    # Any terminal state – no further actions are allowed.
    return [], None


def dump_state(state: AccountState) -> dict:
    """Serialize ``AccountState`` for storage in the session manager."""

    data = asdict(state)
    data["last_sent_at"] = _serialize_dt(state.last_sent_at)
    data["next_eligible_at"] = _serialize_dt(state.next_eligible_at)
    for hist in data.get("history", []):
        hist["timestamp"] = _serialize_dt(hist.get("timestamp"))
    for ev in data.get("outcome_history", []):
        if hasattr(ev.get("outcome"), "value"):
            ev["outcome"] = ev["outcome"].value
    return data


def load_state(data: dict) -> AccountState:
    """Deserialize an ``AccountState`` from session data."""

    data = dict(data)
    if isinstance(data.get("last_sent_at"), str):
        data["last_sent_at"] = _deserialize_dt(data["last_sent_at"])
    if isinstance(data.get("next_eligible_at"), str):
        data["next_eligible_at"] = _deserialize_dt(data["next_eligible_at"])
    hist = []
    for item in data.get("history", []):
        ts = item.get("timestamp")
        if isinstance(ts, str):
            ts = _deserialize_dt(ts)
        hist.append(
            StateTransition(
                from_status=AccountStatus(item["from_status"]),
                to_status=AccountStatus(item["to_status"]),
                actor=item.get("actor", "unknown"),
                timestamp=ts if ts else datetime.utcnow(),
            )
        )
    data["history"] = hist
    events = []
    for item in data.get("outcome_history", []):
        events.append(OutcomeEvent(**item))
    data["outcome_history"] = tuple(events)
    return AccountState(**data)
