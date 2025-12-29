"""Planner entry points."""

from __future__ import annotations

import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List

from backend.analytics.analytics_tracker import set_metric
from backend.telemetry.metrics import emit_counter
from backend.api.session_manager import get_session, update_session
from backend.audit.audit import emit_event, set_log_context
from backend.core.locks import account_lock
from backend.core.models import AccountState, AccountStatus
from backend.outcomes import Outcome, OutcomeEvent

from .state_machine import dump_state, evaluate_state, load_state, record_wait_time


def _ensure_account_states(
    session: dict, stored_states: Dict[str, dict]
) -> Dict[str, dict]:
    """Ensure every account in the current strategy has a tracked state."""

    strategy = session.get("strategy", {}) or {}
    accounts = strategy.get("accounts", [])
    for acc in accounts:
        acc_id = str(acc.get("account_id") or "")
        if acc_id and acc_id not in stored_states:
            state = AccountState(
                account_id=acc_id,
                current_cycle=0,
                current_step=0,
                status=AccountStatus.PLANNED,
            )
            stored_states[acc_id] = dump_state(state)
    return stored_states


def plan_next_step(
    session: dict, action_tags: Iterable[str], now: datetime | None = None
) -> List[str]:
    """Evaluate the FSM for all accounts and persist results.

    The function loads persisted ``AccountState`` objects for the provided
    ``session``, evaluates them via the finite-state machine and stores the
    updated state back to the session manager *before* any tactical side
    effects occur.  ``action_tags`` contains the Stage 2.5 tags that the
    strategist proposed for this run; the planner intersects these with the
    state machine's allowed tags to enforce cycle/SLA restrictions.

    Args:
        session: A mapping containing at least ``session_id`` and ``strategy``.
        action_tags: Iterable of action tags proposed by the strategist.

    Returns:
        A sorted list of planner-approved tags for the current step.
    """

    try:
        session_id = session.get("session_id")
        if not session_id:
            return []

        stored = get_session(session_id) or {}
        states_data: Dict[str, dict] = stored.get("account_states", {}) or {}
        states_data = _ensure_account_states(session, states_data)

        allowed: List[str] = []
        now = now or datetime.utcnow()
        total_accounts = len(states_data)
        resolved_accounts = 0
        resolved_cycles = 0
        for acc_id, data in states_data.items():
            with account_lock(acc_id):
                state = load_state(data)
                if state.status == AccountStatus.COMPLETED:
                    resolved_accounts += 1
                    resolved_cycles += state.current_cycle
                    states_data[acc_id] = dump_state(state)
                    continue

                tags: List[str] = []
                next_eligible_at = state.next_eligible_at

                outcome = (state.last_outcome or "").lower()
                if outcome == "verified":
                    tags = ["mov", "direct_dispute"]
                    next_eligible_at = None
                elif outcome == "updated":
                    tags = ["bureau_dispute"]
                    next_eligible_at = None
                elif outcome == "nochange":
                    if next_eligible_at and now < next_eligible_at:
                        record_wait_time(state, now, next_eligible_at)
                        states_data[acc_id] = dump_state(state)
                        continue
                    tags = ["mov"]
                    next_eligible_at = None
                elif state.next_eligible_at and now < state.next_eligible_at:
                    record_wait_time(state, now, state.next_eligible_at)
                    states_data[acc_id] = dump_state(state)
                    continue
                else:
                    tags, next_eligible_at = evaluate_state(state, now=now)

                state.next_eligible_at = next_eligible_at
                if next_eligible_at and now < next_eligible_at:
                    record_wait_time(state, now, next_eligible_at)
                states_data[acc_id] = dump_state(state)
                allowed.extend(tags)

        if total_accounts:
            set_metric(
                "planner.resolution_rate",
                resolved_accounts / total_accounts,
            )
        if resolved_accounts:
            set_metric(
                "planner.avg_cycles_per_resolution",
                resolved_cycles / resolved_accounts,
            )

        action_set = {t for t in action_tags if t}
        if action_set:
            allowed = [t for t in allowed if t in action_set]

        update_session(session_id, account_states=states_data)
        return sorted(set(allowed))
    except Exception:
        emit_counter("planner.error_count")
        raise


def record_send(
    session: dict,
    account_ids: Iterable[str],
    now: datetime | None = None,
    sla_days: int = 30,
) -> None:
    """Record that letters were sent for the given accounts."""

    try:
        session_id = session.get("session_id")
        if not session_id:
            return

        stored = get_session(session_id) or {}
        states_data: Dict[str, dict] = stored.get("account_states", {}) or {}
        if not states_data:
            return

        now = now or datetime.utcnow()
        for acc_id in account_ids:
            data = states_data.get(str(acc_id))
            if not data:
                continue
            state = load_state(data)
            if state.last_sent_at and now > state.last_sent_at + timedelta(
                days=sla_days
            ):
                emit_counter("planner.sla_violations_total")
            state.last_sent_at = now
            state.next_eligible_at = now + timedelta(days=sla_days)
            state.transition(AccountStatus.SENT, actor="planner")
            state.current_step += 1
            emit_counter(
                "planner.cycle_progress",
                {"cycle": state.current_cycle, "step": state.current_step},
            )
            record_wait_time(state, now, state.next_eligible_at)
            emit_event(
                "audit.planner_transition",
                {
                    "account_id": str(acc_id),
                    "cycle": state.current_cycle,
                    "step": state.current_step,
                    "reason": "letters_sent",
                },
                extra={"cycle_id": state.current_cycle},
            )
            states_data[str(acc_id)] = dump_state(state)

        update_session(session_id, account_states=states_data)
    except Exception:
        emit_counter("planner.error_count")
        raise


def handle_outcome(
    session: dict, event: OutcomeEvent, now: datetime | None = None, sla_days: int = 30
) -> List[str]:
    """Update account state based on a bureau outcome and suggest next steps.

    Returns a list of planner tags that are immediately allowed as a result of
    the outcome. If no immediate action is permitted, an empty list is
    returned and ``next_eligible_at`` is set on the account state.
    """

    try:
        session_id = session.get("session_id")
        if not session_id:
            return []

        with account_lock(str(event.account_id)):
            stored = get_session(session_id) or {}
            states_data: Dict[str, dict] = stored.get("account_states", {}) or {}
            history: Dict[str, List[Dict[str, Any]]] = stored.get("outcome_history", {}) or {}
            data = states_data.get(str(event.account_id))
            if not data:
                events = history.get(str(event.account_id), [])
                record = asdict(event)
                if isinstance(record.get("outcome"), Outcome):
                    record["outcome"] = record["outcome"].value
                events.append(record)
                history[str(event.account_id)] = events
                update_session(session_id, outcome_history=history)
                set_log_context(
                    session_id=session_id,
                    family_id=event.family_id,
                    cycle_id=event.cycle_id,
                    audit_id=event.audit_id,
                )
                logging.getLogger(__name__).info(
                    "persisted_outcome_event",
                    extra={
                        "audit_id": event.audit_id,
                        "diff": event.diff_snapshot,
                        "session_id": session_id,
                        "family_id": event.family_id,
                        "cycle_id": event.cycle_id,
                    },
                )
                return []

            state = load_state(data)
            outcome_val = (
                event.outcome.value.lower()
                if hasattr(event.outcome, "value")
                else str(event.outcome).lower()
            )

            allowed_tags: List[str] = []
            now = now or datetime.utcnow()

            if outcome_val == "verified":
                state.transition(AccountStatus.CRA_RESPONDED_VERIFIED, actor="cra")
                allowed_tags = ["mov", "direct_dispute"]
                state.next_eligible_at = None
            elif outcome_val == "updated":
                state.transition(AccountStatus.CRA_RESPONDED_UPDATED, actor="cra")
                allowed_tags = ["bureau_dispute"]
                state.next_eligible_at = None
            elif outcome_val == "deleted":
                state.transition(AccountStatus.CRA_RESPONDED_DELETED, actor="cra")
                state.transition(AccountStatus.COMPLETED, actor="system")
                state.current_cycle += 1
                state.current_step = 0
                state.next_eligible_at = None
            elif outcome_val == "nochange":
                state.transition(AccountStatus.CRA_RESPONDED_NOCHANGE, actor="cra")
                state.next_eligible_at = (state.last_sent_at or now) + timedelta(
                    days=sla_days
                )
            else:
                return []

            state.record_outcome(event)

            events = history.get(str(event.account_id), [])
            record = asdict(event)
            if isinstance(record.get("outcome"), Outcome):
                record["outcome"] = record["outcome"].value
            events.append(record)
            history[str(event.account_id)] = events
            states_data[str(event.account_id)] = dump_state(state)
            update_session(
                session_id, account_states=states_data, outcome_history=history
            )
            set_log_context(
                session_id=session_id,
                family_id=event.family_id,
                cycle_id=event.cycle_id,
                audit_id=event.audit_id,
            )
            logging.getLogger(__name__).info(
                "persisted_outcome_event",
                extra={
                    "audit_id": event.audit_id,
                    "diff": event.diff_snapshot,
                    "session_id": session_id,
                    "family_id": event.family_id,
                    "cycle_id": event.cycle_id,
                },
            )
            return allowed_tags
    except Exception:
        emit_counter("planner.error_count")
        raise
