import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List
from uuid import uuid4

from backend.api import session_manager
from backend.audit.audit import set_log_context


class Outcome(str, Enum):
    """Possible CRA responses."""

    VERIFIED = "Verified"
    UPDATED = "Updated"
    DELETED = "Deleted"
    NOCHANGE = "NoChange"


@dataclass(frozen=True)
class OutcomeEvent:
    """Record produced when a bureau responds to a dispute."""

    outcome_id: str
    account_id: str
    cycle_id: int
    family_id: str
    outcome: Outcome | str
    raw_report_ref: str | None = None
    diff_snapshot: Dict[str, Any] | None = None
    audit_id: str = field(default_factory=lambda: str(uuid4()))


def save_outcome_event(session_id: str, event: OutcomeEvent) -> None:
    """Append an outcome event to persistent session storage."""

    stored = session_manager.get_session(session_id) or {}
    history: Dict[str, List[Dict[str, Any]]] = stored.get("outcome_history", {}) or {}
    events = history.get(event.account_id, [])
    record = asdict(event)
    # Store enum value as plain string
    if isinstance(record.get("outcome"), Outcome):
        record["outcome"] = record["outcome"].value
    events.append(record)
    history[event.account_id] = events
    session_manager.update_session(session_id, outcome_history=history)
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


def load_outcome_history(session_id: str, account_id: str) -> List[OutcomeEvent]:
    """Load outcome events for an account from session storage."""

    stored = session_manager.get_session(session_id) or {}
    history: Dict[str, List[Dict[str, Any]]] = stored.get("outcome_history", {}) or {}
    events = history.get(account_id, [])
    return [OutcomeEvent(**e) for e in events]
