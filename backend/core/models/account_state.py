from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple

from backend.outcomes.models import OutcomeEvent


class AccountStatus(str, Enum):
    PLANNED = "Planned"
    SENT = "Sent"
    CRA_RESPONDED_VERIFIED = "CRA_RespondedVerified"
    CRA_RESPONDED_UPDATED = "CRA_RespondedUpdated"
    CRA_RESPONDED_DELETED = "CRA_RespondedDeleted"
    CRA_RESPONDED_NOCHANGE = "CRA_RespondedNoChange"
    COMPLETED = "Completed"


@dataclass
class StateTransition:
    """Record of a state change for auditing."""

    from_status: AccountStatus
    to_status: AccountStatus
    actor: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AccountState:
    """Tracks the current state of an account in the strategy pipeline."""

    account_id: str
    current_cycle: int
    current_step: int
    status: AccountStatus
    last_sent_at: Optional[datetime] = None
    next_eligible_at: Optional[datetime] = None
    history: List[StateTransition] = field(default_factory=list)
    last_outcome: str | None = None
    resolution_cycle_count: int = 0
    outcome_history: Tuple[OutcomeEvent, ...] = field(default_factory=tuple)

    def transition(self, to_status: AccountStatus, actor: str) -> None:
        """Update status and append a transition record."""

        self.history.append(
            StateTransition(from_status=self.status, to_status=to_status, actor=actor)
        )
        self.status = to_status

    def record_outcome(self, event: OutcomeEvent) -> None:
        """Record an outcome event in the account's history."""

        outcome = event.outcome.value if hasattr(event.outcome, "value") else event.outcome
        self.last_outcome = outcome
        self.outcome_history = (*self.outcome_history, event)
        self.resolution_cycle_count = len(self.outcome_history)
