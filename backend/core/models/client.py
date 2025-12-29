from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Type


@dataclass
class ClientInfo:
    """Client metadata used across the workflow.

    This model replaces the previous free-form ``dict`` usage. Additional
    attributes are stored in ``extras``.
    """

    name: str | None = None
    legal_name: str | None = None
    address: str | None = None
    email: str | None = None
    state: str | None = None
    goal: str | None = None
    session_id: str = "session"
    structured_summaries: Any = None
    account_inquiry_matches: List[Dict[str, Any]] | None = None
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls: Type["ClientInfo"], data: Dict[str, Any]) -> "ClientInfo":
        known_keys = {
            "name",
            "legal_name",
            "address",
            "email",
            "state",
            "goal",
            "session_id",
            "structured_summaries",
            "account_inquiry_matches",
        }
        extras = {k: v for k, v in data.items() if k not in known_keys}
        return cls(
            name=data.get("name"),
            legal_name=data.get("legal_name"),
            address=data.get("address"),
            email=data.get("email"),
            state=data.get("state"),
            goal=data.get("goal"),
            session_id=data.get("session_id", "session"),
            structured_summaries=data.get("structured_summaries"),
            account_inquiry_matches=data.get("account_inquiry_matches"),
            extras=extras,
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        extras = data.pop("extras", {})
        data.update(extras)
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class ProofDocuments:
    """Paths to uploaded or generated proof documents."""

    smartcredit_report: str
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls: Type["ProofDocuments"], data: Dict[str, Any]
    ) -> "ProofDocuments":
        known = {"smartcredit_report"}
        extras = {k: v for k, v in data.items() if k not in known}
        return cls(smartcredit_report=data.get("smartcredit_report", ""), extras=extras)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        extras = data.pop("extras", {})
        data.update(extras)
        return data
