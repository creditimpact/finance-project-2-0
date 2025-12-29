from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Type

from .account import Inquiry
from .problem_account import ProblemAccount


@dataclass
class BureauPayload:
    """Structured data for a single bureau.

    Replaces the previous free-form ``dict`` layout used across the codebase.
    """

    disputes: List[ProblemAccount] = field(default_factory=list)
    goodwill: List[ProblemAccount] = field(default_factory=list)
    inquiries: List[Inquiry] = field(default_factory=list)
    high_utilization: List[ProblemAccount] = field(default_factory=list)

    @classmethod
    def from_dict(cls: Type["BureauPayload"], data: Dict[str, Any]) -> "BureauPayload":
        return cls(
            disputes=[
                BureauAccount.from_dict(d) if isinstance(d, dict) else d
                for d in data.get("disputes", [])
            ],
            goodwill=[
                BureauAccount.from_dict(d) if isinstance(d, dict) else d
                for d in data.get("goodwill", [])
            ],
            inquiries=[
                Inquiry.from_dict(i) if isinstance(i, dict) else i
                for i in data.get("inquiries", [])
            ],
            high_utilization=[
                ProblemAccount.from_dict(d) if isinstance(d, dict) else d
                for d in data.get("high_utilization", [])
            ],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "disputes": [a.to_dict() for a in self.disputes],
            "goodwill": [a.to_dict() for a in self.goodwill],
            "inquiries": [i.to_dict() for i in self.inquiries],
            "high_utilization": [a.to_dict() for a in self.high_utilization],
        }


@dataclass
class BureauAccount(ProblemAccount):
    """Account entry associated with a specific credit bureau."""

    bureau: Optional[str] = None
    section: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BureauAccount":
        data = {k: v for k, v in data.items() if k not in {"bureaus"}}
        base = ProblemAccount.from_dict(
            {k: v for k, v in data.items() if k not in {"bureau", "section"}}
        )
        return cls(
            **asdict(base),
            bureau=data.get("bureau"),
            section=data.get("section"),
        )

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({"bureau": self.bureau, "section": self.section})
        return d


@dataclass
class BureauSection:
    """Collection of accounts belonging to a report section."""

    name: str
    accounts: List[BureauAccount] = field(default_factory=list)

    @classmethod
    def from_dict(cls, name: str, data: List[dict[str, Any]]) -> "BureauSection":
        return cls(name=name, accounts=[BureauAccount.from_dict(d) for d in data])

    def to_dict(self) -> dict[str, Any]:
        return {self.name: [acc.to_dict() for acc in self.accounts]}
