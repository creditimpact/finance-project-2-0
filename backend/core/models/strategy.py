from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional


@dataclass
class Recommendation:
    """Strategist recommendation details."""

    action_tag: Optional[str] = None
    recommended_action: Optional[str] = None
    advisor_comment: Optional[str] = None
    flags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recommendation":
        return cls(
            action_tag=data.get("action_tag"),
            recommended_action=data.get("recommended_action"),
            advisor_comment=data.get("advisor_comment"),
            flags=list(data.get("flags", []) or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyItem:
    """Strategy recommendation for a single account."""

    account_id: str
    name: str
    account_number: Optional[str] = None
    recommendation: Recommendation | None = None
    legal_safe_summary: Optional[str] = None
    suggested_dispute_frame: Optional[str] = None
    rule_hits: List[str] = field(default_factory=list)
    needs_evidence: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyItem":
        return cls(
            account_id=str(data.get("account_id") or ""),
            name=data.get("name", ""),
            account_number=data.get("account_number"),
            recommendation=Recommendation.from_dict(data) if data else None,
            legal_safe_summary=data.get("legal_safe_summary"),
            suggested_dispute_frame=data.get("suggested_dispute_frame"),
            rule_hits=list(data.get("rule_hits", []) or []),
            needs_evidence=list(data.get("needs_evidence", []) or []),
            red_flags=list(data.get("red_flags", []) or []),
        )

    def to_dict(self) -> dict[str, Any]:
        base = asdict(self)
        if self.recommendation is not None:
            base["recommendation"] = self.recommendation.to_dict()
        return base


@dataclass
class StrategyPlan:
    """Container for all strategy recommendations."""

    accounts: List[StrategyItem] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyPlan":
        accounts = [StrategyItem.from_dict(d) for d in data.get("accounts", [])]
        return cls(accounts=accounts)

    def to_dict(self) -> dict[str, Any]:
        return {"accounts": [item.to_dict() for item in self.accounts]}
