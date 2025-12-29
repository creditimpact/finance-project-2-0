from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Type


@dataclass
class ProblemAccount:
    """Simplified account model used in Stage A responses."""

    name: str
    account_number_last4: Optional[str] = None
    account_fingerprint: Optional[str] = None
    primary_issue: str = "unknown"
    issue_types: List[str] = field(default_factory=list)
    late_payments: Dict[str, Any] = field(default_factory=dict)
    payment_statuses: Dict[str, Any] = field(default_factory=dict)
    bureau_statuses: Dict[str, Any] = field(default_factory=dict)
    original_creditor: Optional[str] = None
    source_stage: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(
        cls: Type["ProblemAccount"], data: Dict[str, Any]
    ) -> "ProblemAccount":
        known = {
            "name",
            "account_number_last4",
            "account_fingerprint",
            "primary_issue",
            "issue_types",
            "late_payments",
            "payment_statuses",
            "bureau_statuses",
            "original_creditor",
            "source_stage",
        }
        return cls(
            name=data.get("name", ""),
            account_number_last4=data.get("account_number_last4"),
            account_fingerprint=data.get("account_fingerprint"),
            primary_issue=data.get("primary_issue", "unknown"),
            issue_types=list(data.get("issue_types", []) or []),
            late_payments=dict(data.get("late_payments", {}) or {}),
            payment_statuses=dict(data.get("payment_statuses", {}) or {}),
            bureau_statuses=dict(data.get("bureau_statuses", {}) or {}),
            original_creditor=data.get("original_creditor"),
            source_stage=data.get("source_stage", ""),
            extras={k: v for k, v in data.items() if k not in known},
        )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        extras = data.pop("extras", {})
        data.update(extras)
        return data
