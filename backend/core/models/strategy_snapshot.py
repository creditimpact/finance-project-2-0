from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, List


@dataclass
class StrategySnapshot:
    """Per-account snapshot of rulebook evaluation results."""

    legal_safe_summary: str = ""
    suggested_dispute_frame: str = ""
    rulebook_version: str = ""
    precedence_version: str = ""
    rule_hits: List[str] = field(default_factory=list)
    needs_evidence: List[str] = field(default_factory=list)
    red_flags: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategySnapshot":
        return cls(
            legal_safe_summary=data.get("legal_safe_summary", ""),
            suggested_dispute_frame=data.get("suggested_dispute_frame", ""),
            rulebook_version=data.get("rulebook_version", ""),
            precedence_version=data.get("precedence_version", ""),
            rule_hits=list(data.get("rule_hits", []) or []),
            needs_evidence=list(data.get("needs_evidence", []) or []),
            red_flags=list(data.get("red_flags", []) or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
