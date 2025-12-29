from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field, confloat, constr

PrimaryIssue = Literal[
    "collection",
    "charge_off",
    "bankruptcy",
    "repossession",
    "foreclosure",
    "severe_delinquency",
    "derogatory",
    "high_utilization",
    "none",
    "unknown",
]

Tier = Literal["Tier1", "Tier2", "Tier3", "Tier4", "none"]


class AIAdjudicateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_fingerprint: constr(strip_whitespace=True)
    account_fingerprint: constr(strip_whitespace=True)
    hierarchy_version: constr(strip_whitespace=True) = "v1"
    fields: Dict[str, Any]


class AIAdjudicateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_issue: PrimaryIssue
    tier: Tier
    problem_reasons: List[constr(strip_whitespace=True, max_length=200)] = Field(
        default_factory=list, min_length=0, max_length=10
    )
    confidence: confloat(ge=0.0, le=1.0)
    fields_used: List[
        constr(strip_whitespace=True, pattern=r"^[a-z0-9_]+$", max_length=64)
    ] = Field(default_factory=list, min_length=0, max_length=12)
    decision_source: Literal["ai"] = "ai"


__all__ = [
    "PrimaryIssue",
    "Tier",
    "AIAdjudicateRequest",
    "AIAdjudicateResponse",
]
