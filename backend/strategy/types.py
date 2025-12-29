"""Dataclasses shared across the strategy planner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Finding:
    """Normalized validation finding used by the planner."""

    field: str
    category: str
    min_days: int
    duration_unit: str
    default_decision: Optional[str] = None
    reason_code: Optional[str] = None
    documents: Optional[List[str]] = None
    bureaus: Optional[List[str]] = None
    present_count: Optional[int] = None
    is_missing: bool = False
    is_mismatch: bool = False
    bureau_dispute_state: Optional[Dict[str, str]] = None
    missing_count: Optional[int] = None
    ai_decision: Optional[str] = None
    ai_rationale: Optional[str] = None
    ai_citations: Optional[List[str]] = None
    ai_legacy_decision: Optional[str] = None
