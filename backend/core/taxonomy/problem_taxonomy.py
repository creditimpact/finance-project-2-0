from __future__ import annotations

"""Problem taxonomy helpers.

This module provides a single source of truth for mapping problem
``primary_issue`` values to tiers and utilities for working with them.
"""

from typing import List, Dict, Any, Optional, Literal, cast

PrimaryIssue = Literal[
    "bankruptcy",
    "charge_off",
    "collection",
    "repossession",
    "foreclosure",
    "severe_delinquency",
    "moderate_delinquency",
    "derogatory",
    "high_utilization",
    "none",
    "unknown",
]

Tier = Literal["Tier1", "Tier2", "Tier3", "Tier4", "none"]

# Mapping of canonical issue -> tier
ISSUE_TIER_MAP: Dict[str, Tier] = {
    "bankruptcy": "Tier1",
    "charge_off": "Tier1",
    "collection": "Tier1",
    "repossession": "Tier1",
    "foreclosure": "Tier1",
    "severe_delinquency": "Tier2",
    "moderate_delinquency": "Tier3",
    "derogatory": "Tier3",
    "high_utilization": "Tier4",
    "none": "none",
    "unknown": "none",
}

# Ranking used to compare tiers
_TIER_RANK = {"Tier1": 4, "Tier2": 3, "Tier3": 2, "Tier4": 1, "none": 0}

# Aliases for clamping non canonical issue strings
_ISSUE_ALIASES: Dict[str, str] = {
    "chargeoff": "charge_off",
    "repo": "repossession",
    "moderate_deliquency": "moderate_delinquency",  # common misspelling
}


def clamp_issue(issue: str) -> PrimaryIssue:
    """Coerce a freeform issue string into the canonical enum.

    Any unknown value resolves to ``"unknown"``.
    """

    if not isinstance(issue, str):
        return cast(PrimaryIssue, "unknown")

    key = issue.strip().lower()
    key = _ISSUE_ALIASES.get(key, key)
    if key in ISSUE_TIER_MAP:
        return cast(PrimaryIssue, key)
    return cast(PrimaryIssue, "unknown")


def issue_to_tier(issue: str) -> Tier:
    """Map a primary issue to its tier."""

    canonical = clamp_issue(issue)
    return ISSUE_TIER_MAP.get(canonical, "none")


def compare_tiers(a: Tier, b: Tier) -> Tier:
    """Return the stronger of two tiers."""

    return a if _TIER_RANK[a] >= _TIER_RANK[b] else b


def strongest_tier(issues: List[str]) -> Tier:
    """Compute the strongest tier across a list of issues."""

    current: Tier = "none"
    for issue in issues:
        current = compare_tiers(current, issue_to_tier(issue))
    return current


def _dedupe_reasons(reasons: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for reason in reasons:
        if reason is None:
            continue
        r = str(reason)[:200]
        if r in seen:
            continue
        seen.add(r)
        result.append(r)
        if len(result) >= 10:
            break
    return result


def normalize_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a decision dict into canonical enums and tiers.

    Parameters
    ----------
    decision:
        Dictionary containing keys such as ``primary_issue``, ``tier``,
        ``decision_source``, ``problem_reasons`` and ``confidence``.
    """

    normalized = dict(decision)  # shallow copy to avoid mutating caller input

    primary_issue = clamp_issue(normalized.get("primary_issue", "unknown"))
    normalized["primary_issue"] = primary_issue

    if primary_issue in {"unknown", "none"}:
        tier: Tier = "none"
    else:
        tier = issue_to_tier(primary_issue)
    normalized["tier"] = tier

    # Normalize decision source to 'ai' or 'rules'
    decision_source = normalized.get("decision_source", "ai")
    normalized["decision_source"] = "rules" if decision_source == "rules" else "ai"

    # Dedupe and trim problem reasons
    reasons = normalized.get("problem_reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    normalized["problem_reasons"] = _dedupe_reasons(reasons)

    # Clamp confidence
    confidence = normalized.get("confidence", 0.0)
    try:
        confidence_f = float(confidence)
    except (TypeError, ValueError):
        confidence_f = 0.0
    confidence_f = max(0.0, min(1.0, confidence_f))
    normalized["confidence"] = confidence_f

    return normalized
