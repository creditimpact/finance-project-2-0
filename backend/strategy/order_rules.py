"""Business rules for ordering validation findings into dispute strategies."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from .types import Finding

_OPENER_DECISION = {"strong_actionable", "dispute_primary"}
_SUPPORTIVE_DECISION = {"supportive_needs_companion", "support_paired"}
_CATEGORY_PRIORITY = {"status": 0, "activity": 1, "terms": 2, "history": 3}
_RARE_DOC_TOKENS = {"audit_log", "collection_history", "cra_report_7y"}


def _is_natural_text(finding: Finding) -> bool:
    category = (finding.category or "").strip().lower()
    if category == "natural_text":
        return True
    marker = getattr(finding, "is_natural_text", None)
    if isinstance(marker, bool):
        return marker
    return False


def _category_rank(category: str) -> int:
    return _CATEGORY_PRIORITY.get(category.lower(), len(_CATEGORY_PRIORITY))


def _normalized_decision(value: str | None) -> str:
    return (value or "").strip().lower()


def _reason_bonus(reason_code: str | None) -> int:
    code = (reason_code or "").upper()
    if code == "C5_ALL_DIFF":
        return 2
    if code == "C4_TWO_MATCH_ONE_DIFF":
        return 1
    return 0


def _doc_rarity_bonus(documents: Iterable[str] | None) -> int:
    if not documents:
        return 0
    for doc in documents:
        lowered = doc.lower()
        for token in _RARE_DOC_TOKENS:
            if token in lowered:
                return 1
    return 0


def _bureaus_present(finding: Finding) -> int:
    if finding.present_count is not None:
        return max(int(finding.present_count or 0), 0)
    if finding.bureaus:
        return len([bureau for bureau in finding.bureaus if bureau])
    return 0


def _score_components(finding: Finding) -> tuple[int, int, int]:
    reason_bonus = _reason_bonus(finding.reason_code)
    doc_bonus = _doc_rarity_bonus(finding.documents)
    base_score = max(int(finding.min_days or 0), 0)
    total = base_score + reason_bonus + doc_bonus
    return total, reason_bonus, doc_bonus


def _sort_key(finding: Finding) -> tuple[int, int, int, int, int, str]:
    score, reason_bonus, doc_bonus = _score_components(finding)
    present = _bureaus_present(finding)
    # Higher score first, then higher min_days, more bureaus, category priority, then field name
    return (
        -score,
        -max(int(finding.min_days or 0), 0),
        -present,
        _category_rank(finding.category),
        -(reason_bonus + doc_bonus),
        finding.field,
    )


def rank_findings(
    findings: list[Finding], *, include_supporters: bool = True, exclude_natural_text: bool = True
) -> tuple[list[Finding], list[Finding], list[Finding], list[Dict[str, str]]]:
    """Rank findings into opener, middle, and closer buckets."""

    openers: list[Finding] = []
    middle: list[Finding] = []
    skipped: list[Dict[str, str]] = []

    for finding in findings:
        if exclude_natural_text and _is_natural_text(finding):
            skipped.append({"field": finding.field, "reason": "excluded_category"})
            continue

        decision = _normalized_decision(getattr(finding, "default_decision", None))
        if decision in _OPENER_DECISION:
            openers.append(finding)
        elif decision in _SUPPORTIVE_DECISION and include_supporters:
            middle.append(finding)
        else:
            reason = "unsupported_decision"
            if decision in _SUPPORTIVE_DECISION and not include_supporters:
                reason = "supporters_disabled"
            if getattr(finding, "min_days", None) is None:
                reason = "no_sla_or_min_days"
            skipped.append({"field": finding.field, "reason": reason})
            continue

    openers.sort(key=_sort_key)
    middle.sort(key=_sort_key)
    closers = list(reversed(openers))
    return openers, middle, closers, skipped


def build_strategy_orders(
    findings: Sequence[Finding], *, include_supporters: bool = True, exclude_natural_text: bool = True
) -> tuple[List[Finding], Dict[str, Dict[str, object]], List[Dict[str, str]]]:
    """Return canonical order and per-field role/score metadata."""

    openers, middle, closers, skipped = rank_findings(
        list(findings),
        include_supporters=include_supporters,
        exclude_natural_text=exclude_natural_text,
    )
    sequence: List[Finding] = []
    seen: set[str] = set()

    def _append(finding: Finding) -> None:
        if finding.field in seen:
            return
        seen.add(finding.field)
        sequence.append(finding)

    for finding in openers:
        _append(finding)
    for finding in middle:
        _append(finding)
    for finding in closers:
        _append(finding)

    meta: Dict[str, Dict[str, object]] = {}
    for role, collection in (
        ("opener", openers),
        ("supporter", middle),
        ("closer", closers),
    ):
        for finding in collection:
            if finding.field in meta:
                continue
            total, reason_bonus, doc_bonus = _score_components(finding)
            base_score = max(int(finding.min_days or 0), 0)
            bonuses: Dict[str, int] = {}
            if reason_bonus:
                bonuses["reason_bonus"] = reason_bonus
            if doc_bonus:
                bonuses["doc_bonus"] = doc_bonus
            normalized_decision = _normalized_decision(finding.default_decision)
            meta[finding.field] = {
                "role": role,
                "min_days": base_score,
                "score": {
                    "base": base_score,
                    "bonuses": bonuses,
                    "total": base_score + sum(bonuses.values()),
                },
                "category": finding.category,
                "default_decision": finding.default_decision or "",
                "reason_code": finding.reason_code,
                "documents": list(finding.documents or []),
                "bureaus_present": _bureaus_present(finding),
                "why_here": _why_here(role, normalized_decision, finding),
            }

    return sequence, meta, skipped


def _why_here(role: str, decision: str, finding: Finding) -> str:
    if role == "opener":
        descriptor = decision or "unspecified"
        return f"Top-scoring opener ({descriptor})"
    if role == "supporter":
        return "Supporter chained under opener; maintains cadence"
    return "Closer reinforces earlier disputes"
