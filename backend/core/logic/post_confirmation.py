from __future__ import annotations

"""Adapter utilities executed after user confirmation (Stage C).

This module transforms the user-confirmed account selections into a structured
``BureauPayload`` that downstream components such as the letter generator and
tri-merge outcome ingestion can consume.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

from backend.api.config import env_bool
from backend.core.models.bureau import BureauPayload
from backend.core.logic.report_analysis.report_postprocessing import (
    _assign_issue_types,
    enrich_account_metadata,
)
from backend.core.orchestrators import _annotate_with_tri_merge


@dataclass
class _Sections:
    """Normalized view of account sections after user selection."""

    disputes: list[dict]
    goodwill: list[dict]
    inquiries: list[dict]
    high_utilization: list[dict]

    @classmethod
    def from_selection(cls, data: Mapping[str, Any]) -> "_Sections":
        return cls(
            disputes=list(
                data.get("disputes")
                or data.get("negative_accounts")
                or []
            ),
            goodwill=list(
                data.get("goodwill")
                or data.get("open_accounts_with_issues")
                or []
            ),
            inquiries=list(
                data.get("inquiries")
                or data.get("unauthorized_inquiries")
                or []
            ),
            high_utilization=list(
                data.get("high_utilization")
                or data.get("high_utilization_accounts")
                or []
            ),
        )


def _extract_bureaus(bureaus: Iterable[Any]) -> list[str]:
    """Return bureau names from a heterogeneous ``bureaus`` value."""

    names: list[str] = []
    for b in bureaus:
        if isinstance(b, str):
            names.append(b)
        elif isinstance(b, Mapping):
            name = b.get("bureau") or b.get("name")
            if isinstance(name, str):
                names.append(name)
    return names


def build_dispute_payload(
    selected_accounts: Mapping[str, Any],
    explanations: Mapping[str, Any] | None = None,
) -> Dict[str, BureauPayload]:
    """Assemble per-bureau payload from user selected accounts.

    The input ``selected_accounts`` is expected to contain the user's confirmed
    problematic accounts.  For each account we ensure ``issue_types`` are
    populated, enrich metadata, merge any user-provided explanations, and then
    optionally annotate with tri-merge mismatches.  Finally, accounts are grouped
    into a ``BureauPayload`` for each bureau present.
    """

    explanations = explanations or {}
    sections = _Sections.from_selection(selected_accounts)

    all_accounts = sections.disputes + sections.goodwill + sections.high_utilization
    for acc in all_accounts:
        if not acc.get("issue_types"):
            _assign_issue_types(acc)
        enrich_account_metadata(acc)
        acc_id = str(acc.get("account_id") or "")
        if acc_id and acc_id in explanations and not acc.get("structured_summary"):
            acc["structured_summary"] = explanations[acc_id]

    tri_sections = {
        "negative_accounts": sections.disputes,
        "open_accounts_with_issues": sections.goodwill,
        "high_utilization_accounts": sections.high_utilization,
    }
    if env_bool("ENABLE_TRI_MERGE", False):
        _annotate_with_tri_merge(tri_sections)

    bureau_map: Dict[str, BureauPayload] = {}

    def _ensure(bureau: str) -> BureauPayload:
        return bureau_map.setdefault(bureau, BureauPayload())

    for acc in sections.disputes:
        for bureau in _extract_bureaus(acc.get("bureaus") or []):
            _ensure(bureau).disputes.append(acc)

    for acc in sections.goodwill:
        for bureau in _extract_bureaus(acc.get("bureaus") or []):
            _ensure(bureau).goodwill.append(acc)

    for inq in sections.inquiries:
        bureau = inq.get("bureau") or inq.get("source")
        if bureau:
            _ensure(str(bureau)).inquiries.append(inq)

    for acc in sections.high_utilization:
        for bureau in _extract_bureaus(acc.get("bureaus") or []):
            _ensure(bureau).high_utilization.append(acc)

    return bureau_map


__all__ = ["build_dispute_payload"]

