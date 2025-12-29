"""Goodwill letter preparation utilities.

This module contains helper functions that select accounts eligible for
goodwill adjustment letters and shape account data for AI prompting.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping

from backend.audit.audit import AuditLogger
from backend.core.logic.compliance.rules_loader import get_neutral_phrase
from backend.core.logic.strategy.summary_classifier import ClassificationRecord
from backend.core.logic.utils.names_normalization import normalize_creditor_name
from backend.core.logic.utils.text_parsing import has_late_indicator
from backend.core.models.client import ClientInfo


def select_goodwill_candidates(
    client_info: ClientInfo | Mapping[str, Any],
    bureau_data: Mapping[str, Any],
    *,
    strategy: Mapping[str, Any] | None = None,
) -> Mapping[str, List[Dict[str, Any]]]:
    """Return a mapping of creditor name to accounts needing goodwill letters.

    The selection logic mirrors the historical behaviour from the monolithic
    ``generate_goodwill_letters`` module. Accounts that are in dispute, in a
    negative status, or lack late payment history are excluded.
    """

    goodwill_accounts: Dict[str, List[Dict[str, Any]]] = {}

    def clean_num(num: str | None) -> str:
        return re.sub(r"\D", "", num or "")

    strategy_index: Dict[tuple[str, str], Dict[str, Any]] = {}
    if strategy:
        for acc in strategy.get("accounts", []):
            key = (
                normalize_creditor_name(acc.get("name", "")),
                clean_num(acc.get("account_number"))[-4:],
            )
            strategy_index[key] = acc

    dispute_map: Dict[str, set[str]] = {}
    for content in bureau_data.values():
        for acc in content.get("disputes", []):
            action = str(
                acc.get("action_tag") or acc.get("recommended_action") or ""
            ).lower()
            if action != "dispute":
                continue
            # Earlier versions attempted to read a mis-encoded "name" key from
            # imported data. Those stray byte sequences caused parsing issues on
            # some platforms. We only rely on the standard ASCII key now.
            name = acc.get("name")
            if not name:
                continue
            name_norm = normalize_creditor_name(name)
            dispute_map.setdefault(name_norm, set()).add(
                clean_num(acc.get("account_number"))
            )

    def consider_account(account: Dict[str, Any]) -> None:
        strat_key = (
            normalize_creditor_name(account.get("name", "")),
            clean_num(account.get("account_number"))[-4:],
        )
        strat = strategy_index.get(strat_key)
        if strat:
            for field in [
                "action_tag",
                "priority",
                "needs_evidence",
                "legal_notes",
                "flags",
            ]:
                if strat.get(field) is not None and not account.get(field):
                    account[field] = strat[field]
            if strat.get("action_tag") and strat["action_tag"].lower() != "goodwill":
                return

        status_text = str(
            account.get("status")
            or account.get("account_status")
            or account.get("payment_status")
            or ""
        ).lower()
        if any(
            kw in status_text
            for kw in (
                "collection",
                "chargeoff",
                "charge-off",
                "charge off",
                "repossession",
                "repos",
                "delinquent",
                "late payments",
            )
        ):
            return

        # Avoid non-ASCII fallbacks that previously appeared in some exports.
        # The canonical "name" key is sufficient for our tests.
        name = account.get("name")
        if not name:
            return
        name_norm = normalize_creditor_name(name)

        acct_num = clean_num(
            account.get("account_number") or account.get("acct_number")
        )
        dispute_nums = dispute_map.get(name_norm)
        if dispute_nums is not None:
            for dn in dispute_nums:
                if not dn or not acct_num or dn == acct_num:
                    # Account already in dispute for this creditor.
                    return

        late_info = account.get("late_payments")

        def _total_lates(info) -> int:
            total = 0
            if isinstance(info, dict):
                for bureau_vals in info.values():
                    if isinstance(bureau_vals, dict):
                        for v in bureau_vals.values():
                            try:
                                total += int(v)
                            except (TypeError, ValueError):
                                continue
            return total

        if _total_lates(late_info) == 0 and not has_late_indicator(account):
            return

        creditor = account.get("name")
        goodwill_accounts.setdefault(creditor, []).append(account)

    for content in bureau_data.values():
        candidate_sections = [
            content.get("goodwill", []),
            content.get("disputes", []),
            content.get("high_utilization", []),
        ]
        for section in candidate_sections:
            for account in section:
                consider_account(account)

    for section in [
        "all_accounts",
        "positive_accounts",
        "open_accounts_with_issues",
        "negative_accounts",
        "high_utilization_accounts",
    ]:
        for account in client_info.get(section, []):
            consider_account(account)

    return goodwill_accounts


def prepare_account_summaries(
    accounts: List[Dict[str, Any]],
    structured_summaries: Dict[str, Dict[str, Any]] | None,
    classification_map: Mapping[str, ClassificationRecord] | None,
    state: str | None,
    session_id: str | None,
    *,
    audit: AuditLogger | None = None,
) -> List[Dict[str, Any]]:
    """Merge duplicate account records and enrich with strategy metadata."""

    merged_accounts: List[Dict[str, Any]] = []
    seen_numbers: Dict[str, Dict[str, Any]] = {}

    for acc in accounts:
        acc_num = str(acc.get("account_number") or "").strip()
        name_norm = normalize_creditor_name(acc.get("name", ""))
        target = None
        if acc_num and acc_num in seen_numbers:
            target = seen_numbers[acc_num]
        else:
            for existing in merged_accounts:
                if normalize_creditor_name(existing.get("name", "")) == name_norm:
                    if not acc_num or not existing.get("account_number"):
                        target = existing
                        break
        if target is None:
            target = acc.copy()
            merged_accounts.append(target)
            if acc_num:
                seen_numbers[acc_num] = target
        else:
            for k, v in acc.items():
                if v and not target.get(k):
                    target[k] = v
            if acc_num and not target.get("account_number"):
                target["account_number"] = acc_num
                seen_numbers[acc_num] = target

    def summarize_late(late):
        if not isinstance(late, dict):
            return None
        parts = []
        for b, vals in late.items():
            for k, v in vals.items():
                if v:
                    parts.append(f"{v}x{k}-day late ({b})")
        return ", ".join(parts) if parts else None

    account_summaries: List[Dict[str, Any]] = []
    seen_numbers_set: set[str] = set()

    for acc in merged_accounts:
        account_number = acc.get("account_number") or acc.get("acct_number")
        status = (
            acc.get("reported_status")
            or acc.get("status")
            or acc.get("account_status")
            or acc.get("payment_status")
        )
        account_number_str = str(account_number or "").strip()
        if account_number_str in seen_numbers_set:
            continue
        seen_numbers_set.add(account_number_str)

        summary: Dict[str, Any] = {
            "name": acc.get("name", "Unknown"),
            "account_number": account_number_str or "Unavailable",
            "status": status or "N/A",
            "hardship_reason": acc.get("hardship_reason"),
            "recovery_summary": acc.get("recovery_summary"),
            "personal_note": acc.get("personal_note"),
            "repayment_status": acc.get("account_status") or acc.get("payment_status"),
        }

        if structured_summaries:
            struct = structured_summaries.get(acc.get("account_id"), {})
            summary["structured_summary"] = struct
            record = None
            if classification_map:
                record = classification_map.get(str(acc.get("account_id")))
            cls = record.classification if record else {}
            summary.update(
                {
                    "dispute_reason": cls.get("category"),
                    "legal_hook": cls.get("legal_tag"),
                    "tone": cls.get("tone"),
                    "dispute_approach": cls.get("dispute_approach"),
                }
            )
            if cls.get("state_hook"):
                summary["state_hook"] = cls["state_hook"]
            neutral, neutral_reason = get_neutral_phrase(cls.get("category"), struct)
            if neutral:
                summary["neutral_phrase"] = neutral
            if audit:
                audit.log_account(
                    acc.get("account_id") or acc.get("name"),
                    {
                        "stage": "goodwill_letter",
                        "classification": cls,
                        "neutral_phrase": neutral,
                        "neutral_phrase_reason": neutral_reason,
                        "structured_summary": struct,
                    },
                )

        late_summary = summarize_late(acc.get("late_payments"))
        if late_summary:
            summary["late_history"] = late_summary
        if acc.get("advisor_comment"):
            summary["advisor_comment"] = acc.get("advisor_comment")
        if acc.get("action_tag"):
            summary["action_tag"] = acc.get("action_tag")
        if acc.get("recommended_action"):
            summary["recommended_action"] = acc.get("recommended_action")
        if acc.get("flags"):
            summary["flags"] = acc.get("flags")
        if acc.get("priority"):
            summary["priority"] = acc.get("priority")
        if acc.get("needs_evidence"):
            summary["needs_evidence"] = acc.get("needs_evidence")
        if acc.get("legal_notes"):
            summary["legal_notes"] = acc.get("legal_notes")

        account_summaries.append(summary)

    return account_summaries


__all__ = ["select_goodwill_candidates", "prepare_account_summaries"]
