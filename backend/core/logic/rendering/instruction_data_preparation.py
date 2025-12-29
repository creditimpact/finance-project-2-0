"""Preparation utilities for instruction generation.

This module isolates the data munging required to build client
instructions. It merges account data across bureaus, performs
basic de-duplication, and generates a human friendly action
sentence for each account. The resulting structure is consumed
by :mod:`instruction_renderer` to build the final HTML and PDF
output.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Tuple

from backend.core.logic.utils.names_normalization import normalize_creditor_name
from backend.core.models.account import Account
from backend.core.services.ai_client import AIClient
from backend.analytics.analytics_tracker import log_ai_request, log_ai_stage


_INPUT_COST_PER_TOKEN = 0.01 / 1000
_OUTPUT_COST_PER_TOKEN = 0.03 / 1000


def extract_clean_name(full_name: str) -> str:
    """Return a deduplicated version of a client's full name."""
    parts = full_name.strip().split()
    seen = set()
    unique_parts = []
    for part in parts:
        if part.lower() not in seen:
            unique_parts.append(part)
            seen.add(part.lower())
    return " ".join(unique_parts)


def generate_account_action(
    account: Account | dict[str, Any], ai_client: AIClient
) -> str:
    """Return a human-readable action sentence for an account using GPT."""
    try:
        acc_dict = account.to_dict() if isinstance(account, Account) else account
        prompt = (
            "You are a friendly credit repair coach speaking in plain English. "
            "Write one short sentence explaining what the client should do next "
            "for the account below. Keep it simple and avoid jargon like 'utilization' or 'negatively impacts.' "
            "If no action is needed, give a quick reassuring note.\n\n"
            f"Account data:\n{json.dumps(acc_dict, indent=2)}\n\n"
            "Respond with only the sentence."
        )
        start = time.perf_counter()
        response = ai_client.chat_completion(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        usage = getattr(response, "usage", None)
        tokens_in = getattr(usage, "prompt_tokens", 0)
        tokens_out = getattr(usage, "completion_tokens", 0)
        cost = tokens_in * _INPUT_COST_PER_TOKEN + tokens_out * _OUTPUT_COST_PER_TOKEN
        log_ai_request(tokens_in, tokens_out, cost, latency_ms)
        log_ai_stage("candidate", tokens_in + tokens_out, cost)
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.replace("```", "").strip()
        return content
    except Exception as e:
        print(f"[âš ï¸] GPT action generation failed: {e}")
        return "Review the attached letter and follow the standard mailing steps."


def prepare_instruction_data(
    client_info,
    bureau_data,
    is_identity_theft: bool,
    run_date: str,
    logo_base64: str,
    ai_client: AIClient,
    strategy: dict | None = None,
):
    """Prepare structured data used for instruction rendering.

    Parameters mirror those of :func:`instructions_generator.generate_instruction_file`.
    The return value is a tuple of (context, all_accounts) where ``context`` is
    consumed by :func:`instruction_renderer.build_instruction_html`.
    """

    client_name = extract_clean_name(client_info.get("name") or "Client")

    all_accounts: List[dict] = []

    def sanitize_number(num: str | None) -> str:
        if not num:
            return ""
        return re.sub(r"\D", "", num)

    strategy_index: Dict[Tuple[str, str], dict] = {}
    if strategy:
        for acc in strategy.get("accounts", []):
            key = (
                normalize_creditor_name(acc.get("name", "")),
                acc.get("account_number_last4")
                or sanitize_number(acc.get("account_number"))[-4:]
                or acc.get("account_fingerprint", ""),
            )
            strategy_index[key] = acc

    def can_merge(existing: dict, new: dict) -> bool:
        """Return True if the two account records likely refer to the same account."""
        name1 = normalize_creditor_name(existing.get("name", "")).lower()
        name2 = normalize_creditor_name(new.get("name", "")).lower()
        if name1 != name2:
            return False

        num1 = (
            existing.get("account_number_last4")
            or sanitize_number(existing.get("account_number"))[-4:]
            or existing.get("account_fingerprint")
        )
        num2 = (
            new.get("account_number_last4")
            or sanitize_number(new.get("account_number"))[-4:]
            or new.get("account_fingerprint")
        )
        if num1 and num2 and num1 != num2:
            return False
        if not num1 and not num2:
            status1 = (existing.get("status") or "").lower()
            status2 = (new.get("status") or "").lower()
            return status1 == status2

        return True

    for bureau, section in bureau_data.items():
        for acc in section.get("all_accounts", []):
            acc_copy = acc.copy()
            acc_copy.setdefault("bureaus", acc.get("bureaus", [bureau]))
            acc_copy.setdefault("categories", set(acc.get("categories", [])))
            strat_key = (
                normalize_creditor_name(acc_copy.get("name", "")),
                acc_copy.get("account_number_last4")
                or sanitize_number(acc_copy.get("account_number"))[-4:]
                or acc_copy.get("account_fingerprint", ""),
            )
            strat = strategy_index.get(strat_key)
            if strat:
                for field in [
                    "action_tag",
                    "priority",
                    "needs_evidence",
                    "legal_notes",
                    "flags",
                    "recommended_action",
                    "advisor_comment",
                    "status",
                    "utilization",
                    "dispute_type",
                    "goodwill_candidate",
                    "letter_type",
                    "custom_letter_note",
                ]:
                    if strat.get(field) is not None and not acc_copy.get(field):
                        acc_copy[field] = strat[field]

            merged = False
            for existing in all_accounts:
                if can_merge(existing, acc_copy):
                    existing["bureaus"].update(acc_copy.get("bureaus", []))
                    existing["categories"].update(acc_copy.get("categories", []))
                    for field in [
                        "action_tag",
                        "priority",
                        "needs_evidence",
                        "legal_notes",
                        "flags",
                        "recommended_action",
                        "advisor_comment",
                        "status",
                        "utilization",
                        "dispute_type",
                        "goodwill_candidate",
                        "letter_type",
                        "custom_letter_note",
                    ]:
                        if not existing.get(field) and acc_copy.get(field):
                            existing[field] = acc_copy[field]
                    if acc_copy.get("duplicate_suspect"):
                        existing["duplicate_suspect"] = True
                    merged = True
                    break
            if not merged:
                acc_copy["bureaus"] = set(acc_copy.get("bureaus", []))
                acc_copy["categories"] = set(acc_copy.get("categories", []))
                all_accounts.append(acc_copy)

    # Additional de-duplication across bureaus using creditor name + bureau key
    deduped: Dict[Tuple[str, str], dict] = {}
    for acc in all_accounts:
        name_key = normalize_creditor_name(acc.get("name", ""))
        for b in acc.get("bureaus", []):
            key = (name_key, b)
            existing = deduped.get(key)
            if existing:
                existing["bureaus"].update(acc.get("bureaus", []))
                existing["categories"].update(acc.get("categories", []))
                for field in [
                    "action_tag",
                    "priority",
                    "needs_evidence",
                    "legal_notes",
                    "flags",
                    "recommended_action",
                    "advisor_comment",
                    "status",
                    "utilization",
                    "dispute_type",
                    "goodwill_candidate",
                    "letter_type",
                    "custom_letter_note",
                ]:
                    if not existing.get(field) and acc.get(field):
                        existing[field] = acc[field]
                if acc.get("duplicate_suspect"):
                    existing["duplicate_suspect"] = True
            else:
                deduped[key] = acc

    all_accounts = list({id(v): v for v in deduped.values()}.values())

    # Remove goodwill-only items entirely if this is an identity theft case.
    if is_identity_theft:
        all_accounts = [
            acc
            for acc in all_accounts
            if str(acc.get("action_tag", "")).lower() != "goodwill"
            and str(acc.get("recommended_action", "")).lower() != "goodwill"
        ]

    has_dupes = any(acc.get("duplicate_suspect") for acc in all_accounts)

    sections: Dict[str, List[dict]] = {
        "problematic": [],
        "improve": [],
        "positive": [],
    }

    for acc in all_accounts:
        name = acc.get("name", "Unknown")
        advisor_comment = acc.get("advisor_comment", "")
        action_tag = acc.get("action_tag", "")
        recommended_action = acc.get("recommended_action") or (
            action_tag.replace("_", " ").title() if action_tag else None
        )
        bureaus = sorted(acc.get("bureaus", []))
        status = acc.get("reported_status") or acc.get("status") or ""
        utilization = acc.get("utilization")
        dispute_type = acc.get("dispute_type", "")
        categories = {c.lower() for c in acc.get("categories", [])}

        def get_group():
            util_pct = None
            if utilization:
                try:
                    util_pct = int(utilization.replace("%", ""))
                except Exception:
                    pass
            status_l = status.lower()
            if (
                "negative_accounts" in categories
                or any(
                    kw in status_l
                    for kw in (
                        "chargeoff",
                        "charge-off",
                        "charge off",
                        "collection",
                        "repossession",
                        "repos",
                        "delinquent",
                        "late payments",
                    )
                )
                or dispute_type
                or acc.get("goodwill_candidate")
            ):
                return "problematic"
            if (
                "open_accounts_with_issues" in categories
                or "high_utilization_accounts" in categories
                or (util_pct is not None and util_pct > 30)
            ):
                return "improve"
            return "positive"

        group = get_group()

        letters = []
        if action_tag.lower() == "dispute":
            letters.append("Dispute")
        if action_tag.lower() == "goodwill" and not is_identity_theft:
            letters.append("Goodwill")
        if acc.get("letter_type") == "custom" or action_tag.lower() == "custom_letter":
            letters.append("Custom")

        action_context = {
            "name": name,
            "bureaus": bureaus,
            "status": status,
            "utilization": utilization,
            "dispute_type": dispute_type,
            "goodwill_candidate": acc.get("goodwill_candidate"),
            "categories": list(categories),
            "action_tag": action_tag,
            "recommended_action": recommended_action,
            "advisor_comment": advisor_comment,
            "priority": acc.get("priority"),
            "needs_evidence": acc.get("needs_evidence"),
            "legal_notes": acc.get("legal_notes"),
            "flags": acc.get("flags"),
        }
        action_sentence = generate_account_action(action_context, ai_client)

        entry = {
            "name": name,
            "bureaus": bureaus,
            "status": status,
            "utilization": utilization,
            "dispute_type": dispute_type,
            "goodwill_candidate": acc.get("goodwill_candidate"),
            "categories": list(categories),
            "action_tag": action_tag,
            "recommended_action": recommended_action,
            "advisor_comment": advisor_comment,
            "priority": acc.get("priority"),
            "needs_evidence": acc.get("needs_evidence"),
            "legal_notes": acc.get("legal_notes"),
            "flags": acc.get("flags"),
            "late_payments": acc.get("late_payments"),
            "letters": letters,
            "action_sentence": action_sentence,
        }

        sections[group].append(entry)

    per_account_actions = [
        {"account_ref": acc["name"], "action_sentence": acc["action_sentence"]}
        for group in sections.values()
        for acc in group
    ]
    advisories = []
    if has_dupes:
        advisories.append("duplicates")
    tips = []
    if strategy:
        tips = strategy.get("global_recommendations", [])

    context = {
        "date": run_date,
        "client_name": client_name,
        "accounts_summary": sections,
        "per_account_actions": per_account_actions,
        "is_identity_theft": is_identity_theft,
        "logo_base64": logo_base64,
        "advisories": advisories,
        "tips": tips,
    }

    return context, all_accounts
