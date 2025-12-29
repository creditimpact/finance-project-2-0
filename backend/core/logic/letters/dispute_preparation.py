"""Utilities for preparing disputes and inquiries before letter generation."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from backend.core.logic.strategy.fallback_manager import determine_fallback_action
from backend.core.logic.utils.names_normalization import normalize_creditor_name
from backend.core.models.bureau import BureauPayload
from backend.core.models.client import ClientInfo
from backend.core.letters.router import select_template


def dedupe_disputes(
    disputes: List[dict], bureau_name: str, log: List[str]
) -> List[dict]:
    """Remove duplicate dispute entries based on creditor name and account number."""

    def _sanitize(num: str | None) -> str | None:
        if not num:
            return None
        digits = "".join(c for c in str(num) if c.isdigit())
        if not digits:
            return None
        return digits[-4:] if len(digits) >= 4 else digits

    seen: set[Tuple[str, str | None]] = set()
    deduped: List[dict] = []
    for d in disputes:
        name_key = normalize_creditor_name(d.get("name", "")).lower()
        num = d.get("account_number_last4") or _sanitize(d.get("account_number"))
        if num is None:
            num = d.get("account_fingerprint")
        key = (name_key, num)
        if key in seen:
            log.append(f"[{bureau_name}] Skipping duplicate account '{d.get('name')}'")
            continue
        seen.add(key)
        deduped.append(d)
    return deduped


def prepare_disputes_and_inquiries(
    bureau_name: str,
    payload: BureauPayload | Mapping[str, Any],
    client_info: ClientInfo | Mapping[str, Any],
    account_inquiry_matches: List[dict],
    log_messages: List[str],
) -> Tuple[List[dict], List[dict], Dict[Tuple[str, str], dict]]:
    """Filter disputes, deduplicate accounts, and match inquiries.

    Returns the filtered disputes, inquiries to be disputed, and a mapping of
    account identifiers to account metadata.
    """

    decision = select_template("dispute", {"bureau": bureau_name}, phase="candidate")
    log_messages.append(
        f"[{bureau_name}] Router selected template '{decision.template_path}'"
    )
    disputes: List[dict] = []
    for d in payload.get("disputes", []):
        action = str(d.get("action_tag") or d.get("recommended_action") or "").lower()
        if action != "dispute":
            fallback_action = determine_fallback_action(d)
            if fallback_action == "dispute":
                d["action_tag"] = "dispute"
                d.setdefault("recommended_action", "Dispute")
                log_messages.append(
                    f"[{bureau_name}] Fallback dispute tag applied to '{d.get('name')}'"
                )
                action = "dispute"
        if action == "dispute":
            disputes.append(d)
        else:
            log_messages.append(
                f"[{bureau_name}] Skipping account '{d.get('name')}' - recommended_action='{action}'"
            )

    disputes = dedupe_disputes(disputes, bureau_name, log_messages)

    acc_type_map: Dict[Tuple[str, str], dict] = {}
    for d in disputes:
        key = (
            normalize_creditor_name(d.get("name", "")),
            d.get("account_number_last4")
            or (d.get("account_number") or "").replace("*", "").strip()
            or d.get("account_fingerprint", ""),
        )
        acc_type_map[key] = {
            "account_type": str(d.get("account_type") or ""),
            "status": str(d.get("status") or d.get("account_status") or ""),
        }

    inquiries = payload.get("inquiries", [])
    print(f"[INFO] {len(inquiries)} inquiries for {bureau_name} to evaluate:")
    for raw_inq in inquiries:
        print(
            f"    -> {raw_inq.get('creditor_name')} - {raw_inq.get('date')} ({raw_inq.get('bureau', bureau_name)})"
        )

    matched_set = {
        normalize_creditor_name(m.get("creditor_name", ""))
        for m in account_inquiry_matches
    }
    open_account_names = set()
    open_account_map: Dict[str, str] = {}
    for a in payload.get("all_accounts", []):
        status_text = str(a.get("account_status") or a.get("status") or "").lower()
        if "closed" not in status_text:
            norm_name = normalize_creditor_name(a.get("name", ""))
            open_account_names.add(norm_name)
            open_account_map[norm_name] = a.get("name")

    for section in [
        "all_accounts",
        "open_accounts_with_issues",
        "positive_accounts",
        "negative_accounts",
    ]:
        for a in client_info.get(section, []):
            status_text = str(a.get("account_status") or a.get("status") or "").lower()
            if "closed" not in status_text:
                norm_name = normalize_creditor_name(a.get("name", ""))
                open_account_names.add(norm_name)
                open_account_map.setdefault(norm_name, a.get("name"))

    filtered_inquiries: List[dict] = []
    for inq in inquiries:
        name_norm = normalize_creditor_name(inq.get("creditor_name", ""))
        matched = name_norm in matched_set or name_norm in open_account_names
        matched_label = open_account_map.get(name_norm)
        if name_norm in matched_set and not matched_label:
            matched_label = "matched list"
        print(
            "Inquiry being evaluated: {name} on {bureau} {date} - {status}".format(
                name=inq.get("creditor_name"),
                bureau=inq.get("bureau", bureau_name),
                date=inq.get("date"),
                status="matched to " + matched_label if matched_label else "no match",
            )
        )
        if not matched:
            filtered_inquiries.append(inq)
            print(
                f"[Will be disputed] Inquiry detected: {inq.get('creditor_name')}, {inq.get('date')}, {bureau_name}"
            )
            print(
                f"... Inquiry added to dispute letter: {inq.get('creditor_name')} - {inq.get('date')} ({bureau_name})"
            )
        else:
            print(
                f"Inquiry skipped due to open account match: {matched_label or inq.get('creditor_name')}"
            )
            log_messages.append(
                f"[{bureau_name}] Skipping inquiry '{inq.get('creditor_name')}' matched to existing account"
            )

    return disputes, filtered_inquiries, acc_type_map


__all__ = ["dedupe_disputes", "prepare_disputes_and_inquiries"]
