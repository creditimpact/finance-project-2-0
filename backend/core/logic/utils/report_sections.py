"""Utilities for filtering and summarizing report sections by bureau."""

from __future__ import annotations

from .names_normalization import BUREAUS, normalize_bureau_name, normalize_creditor_name
from .text_parsing import has_late_indicator


def filter_sections_by_bureau(sections, bureau_name, log_list=None):
    """Return relevant subsets only for the specified bureau.

    ``log_list`` if provided will be appended with human readable
    explanations when items are skipped or categorised.
    """
    bureau_name = normalize_bureau_name(bureau_name)

    filtered = {"disputes": [], "goodwill": [], "inquiries": [], "high_utilization": []}

    for acc in sections.get("negative_accounts", []):
        reported = [normalize_bureau_name(b) for b in acc.get("bureaus", [])]
        if bureau_name in reported:
            filtered["disputes"].append(acc)
        elif log_list is not None:
            log_list.append(
                f"[{bureau_name}] Skipped negative account '{acc.get('name')}' - not reported to this bureau"
            )

    for acc in sections.get("open_accounts_with_issues", []):
        reported = [normalize_bureau_name(b) for b in acc.get("bureaus", [])]
        if bureau_name in reported:
            if acc.get("goodwill_candidate", False):
                filtered["goodwill"].append(acc)
            else:
                filtered["disputes"].append(acc)
        elif log_list is not None:
            log_list.append(
                f"[{bureau_name}] Skipped account '{acc.get('name')}' - not reported to this bureau"
            )

    for acc in sections.get("high_utilization_accounts", []):
        reported = [normalize_bureau_name(b) for b in acc.get("bureaus", [])]
        if bureau_name in reported:
            filtered["high_utilization"].append(acc)
        elif log_list is not None:
            log_list.append(
                f"[{bureau_name}] Skipped high utilization account '{acc.get('name')}' - not reported to this bureau"
            )

    for inquiry in sections.get("inquiries", []):
        inquiry_bureau = normalize_bureau_name(inquiry.get("bureau"))
        if inquiry_bureau == bureau_name:
            filtered["inquiries"].append(inquiry)
        elif log_list is not None:
            if inquiry.get("bureau"):
                log_list.append(
                    f"[{bureau_name}] Skipped inquiry '{inquiry.get('creditor_name')}' - belongs to {inquiry.get('bureau')}"
                )

    # ðŸ" Detect late payment indicators in positive or uncategorized accounts
    seen = {
        (
            normalize_creditor_name(acc.get("name", "")),
            acc.get("account_number"),
            bureau_name,
        )
        for section in filtered.values()
        for acc in section
        if isinstance(acc, dict)
    }

    extra_sources = sections.get("positive_accounts", []) + sections.get(
        "all_accounts", []
    )
    for acc in extra_sources:
        reported = [normalize_bureau_name(b) for b in acc.get("bureaus", [])]
        if bureau_name not in reported:
            continue
        key = (
            normalize_creditor_name(acc.get("name", "")),
            acc.get("account_number"),
            bureau_name,
        )
        if key in seen:
            continue
        if has_late_indicator(acc):
            enriched = acc.copy()
            text = " ".join(
                str(acc.get(field, ""))
                for field in ["status", "remarks", "advisor_comment", "flags"]
            )
            if (
                "good standing" in text.lower()
                or "closed" in str(acc.get("account_status", "")).lower()
            ):
                enriched["goodwill_candidate"] = True
                filtered["goodwill"].append(enriched)
            else:
                filtered["disputes"].append(enriched)
            seen.add(key)

    return filtered


def extract_summary_from_sections(sections):
    """Returns analytical summary from full data structure."""
    summary = {
        "total_negative": len(sections.get("negative_accounts", [])),
        "total_late_payments": len(
            [
                acc
                for acc in sections.get("open_accounts_with_issues", [])
                if has_late_indicator(acc)
            ]
        ),
        "high_utilization_accounts": len(sections.get("high_utilization_accounts", [])),
        "recent_inquiries": len(sections.get("inquiries", [])),
        "identity_theft_suspicions": len(
            [
                acc
                for acc in sections.get("negative_accounts", [])
                if acc.get("is_suspected_identity_theft")
            ]
        ),
        "by_bureau": {
            bureau: {
                "disputes": len(
                    [
                        acc
                        for acc in sections.get("negative_accounts", [])
                        if bureau in acc.get("bureaus", [])
                    ]
                ),
                "goodwill": len(
                    [
                        acc
                        for acc in sections.get("open_accounts_with_issues", [])
                        if acc.get("goodwill_candidate")
                        and bureau in acc.get("bureaus", [])
                    ]
                ),
                "high_utilization": len(
                    [
                        acc
                        for acc in sections.get("high_utilization_accounts", [])
                        if bureau in acc.get("bureaus", [])
                    ]
                ),
            }
            for bureau in BUREAUS
        },
    }
    return summary
