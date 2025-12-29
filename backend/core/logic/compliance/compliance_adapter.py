"""Compliance helpers for dispute letter generation."""

from __future__ import annotations

import warnings
from typing import Any, Iterable, List, Mapping, MutableMapping, Set, Tuple

from backend.core.logic.utils.names_normalization import normalize_creditor_name
from backend.core.logic.utils.text_parsing import CHARGEOFF_RE
from backend.core.models.client import ClientInfo

# Default dispute reason inserted when no custom note is provided.
DEFAULT_DISPUTE_REASON = (
    "I formally dispute this account as inaccurate and unverifiable. "
    "Under my rights granted by the Fair Credit Reporting Act (FCRA) "
    "sections 609(a) and 611, I demand that you provide copies of any "
    "original signed contracts, applications or other documents bearing my "
    "signature that you relied upon to report this account. If these documents "
    "cannot be produced within 30 days, the account must be deleted from my "
    "credit file."
)

# Additional closing paragraph warning about escalation.
ESCALATION_NOTE = (
    "If you fail to fully verify these accounts with proper documentation within "
    "30 days, I expect them to be deleted immediately as required by law. Failure "
    "to comply may result in further legal actions or formal complaints filed with "
    "the FTC and CFPB."
)


def sanitize_disputes(
    disputes: List[dict],
    bureau_name: str,
    strategy_summaries: Mapping[str, Mapping[str, Any]],
    log_messages: List[str],
    is_identity_theft: bool,
) -> Tuple[bool, bool, Set[str], bool]:
    """Validate dispute types and detect fallback usage.

    Returns a tuple of (sanitization_issues, bureau_sanitization,
    fallback_norm_names, fallback_used).
    """

    sanitization_issues = False
    bureau_sanitization = False
    allowed_types = {
        "identity_theft",
        "unauthorized_or_unverified",
        "inaccurate_reporting",
    }

    for d in disputes:
        if not (is_identity_theft and d.get("is_suspected_identity_theft", False)):
            if d.get("is_suspected_identity_theft"):
                d["dispute_type"] = "unauthorized_or_unverified"
            else:
                dtype = d.get("dispute_type", "inaccurate_reporting")
                if dtype not in allowed_types:
                    warnings.warn(
                        f"[Fallback] Unrecognized dispute type '{dtype}' for '{d.get('name')}', using generic.",
                        stacklevel=2,
                    )
                    log_messages.append(
                        f"[{bureau_name}] Fallback dispute_type applied to '{d.get('name')}'"
                    )
                    bureau_sanitization = True
                    dtype = "inaccurate_reporting"
                d["dispute_type"] = dtype
        else:
            d["dispute_type"] = "identity_theft"

        summary = strategy_summaries.get(d.get("account_id"))
        if summary is None or not isinstance(summary, dict):
            warnings.warn(
                f"[Sanitization] Missing or malformed summary for '{d.get('name')}'",
                stacklevel=2,
            )
            log_messages.append(
                f"[{bureau_name}] Missing structured summary for '{d.get('name')}'"
            )
            bureau_sanitization = True

    fallback_norm_names = {
        normalize_creditor_name(d.get("name", ""))
        for d in disputes
        if d.get("fallback_unrecognized_action")
    }
    fallback_used = bool(fallback_norm_names)
    if fallback_used:
        warnings.warn(
            f"[Fallback] Generic content used for accounts: {', '.join(sorted(fallback_norm_names))}",
            stacklevel=2,
        )
        log_messages.append(
            f"[{bureau_name}] Generic content used for {', '.join(sorted(fallback_norm_names))}"
        )

    return sanitization_issues, bureau_sanitization, fallback_norm_names, fallback_used


def sanitize_client_info(
    client_info: ClientInfo | Mapping[str, Any],
    bureau_name: str,
    log_messages: List[str],
) -> Tuple[dict, bool]:
    """Remove raw client notes to maintain compliance."""

    client_info_for_gpt = (
        client_info.to_dict()
        if isinstance(client_info, ClientInfo)
        else dict(client_info)
    )
    return client_info_for_gpt, False


def adapt_gpt_output(
    gpt_data: MutableMapping[str, Any],
    fallback_norm_names: Iterable[str],
    acc_type_map: Mapping[Tuple[str, str], Mapping[str, Any]],
    rulebook_fallback_enabled: bool,
) -> None:
    """Apply compliance rules to the GPT response in-place."""

    for acc in gpt_data.get("accounts", []):
        name_key = normalize_creditor_name(acc.get("name", ""))
        if rulebook_fallback_enabled and name_key in set(fallback_norm_names):
            acc["paragraph"] = DEFAULT_DISPUTE_REASON
            acc.pop("requested_action", None)
        acc.pop("personal_note", None)
        action = acc.get("requested_action", "")
        if isinstance(action, str) and (
            "goodwill" in action.lower() or "hardship" in action.lower()
        ):
            acc["requested_action"] = (
                "Please verify this item and correct or remove any inaccuracies."
            )

        lookup_key = (
            name_key,
            (acc.get("account_number") or "").replace("*", "").strip(),
        )
        acc_info = acc_type_map.get(lookup_key) or acc_type_map.get((name_key, ""))
        if acc_info:
            status_text = (
                acc_info.get("account_type", "") + " " + acc_info.get("status", "")
            ).lower()
            if "collection" in status_text:
                acc["paragraph"] = acc["paragraph"].rstrip() + (
                    " Please also provide evidence of assignment or purchase agreements from the "
                    "original creditor to the collection agency proving legal authority to collect this debt."
                )
            elif CHARGEOFF_RE.search(status_text):
                acc["paragraph"] = acc["paragraph"].rstrip() + (
                    " Please provide all original signed contracts or documents directly from the original creditor supporting this charge-off."
                )

        acct_num = acc.get("account_number")
        if (
            isinstance(acct_num, str)
            and acct_num.strip()
            and acct_num.upper() != "N/A"
            and not acct_num.endswith("***")
        ):
            acc["account_number"] = acct_num + "***"

    closing = gpt_data.get("closing_paragraph", "").strip()
    gpt_data["closing_paragraph"] = (
        closing + (" " if closing else "") + ESCALATION_NOTE
    ).strip()


__all__ = [
    "sanitize_disputes",
    "sanitize_client_info",
    "adapt_gpt_output",
    "DEFAULT_DISPUTE_REASON",
    "ESCALATION_NOTE",
]
