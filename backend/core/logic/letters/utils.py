from __future__ import annotations

from typing import Any, Iterable, Mapping

from backend.core.models.account import Account

from .exceptions import StrategyContextMissing


def _get_fields(acc: Account | Mapping[str, Any]) -> tuple[str | None, str | None]:
    if isinstance(acc, Account):
        data = acc.to_dict()
        return data.get("action_tag"), data.get("account_id")
    if isinstance(acc, Mapping):
        return acc.get("action_tag"), acc.get("account_id")
    return getattr(acc, "action_tag", None), getattr(acc, "account_id", None)


def ensure_strategy_context(
    accounts: Iterable[Account | Mapping[str, Any]],
    enforcement_enabled: bool,
) -> None:
    """Ensure each account has an action tag when enforcement is enabled."""

    if not enforcement_enabled:
        return

    for acc in accounts:
        action_tag, account_id = _get_fields(acc)
        if action_tag:
            continue
        raise StrategyContextMissing(account_id)


def populate_required_fields(
    account: dict[str, Any], strat: Mapping[str, Any] | None = None
) -> None:
    """Fill per-action required fields using account and strategy data.

    The router validates that certain fields are present before rendering the
    final letter.  This helper ensures those fields are populated after strategy
    data has been merged into the account context.
    """

    tag = str(account.get("action_tag") or "").lower()
    strat = strat or {}

    # Basic collectors/furnishers -------------------------------------------------
    if tag in {"debt_validation", "pay_for_delete", "cease_and_desist"}:
        account.setdefault("collector_name", account.get("name"))

    if tag == "direct_dispute":
        account.setdefault("furnisher_address", account.get("address"))
        account.setdefault("creditor_name", account.get("name"))

    if tag == "fraud_dispute":
        account.setdefault("creditor_name", account.get("name"))
        account.setdefault("is_identity_theft", True)
        if strat.get("ftc_report_id") is not None and not account.get("ftc_report_id"):
            account["ftc_report_id"] = strat["ftc_report_id"]

    if tag in {"bureau_dispute", "inquiry_dispute", "medical_dispute"}:
        account.setdefault("creditor_name", account.get("name"))
        for field in ["account_number_masked", "bureau", "legal_safe_summary"]:
            if strat.get(field) is not None and not account.get(field):
                account[field] = strat[field]
        if tag == "inquiry_dispute":
            account.setdefault(
                "inquiry_creditor_name",
                account.get("creditor_name") or account.get("name"),
            )
            if account.get("date") and not account.get("inquiry_date"):
                account["inquiry_date"] = account["date"]
        if tag == "medical_dispute":
            if strat.get("amount") is not None and not account.get("amount"):
                account["amount"] = strat["amount"]
            account.setdefault("medical_status", account.get("status"))

    # Strategy‑provided fields -----------------------------------------------------
    if tag == "pay_for_delete" and strat.get("offer_terms") is not None:
        account.setdefault("offer_terms", strat.get("offer_terms"))

    if tag == "goodwill":
        for field in ["months_since_last_late", "account_history_good"]:
            if strat.get(field) is not None and not account.get(field):
                account[field] = strat[field]

    if tag == "mov":
        for field in ["cra_last_result", "days_since_cra_result"]:
            if strat.get(field) is not None and not account.get(field):
                account[field] = strat[field]


__all__ = [
    "ensure_strategy_context",
    "populate_required_fields",
]
