"""Centralized fallback logic for account action tagging."""

from typing import Any, Mapping

# Keywords that imply an account should be disputed when no explicit action is provided.
_STATUS_KEYWORDS = (
    "collection",
    "chargeoff",
    "charge-off",
    "charge off",
    "repossession",
    "repos",
    "delinquent",
    "late payments",
)


def _get(account: Mapping[str, Any] | Any, key: str) -> Any:
    """Safely retrieve a value from ``account`` regardless of type."""
    getter = getattr(account, "get", None)
    if callable(getter):
        return getter(key)
    return account[key] if isinstance(account, Mapping) and key in account else None


def determine_fallback_action(account: Mapping[str, Any] | Any) -> str:
    """Return a fallback ``action_tag`` for ``account``.

    Currently returns ``"dispute"`` when the account's status contains
    derogatory keywords or a ``dispute_type`` is present. Otherwise
    returns an empty string.
    """
    status_text = str(
        _get(account, "status") or _get(account, "account_status") or ""
    ).lower()
    if any(k in status_text for k in _STATUS_KEYWORDS) or _get(account, "dispute_type"):
        return "dispute"
    return ""


__all__ = ["determine_fallback_action"]
