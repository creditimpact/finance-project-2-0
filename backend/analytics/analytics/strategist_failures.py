from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable


def tally_failure_reasons(audit: Any) -> Dict[str, int]:
    """Return counts of strategist failure reasons from an audit object.

    The ``audit`` argument may be an :class:`AuditLogger` instance or a raw
    ``dict`` representing the audit data. Only entries that contain a
    ``failure_reason`` key are tallied.
    """
    if audit is None:
        return {}

    data = getattr(audit, "data", audit) or {}
    accounts: Dict[str, Iterable[Dict[str, Any]]] = data.get("accounts", {})

    counter: Counter[str] = Counter()
    seen: set[tuple[str, str]] = set()
    for account_id, entries in accounts.items():
        for entry in entries:
            reason = entry.get("failure_reason")
            if reason and (account_id, reason) not in seen:
                counter[reason] += 1
                seen.add((account_id, reason))

    return dict(counter)


def tally_fallback_vs_decision(audit: Any) -> Dict[str, int]:
    """Count accounts that used strategy fallback versus direct decisions.

    Returns a dictionary with two keys:

    - ``strategy_fallback`` - number of accounts that include a
      ``strategy_fallback`` entry.
    - ``strategy_decision_only`` - number of accounts that have a
      ``strategy_decision`` entry but no ``strategy_fallback`` entry.

    The ``audit`` argument may be an :class:`AuditLogger` instance or a raw
    ``dict`` representing the audit data.
    """

    if audit is None:
        return {}

    data = getattr(audit, "data", audit) or {}
    accounts: Dict[str, Iterable[Dict[str, Any]]] = data.get("accounts", {})

    counts = {"strategy_fallback": 0, "strategy_decision_only": 0}

    for entries in accounts.values():
        stages = {e.get("stage") for e in entries}
        if "strategy_fallback" in stages:
            counts["strategy_fallback"] += 1
        elif "strategy_decision" in stages:
            counts["strategy_decision_only"] += 1

    return counts


__all__ = ["tally_failure_reasons", "tally_fallback_vs_decision"]
