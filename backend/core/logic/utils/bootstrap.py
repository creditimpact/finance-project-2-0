"""Bootstrap utilities shared across orchestration layers.

These helpers are intentionally free of side effects so that they can be
imported safely by both the CLI in ``main.py`` and the orchestration logic in
``orchestrators.py``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Mapping


def get_current_month() -> str:
    """Return the current month formatted as ``YYYY-MM``."""
    return datetime.now().strftime("%Y-%m")


def extract_all_accounts(sections: Mapping[str, Any]) -> List[dict]:
    """Return a de-duplicated list of all accounts with source categories.

    Accounts are considered the same only when key fields match. This prevents
    different accounts from the same creditor from being merged together.
    """
    import re
    from datetime import datetime

    from backend.core.logic.utils.names_normalization import normalize_creditor_name

    def sanitize_number(num: str | None) -> str:
        if not num:
            return ""
        digits = "".join(c for c in num if c.isdigit())
        return digits[-4:] if len(digits) >= 4 else digits

    def parse_date(date_str: str | None) -> datetime | None:
        if not date_str:
            return None
        for fmt in ("%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(date_str, fmt)
            except Exception:  # pragma: no cover - safe fallback
                continue
        return None

    accounts: list[dict] = []
    categories = [
        "negative_accounts",
        "open_accounts_with_issues",
        "high_utilization_accounts",
        "positive_accounts",
        "all_accounts",
    ]

    for key in categories:
        for acc in sections.get(key, []):
            acc_copy = acc.copy()
            acc_copy.setdefault("categories", set()).add(key)

            norm_name = normalize_creditor_name(acc_copy.get("name", "")).lower()
            last4 = sanitize_number(acc_copy.get("account_number"))
            bureaus = tuple(sorted(acc_copy.get("bureaus", [])))
            status = (acc_copy.get("status") or "").strip().lower()
            opened = parse_date(acc_copy.get("opened_date"))
            closed = parse_date(acc_copy.get("closed_date"))

            found = None
            for existing in accounts:
                if (
                    normalize_creditor_name(existing.get("name", "")).lower()
                    == norm_name
                    and sanitize_number(existing.get("account_number")) == last4
                    and tuple(sorted(existing.get("bureaus", []))) == bureaus
                    and (existing.get("status") or "").strip().lower() == status
                    and parse_date(existing.get("opened_date")) == opened
                    and parse_date(existing.get("closed_date")) == closed
                ):
                    found = existing
                    break

            if found:
                found.setdefault("categories", set()).add(key)
            else:
                accounts.append(acc_copy)

    from difflib import SequenceMatcher

    def _is_negative(acc: dict) -> bool:
        cats = {c.lower() for c in acc.get("categories", [])}
        status = str(acc.get("status") or acc.get("reported_status") or "").lower()
        if "negative_accounts" in cats:
            return True
        return any(
            kw in status
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

    def _acct_suffix(num: str | None) -> str:
        if not num:
            return ""
        digits = re.sub(r"\D", "", str(num))
        return digits[-4:]

    def _similar_name(a: str, b: str) -> bool:
        n1 = normalize_creditor_name(a or "").lower()
        n2 = normalize_creditor_name(b or "").lower()
        if n1 == n2 or n1.startswith(n2) or n2.startswith(n1):
            return True
        return SequenceMatcher(None, n1, n2).ratio() >= 0.8

    def _parse_amount(val: str | None) -> float | None:
        if not val:
            return None
        clean = re.sub(r"[^0-9.]+", "", str(val))
        try:
            return float(clean)
        except Exception:  # pragma: no cover - safe fallback
            return None

    def _potential_dupe(a: dict, b: dict) -> bool:
        if not _similar_name(a.get("name"), b.get("name")):
            return False

        s1 = _acct_suffix(a.get("account_number"))
        s2 = _acct_suffix(b.get("account_number"))
        if s1 and s2 and s1 != s2:
            return False

        bal1 = _parse_amount(a.get("balance"))
        bal2 = _parse_amount(b.get("balance"))
        if bal1 is not None and bal2 is not None:
            diff = abs(bal1 - bal2)
            if diff > max(100, 0.1 * min(bal1, bal2)):
                return False

        d1 = parse_date(a.get("opened_date"))
        d2 = parse_date(b.get("opened_date"))
        if d1 and d2 and abs((d1 - d2).days) > 90:
            return False
        d1c = parse_date(a.get("closed_date"))
        d2c = parse_date(b.get("closed_date"))
        if d1c and d2c and abs((d1c - d2c).days) > 90:
            return False

        return True

    dupe_indices: set[int] = set()
    for i, acc_a in enumerate(accounts):
        if not _is_negative(acc_a):
            continue
        for j in range(i + 1, len(accounts)):
            acc_b = accounts[j]
            if not _is_negative(acc_b):
                continue
            if _potential_dupe(acc_a, acc_b):
                dupe_indices.update({i, j})

    for idx in dupe_indices:
        accounts[idx]["duplicate_suspect"] = True

    return accounts
