from __future__ import annotations

"""Tactical actions executed after planning."""

from typing import Iterable, List
from contextlib import ExitStack

from backend.core.locks import account_lock


def generate_letters(session: dict, allowed_tags: Iterable[str]):
    """Generate letters for accounts whose tags are allowed by the planner."""

    from backend.core.orchestrators import generate_letters as _core_generate_letters
    import planner

    strategy = session.get("strategy") or {}
    accounts = strategy.get("accounts", [])
    allowed = set(allowed_tags)
    if allowed:
        accounts = [acc for acc in accounts if acc.get("action_tag") in allowed]
        strategy = dict(strategy)
        strategy["accounts"] = accounts

    account_ids: List[str] = [str(acc.get("account_id")) for acc in accounts if acc.get("account_id")]

    with ExitStack() as stack:
        for acc_id in account_ids:
            stack.enter_context(account_lock(acc_id))
        result = _core_generate_letters(
            session.get("client_info"),
            session.get("bureau_data"),
            session.get("sections"),
            session.get("today_folder"),
            session.get("is_identity_theft", False),
            strategy,
            session.get("audit"),
            session.get("log_messages"),
            session.get("classification_map"),
            session.get("ai_client"),
            session.get("app_config"),
        )
        if account_ids:
            planner.record_send(session, account_ids)
    return result
