"""Runflow helpers for recording strategy planner progress."""

from __future__ import annotations

from pathlib import Path

from backend.runflow.decider import record_stage


def record_strategy_stage(
    runs_root: Path | str,
    sid: str,
    *,
    status: str,
    plans_written: int,
    planner_errors: int,
    accounts_seen: int,
    accounts_with_openers: int,
) -> None:
    """Persist the strategy stage summary into runflow.json."""

    record_stage(
        sid,
        "strategy",
        status=status,
        counts={
            "plans_written": plans_written,
            "accounts_seen": accounts_seen,
            "accounts_with_openers": accounts_with_openers,
        },
        empty_ok=True,
        metrics={
            "plans_written": plans_written,
            "planner_errors": planner_errors,
            "accounts_seen": accounts_seen,
            "accounts_with_openers": accounts_with_openers,
        },
        runs_root=runs_root,
    )
