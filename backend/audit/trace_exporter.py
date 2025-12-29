import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from backend.core.models.strategy import StrategyPlan


def export_trace_file(audit: Any, session_id: str) -> Path:
    """Export strategist and fallback diagnostics to trace.json.

    Parameters
    ----------
    audit: AuditLogger or dict
        The audit object or its underlying data structure.
    session_id: str
        Current client session identifier.
    """
    data: Dict[str, Any] = audit.data if hasattr(audit, "data") else audit

    steps = data.get("steps", [])
    accounts = data.get("accounts", {})

    strategist_raw_output = ""
    for step in steps:
        if step.get("stage") == "strategist_raw_output":
            strategist_raw_output = step.get("details", {}).get("content", "")
            break

    strategist_failure_reasons = [
        step.get("details", {}).get("failure_reason")
        for step in steps
        if step.get("stage") == "strategist_failure"
    ]

    strategy_decision_log = []
    fallback_actions = []
    per_account_failures = []
    recommendation_summary = []

    for acc_id, entries in accounts.items():
        for entry in entries:
            stage = entry.get("stage")
            if stage == "strategy_decision":
                decision = {"account_id": acc_id}
                for key in [
                    "action",
                    "recommended_action",
                    "flags",
                    "reason",
                    "classification",
                ]:
                    if entry.get(key) is not None:
                        decision[key] = entry.get(key)
                strategy_decision_log.append(decision)
                recommendation_summary.append(
                    {
                        "account_id": acc_id,
                        "action": entry.get("action"),
                        "recommended_action": entry.get("recommended_action"),
                    }
                )
            elif stage == "strategy_fallback":
                fb = {"account_id": acc_id}
                for key in [
                    "fallback_reason",
                    "strategist_action",
                    "overrode_strategist",
                    "failure_reason",
                    "raw_action",
                ]:
                    if entry.get(key) is not None:
                        fb[key] = entry.get(key)
                fallback_actions.append(fb)
                fail_info = {
                    k: entry.get(k)
                    for k in ("failure_reason", "fallback_reason")
                    if entry.get(k) is not None
                }
                if fail_info:
                    per_account_failures.append({"account_id": acc_id, **fail_info})

    trace: Dict[str, Any] = {
        "strategist_raw_output": strategist_raw_output,
        "strategist_failure_reasons": strategist_failure_reasons,
        "strategy_decision_log": strategy_decision_log,
        "fallback_actions": fallback_actions,
        "per_account_failures": per_account_failures,
        "recommendation_summary": recommendation_summary,
    }

    trace_folder = Path("client_output") / session_id / "trace"
    trace_folder.mkdir(parents=True, exist_ok=True)
    trace_path = trace_folder / "trace.json"
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)
    return trace_path


def export_trace_breakdown(
    audit: Any, strategy: Any, accounts: Any, output_dir: Path | str
) -> None:
    """Export per-account trace breakdown files.

    Parameters
    ----------
    audit: AuditLogger or dict
        Audit logger containing per-account logs.
    strategy: StrategyPlan or mapping
        Raw strategist plan prior to fallback handling.
    accounts: iterable
        Final accounts considered in the run. Only used to determine
        which account IDs to include.
    output_dir: Path or str
        Base output directory (e.g. ``client_output``).
    """

    data: Dict[str, Any] = audit.data if hasattr(audit, "data") else audit
    session_id = data.get("session_id", "session")
    run_date = datetime.now().strftime("%Y-%m-%d")

    trace_folder = Path(output_dir) / session_id / "trace"
    trace_folder.mkdir(parents=True, exist_ok=True)

    # Normalize strategist plan
    plan = StrategyPlan.from_dict(strategy) if isinstance(strategy, dict) else strategy
    raw_accounts: Dict[str, Any] = {}
    if plan is not None:
        for item in getattr(plan, "accounts", []):
            acc_id = str(getattr(item, "account_id", "") or "")
            rec = getattr(item, "recommendation", None)
            raw_accounts[acc_id] = rec.to_dict() if rec else {}

    with (trace_folder / "strategist_raw_output.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_date": run_date,
                "session_id": session_id,
                "accounts": raw_accounts,
            },
            f,
            indent=2,
        )

    # Build decision and fallback maps
    decision_map: Dict[str, Dict[str, Any]] = {}
    fallback_map: Dict[str, Dict[str, Any]] = {}
    summary_map: Dict[str, str] = {}

    reason_text = {
        "keyword_match": "Fallback dispute triggered by keywords",
        "unrecognized_tag": "Strategist action unrecognized",
        "no_recommendation": "No strategist recommendation provided",
    }

    acc_ids: list[str] = []
    try:
        for acc in accounts or []:
            acc_id = str(
                getattr(acc, "account_id", None)
                or getattr(acc, "id", None)
                or acc.get("account_id")
                if isinstance(acc, dict)
                else ""
            )
            if acc_id:
                acc_ids.append(acc_id)
    except Exception:
        pass
    if not acc_ids:
        acc_ids = list(data.get("accounts", {}).keys())

    for acc_id in acc_ids:
        entries = data.get("accounts", {}).get(str(acc_id), [])
        decision_entry = next(
            (e for e in entries if e.get("stage") == "strategy_decision"), None
        )
        fallback_entry = next(
            (e for e in entries if e.get("stage") == "strategy_fallback"), None
        )
        if decision_entry:
            decision_map[str(acc_id)] = {
                "action_tag": decision_entry.get("action"),
                "recommended_action": decision_entry.get("recommended_action"),
                "source": "fallback" if fallback_entry else "strategist",
            }
            summary = decision_entry.get("reason") or ""
            if not summary and fallback_entry:
                summary = reason_text.get(fallback_entry.get("fallback_reason"), "")
            summary_map[str(acc_id)] = summary
        if fallback_entry:
            fb: Dict[str, Any] = {
                "fallback_reason": fallback_entry.get("fallback_reason"),
            }
            if fallback_entry.get("failure_reason"):
                fb["failure_reason"] = fallback_entry.get("failure_reason")
            if fallback_entry.get("strategist_action"):
                fb["strategist_action"] = fallback_entry.get("strategist_action")
            fb["summary"] = reason_text.get(fallback_entry.get("fallback_reason"), "")
            fallback_map[str(acc_id)] = fb

    with (trace_folder / "strategy_decision.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_date": run_date,
                "session_id": session_id,
                "accounts": decision_map,
            },
            f,
            indent=2,
        )

    with (trace_folder / "fallback_reason.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_date": run_date,
                "session_id": session_id,
                "accounts": fallback_map,
            },
            f,
            indent=2,
        )

    with (trace_folder / "recommendation_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "run_date": run_date,
                "session_id": session_id,
                "accounts": summary_map,
            },
            f,
            indent=2,
        )


__all__ = ["export_trace_file", "export_trace_breakdown"]
