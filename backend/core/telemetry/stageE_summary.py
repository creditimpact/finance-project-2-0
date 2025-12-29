from __future__ import annotations

import math
from typing import Any, Mapping, List, Dict, Set

from backend.core.case_store import api as cs_api
from backend.core.case_store.telemetry import emit

_sessions: Dict[str, Dict[str, Any]] = {}


def record_stageA_event(event: str, fields: Mapping[str, Any]) -> None:
    """Record per-account Stage A telemetry for later aggregation.

    Parameters
    ----------
    event:
        Name of the telemetry event. Only ``stageA_eval`` and ``stageA_fallback``
        are recognized.
    fields:
        Telemetry payload emitted for the event.
    """
    session_id = fields.get("session_id")
    if not session_id:
        return
    info = _sessions.setdefault(session_id, {"fallbacks": set(), "latencies": []})
    if event == "stageA_fallback":
        acc_id = fields.get("account_id")
        if acc_id is not None:
            info["fallbacks"].add(str(acc_id))
        latency = fields.get("latency_ms")
        if isinstance(latency, (int, float)):
            info["latencies"].append(float(latency))
    elif event == "stageA_eval":
        latency = fields.get("ai_latency_ms")
        if isinstance(latency, (int, float)):
            info["latencies"].append(float(latency))


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = int(math.ceil(0.95 * len(values))) - 1
    return float(values[idx])


def emit_stageE_summary(session_id: str, problem_accounts: List[Mapping[str, Any]], duration_ms: float | None = None) -> None:
    """Emit aggregated Stage E session summary telemetry."""
    total_accounts = len(cs_api.list_accounts(session_id))
    problem_count = len(problem_accounts or [])
    ai_adoption = sum(1 for acc in problem_accounts if acc.get("decision_source") == "ai")
    sess = _sessions.pop(session_id, {"fallbacks": set(), "latencies": []})
    fallback_count = len(sess.get("fallbacks", set()))
    latencies = sess.get("latencies", [])
    latency_p95 = _p95(latencies)
    emit(
        "stageE_summary",
        session_id=session_id,
        total_accounts=total_accounts,
        problem_accounts=problem_count,
        ai_adoption_pct=ai_adoption / max(1, problem_count),
        fallback_pct=fallback_count / max(1, problem_count),
        ai_latency_p95_ms=latency_p95,
        duration_ms=duration_ms if duration_ms is not None else 0.0,
    )
