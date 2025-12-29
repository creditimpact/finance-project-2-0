import atexit
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Mapping, Optional

from backend.core.logic.utils.pii import redact_pii


def _env_bool(name: str, default: bool) -> bool:
    """Best effort boolean environment reader."""

    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _get_config_module():
    """Lazily import ``backend.api.config`` to avoid circular imports."""

    try:
        from backend.api import config as config_module  # type: ignore import-not-found
    except Exception:  # pragma: no cover - best effort fallback
        return None
    return config_module


def _is_observability_enabled() -> bool:
    """Return the observability flag with environment fallback."""

    config_module = _get_config_module()
    if config_module is not None:
        return bool(getattr(config_module, "ENABLE_OBSERVABILITY_H", True))
    return _env_bool("ENABLE_OBSERVABILITY_H", True)


def _get_ai_base_url() -> str:
    """Return the configured AI base URL with a safe default."""

    config_module = _get_config_module()
    if config_module is not None:
        try:
            ai_config = config_module.get_ai_config()
        except Exception:  # pragma: no cover - configuration may be incomplete
            ai_config = None
        if ai_config is not None:
            base_url = getattr(ai_config, "base_url", None)
            if base_url:
                return str(base_url)
    return os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

# Cache metrics --------------------------------------------------------------

_CACHE_METRICS: Dict[str, int] = {"hits": 0, "misses": 0, "evictions": 0}
_OPS = 0
_SNAPSHOT_INTERVAL = 100

# AI usage metrics -----------------------------------------------------------

_AI_METRICS: Dict[str, float] = {
    "tokens_in": 0,
    "tokens_out": 0,
    "cost": 0.0,
    "latency_ms": 0.0,
}

# Generic counters -----------------------------------------------------------

# Metrics are stored as floats to support both counters and timers.
_COUNTERS: Dict[str, float] = {}
_COUNTER_LOCK = Lock()

# Canary rollout decisions ---------------------------------------------------

_CANARY_DECISIONS: List[Dict[str, str]] = []


def log_canary_decision(decision: str, template: str | None = None) -> None:
    """Record a canary routing decision for analytics snapshots."""
    if not _is_observability_enabled():
        return
    entry = {"timestamp": datetime.now().isoformat(), "decision": decision}
    if template:
        entry["template"] = template
    _CANARY_DECISIONS.append(entry)


def get_canary_decisions() -> List[Dict[str, str]]:
    """Return recorded canary decisions (for tests)."""

    return list(_CANARY_DECISIONS)


def reset_canary_decisions() -> None:
    """Clear stored canary decisions (for tests)."""

    _CANARY_DECISIONS.clear()


def emit_counter(name: str, increment: float | Mapping[str, Any] = 1) -> None:
    """Increment a named metric for analytics.

    ``increment`` may be a float value or a mapping of metadata which
    generates dimensioned counters of the form ``"name.key.value"``.
    """

    if not _is_observability_enabled():
        return
    with _COUNTER_LOCK:
        if isinstance(increment, Mapping):
            _COUNTERS[name] = _COUNTERS.get(name, 0) + 1
            allowed = {"bureau", "mismatch_type", "cycle", "step"}
            for key, value in increment.items():
                if value is None or key not in allowed:
                    continue
                attr_name = f"{name}.{key}.{value}"
                _COUNTERS[attr_name] = _COUNTERS.get(attr_name, 0) + 1
        else:
            _COUNTERS[name] = _COUNTERS.get(name, 0) + increment


def set_metric(name: str, value: float) -> None:
    """Set a named metric to an explicit value."""
    if not _is_observability_enabled():
        return
    with _COUNTER_LOCK:
        _COUNTERS[name] = value


def get_counters() -> Dict[str, float]:
    """Return current generic metrics (for tests)."""

    with _COUNTER_LOCK:
        return _COUNTERS.copy()


def reset_counters() -> None:
    """Reset generic metrics (for tests)."""

    with _COUNTER_LOCK:
        _COUNTERS.clear()


def get_missing_fields_heatmap() -> Dict[str, Dict[str, int]]:
    """Return aggregated missing-field counts grouped by tag and field."""

    heatmap: Dict[str, Dict[str, int]] = {}
    prefix = "router.missing_fields."
    for key, count in _COUNTERS.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        if rest.startswith("finalize."):
            # Finalize metrics are tracked separately and do not include the template
            continue
        try:
            tag, remainder = rest.split(".", 1)
            _template, field = remainder.rsplit(".", 1)
        except ValueError:
            continue
        bucket = heatmap.setdefault(tag, {})
        bucket[field] = bucket.get(field, 0) + int(count)
    return heatmap


def get_router_skipped_counts() -> Dict[str, int]:
    """Return counts of skipped router tags."""

    prefix = "router.skipped."
    skipped: Dict[str, int] = {}
    for key, count in _COUNTERS.items():
        if not key.startswith(prefix):
            continue
        tag = key[len(prefix) :]
        skipped[tag] = skipped.get(tag, 0) + int(count)
    return skipped


def _write_cache_snapshot() -> None:
    """Persist current cache metrics to ``analytics_data`` and reset counters."""
    if not _is_observability_enabled():
        return
    global _OPS
    analytics_dir = Path("analytics_data")
    analytics_dir.mkdir(exist_ok=True)

    now = datetime.now()
    filename = analytics_dir / f"cache_{now.strftime('%Y-%m-%d_%H-%M-%S')}.json"

    payload = {"timestamp": now.isoformat(), "cache": _CACHE_METRICS.copy()}
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    for k in _CACHE_METRICS:
        _CACHE_METRICS[k] = 0
    _OPS = 0


def _maybe_flush() -> None:
    if _OPS >= _SNAPSHOT_INTERVAL:
        _write_cache_snapshot()


def _log_cache_event(key: str) -> None:
    global _OPS
    _CACHE_METRICS[key] += 1
    _OPS += 1
    _maybe_flush()


def log_cache_hit() -> None:
    """Record a classification cache hit."""
    if not _is_observability_enabled():
        return
    _log_cache_event("hits")
    emit_counter("cache_hit")


def log_cache_miss() -> None:
    """Record a classification cache miss."""
    if not _is_observability_enabled():
        return
    _log_cache_event("misses")


def log_cache_eviction() -> None:
    """Record a classification cache eviction."""
    if not _is_observability_enabled():
        return
    _log_cache_event("evictions")


def get_cache_stats() -> Dict[str, int]:
    """Return current cache metrics (for tests)."""

    return _CACHE_METRICS.copy()


def reset_cache_counters() -> None:
    """Reset cache metrics (for tests)."""

    global _OPS
    for k in _CACHE_METRICS:
        _CACHE_METRICS[k] = 0
    _OPS = 0


def log_policy_violations_prevented(count: int) -> None:
    """Record number of policy violations caught by guardrails."""

    emit_counter("policy_violations_prevented_count", count)


def log_letter_without_strategy() -> None:
    """Record that a letter was attempted without strategy context."""
    if not _is_observability_enabled():
        return
    emit_counter("letters_without_strategy_context")


def log_policy_override_reason(reason: str) -> None:
    """Record a policy override along with the associated reason."""
    if not _is_observability_enabled():
        return
    sanitized = str(reason).replace(" ", "_")
    emit_counter(f"policy_override_reason.{sanitized}")


def log_guardrail_fix(letter_type: str | None = None) -> None:
    """Record that guardrails triggered a follow-up fix.

    ``letter_type`` allows tracking fixes by document category while also
    keeping a global total for backward compatibility.
    """

    if not _is_observability_enabled():
        return
    emit_counter("guardrail_fix_count")
    if letter_type:
        emit_counter(f"guardrail_fix_count.{letter_type}")


# AI usage helpers -----------------------------------------------------------


def log_ai_request(
    tokens_in: int, tokens_out: int, cost: float, latency_ms: float
) -> None:
    """Record tokens, estimated cost, and latency for an AI call."""
    if not _is_observability_enabled():
        return
    _AI_METRICS["tokens_in"] += tokens_in
    _AI_METRICS["tokens_out"] += tokens_out
    _AI_METRICS["cost"] += cost
    _AI_METRICS["latency_ms"] += latency_ms


def log_ai_stage(stage: str, tokens: int, cost: float) -> None:
    """Record token and cost usage for a specific pipeline stage."""
    if not _is_observability_enabled():
        return
    emit_counter(f"ai.tokens.{stage}", tokens)
    emit_counter(f"ai.cost.{stage}", cost)


def get_ai_stats() -> Dict[str, float]:
    """Return current AI usage metrics (for tests)."""

    return _AI_METRICS.copy()


def reset_ai_stats() -> None:
    """Reset AI usage metrics (for tests)."""

    _AI_METRICS.update(tokens_in=0, tokens_out=0, cost=0.0, latency_ms=0.0)


# Canary guardrails ----------------------------------------------------------


def check_canary_guardrails(
    render_ms_ceiling: float,
    sanitizer_rate_limit: float,
    ai_daily_cap: float,
) -> bool:
    """Check canary SLOs and halt rollout if breached.

    Returns True if the canary was halted.
    """

    counters = get_counters()
    ai_stats = get_ai_stats()
    breached = False

    templates = set()
    for key in counters:
        if (
            key.startswith("router.finalized.")
            or key.startswith("validation.failed.")
            or key.startswith("sanitizer.applied.")
            or key.startswith("letter.render_ms.")
        ):
            templates.add(key.split(".")[-1])

    for tmpl in templates:
        total = counters.get(f"router.finalized.{tmpl}", 0)
        fails = counters.get(f"validation.failed.{tmpl}", 0)
        renders = counters.get(f"letter.render_ms.{tmpl}", 0.0)
        sanit = counters.get(f"sanitizer.applied.{tmpl}", 0)
        if total:
            if fails / total >= 0.01:
                breached = True
            if sanit / total > sanitizer_rate_limit:
                breached = True
        if renders and renders > render_ms_ceiling:
            breached = True

    if ai_stats.get("cost", 0.0) > ai_daily_cap:
        breached = True

    if breached:
        emit_counter("canary.halt")
        log_canary_decision("halt")
        os.environ["ROUTER_CANARY_PERCENT"] = "0"
    return breached


def _flush_on_exit() -> None:
    if _OPS:
        _write_cache_snapshot()


atexit.register(_flush_on_exit)


def save_analytics_snapshot(
    client_info: dict,
    report_summary: dict,
    strategist_failures: Optional[Dict[str, int]] = None,
) -> None:
    if not _is_observability_enabled():
        return
    logging.getLogger(__name__).info(
        "Analytics tracker using OPENAI_BASE_URL=%s", _get_ai_base_url()
    )
    analytics_dir = Path("analytics_data")
    analytics_dir.mkdir(exist_ok=True)

    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M")

    filename = analytics_dir / f"{timestamp}.json"

    snapshot = {
        "date": now.strftime("%Y-%m-%d"),
        "goal": client_info.get("goal", "N/A"),
        "dispute_type": (
            "identity_theft" if client_info.get("is_identity_theft") else "standard"
        ),
        "client_name": client_info.get("name", "Unknown"),
        "client_state": client_info.get("state", "unknown"),
        "summary": {
            "num_collections": report_summary.get("num_collections", 0),
            "num_late_payments": report_summary.get("num_late_payments", 0),
            "high_utilization": report_summary.get("high_utilization", False),
            "recent_inquiries": report_summary.get("recent_inquiries", 0),
            "total_inquiries": report_summary.get("total_inquiries", 0),
            "num_negative_accounts": report_summary.get("num_negative_accounts", 0),
            "num_accounts_over_90_util": report_summary.get(
                "num_accounts_over_90_util", 0
            ),
            "account_types_in_problem": report_summary.get(
                "account_types_in_problem", []
            ),
        },
        "strategic_recommendations": report_summary.get(
            "strategic_recommendations", []
        ),
    }

    if strategist_failures:
        snapshot["strategist_failures"] = strategist_failures

    snapshot["metrics"] = {
        "counters": get_counters(),
        "ai": get_ai_stats(),
        "router_skipped": get_router_skipped_counts(),
    }

    snapshot["canary"] = get_canary_decisions()

    with open(filename, "w", encoding="utf-8") as f:
        f.write(redact_pii(json.dumps(snapshot, indent=2)))

    print(f"[📊] Analytics snapshot saved: {filename}")
