import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from backend.analytics.analytics_tracker import (
    log_cache_eviction,
    log_cache_hit,
    log_cache_miss,
)
from backend.core.logic.compliance.rules_loader import recompute_rules_version
from backend.core.logic.utils.json_utils import parse_json
from backend.core.services.ai_client import AIClient

logger = logging.getLogger(__name__)

try:  # pragma: no cover - fallback for tests without app config
    from backend.api.config import (
        CLASSIFY_CACHE_ENABLED,
        CLASSIFY_CACHE_MAXSIZE,
        CLASSIFY_CACHE_TTL_SEC,
    )
except Exception:  # pragma: no cover

    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.lower() not in {"0", "false", "no"}

    CLASSIFY_CACHE_ENABLED = _env_bool("CLASSIFY_CACHE_ENABLED", True)
    CLASSIFY_CACHE_MAXSIZE = int(os.getenv("CLASSIFY_CACHE_MAXSIZE", "5000"))
    CLASSIFY_CACHE_TTL_SEC = int(os.getenv("CLASSIFY_CACHE_TTL_SEC", "0"))


RULES_VERSION = recompute_rules_version()

_CACHE: (
    "OrderedDict[Tuple[str, str, str, str, str], Tuple[Mapping[str, str], float]]"
) = OrderedDict()
_CACHE_HITS = 0
_CACHE_MISSES = 0
_CACHE_EVICTIONS = 0


def _prune_expired() -> None:
    if CLASSIFY_CACHE_TTL_SEC <= 0:
        return
    global _CACHE_EVICTIONS
    now = time.time()
    keys = [k for k, (_, ts) in _CACHE.items() if now - ts > CLASSIFY_CACHE_TTL_SEC]
    for k in keys:
        _CACHE.pop(k, None)
        _CACHE_EVICTIONS += 1
        log_cache_eviction()


@dataclass
class ClassificationRecord:
    """Cached classification details for a structured summary."""

    summary: Mapping[str, Any]
    classification: Mapping[str, str]
    summary_hash: str
    state: str | None = None
    rules_version: str | None = None


def summary_hash(summary: Mapping[str, Any]) -> str:
    """Return a stable hash for ``summary``.

    The hash is used to detect when a structured summary has changed so that
    expensive AI classification calls can be skipped when possible.
    """
    data = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def _cache_get(
    session_id: str,
    account_id: str,
    summary: Mapping[str, Any],
    state: str | None,
    rules_version: str,
) -> Mapping[str, str] | None:
    global _CACHE_HITS, _CACHE_MISSES
    if not CLASSIFY_CACHE_ENABLED:
        _CACHE_MISSES += 1
        log_cache_miss()
        return None
    key = (session_id, account_id, summary_hash(summary), state or "", rules_version)
    item = _CACHE.get(key)
    if not item:
        _CACHE_MISSES += 1
        log_cache_miss()
        return None
    value, ts = item
    if CLASSIFY_CACHE_TTL_SEC > 0 and time.time() - ts > CLASSIFY_CACHE_TTL_SEC:
        _CACHE.pop(key, None)
        _CACHE_MISSES += 1
        log_cache_miss()
        global _CACHE_EVICTIONS
        _CACHE_EVICTIONS += 1
        log_cache_eviction()
        return None
    _CACHE.move_to_end(key)
    _CACHE_HITS += 1
    log_cache_hit()
    return value


def _cache_set(
    session_id: str,
    account_id: str,
    summary: Mapping[str, Any],
    state: str | None,
    rules_version: str,
    value: Mapping[str, str],
) -> None:
    if not CLASSIFY_CACHE_ENABLED:
        return
    key = (session_id, account_id, summary_hash(summary), state or "", rules_version)
    _prune_expired()
    if key in _CACHE:
        _CACHE.move_to_end(key)
    _CACHE[key] = (value, time.time())
    global _CACHE_EVICTIONS
    while len(_CACHE) > CLASSIFY_CACHE_MAXSIZE:
        _CACHE.popitem(last=False)
        _CACHE_EVICTIONS += 1
        log_cache_eviction()


def invalidate_summary_cache(session_id: str, account_id: str | None = None) -> None:
    keys = [
        k
        for k in list(_CACHE)
        if k[0] == session_id and (account_id is None or k[1] == account_id)
    ]
    for k in keys:
        _CACHE.pop(k, None)


def cache_hits() -> int:
    return _CACHE_HITS


def cache_misses() -> int:
    return _CACHE_MISSES


def cache_evictions() -> int:
    return _CACHE_EVICTIONS


def reset_cache() -> None:
    """Clear cache and reset counters (for tests)."""
    global _CACHE_HITS, _CACHE_MISSES, _CACHE_EVICTIONS
    _CACHE.clear()
    _CACHE_HITS = _CACHE_MISSES = _CACHE_EVICTIONS = 0


_RULE_MAP = {
    "identity_theft": {
        "legal_tag": "FCRA §605B",
        "dispute_approach": "fraud_block",
        "tone": "urgent",
    },
    "not_mine": {
        "legal_tag": "FCRA §609(e)",
        "dispute_approach": "validation",
        "tone": "firm",
    },
    "goodwill": {
        "legal_tag": "FCRA §623(a)(1)",
        "dispute_approach": "goodwill_adjustment",
        "tone": "conciliatory",
    },
    "inaccurate_reporting": {
        "legal_tag": "FCRA §611",
        "dispute_approach": "reinvestigation",
        "tone": "professional",
    },
}

_STATE_HOOKS = {
    "CA": "California Consumer Credit Reporting Agencies Act",
    "NY": "New York FCRA Article 25",
}


def _heuristic_category(summary: Mapping[str, Any]) -> tuple[str, bool]:
    text_bits = [
        summary.get("dispute_type", ""),
        summary.get("facts_summary", ""),
    ] + summary.get("claimed_errors", [])
    text = " ".join(
        [t.lower().replace("_", " ") for t in text_bits if isinstance(t, str)]
    ).strip()

    if not text:
        return "inaccurate_reporting", True

    dispute_type = str(summary.get("dispute_type", "")).lower()
    if dispute_type in _RULE_MAP:
        return dispute_type, True

    if "identity" in text or "stolen" in text:
        return "identity_theft", True
    if "not mine" in text:
        return "not_mine", True
    if "goodwill" in text:
        return "goodwill", True
    return "inaccurate_reporting", False


def classify_client_summary(
    summary: Mapping[str, Any],
    ai_client: AIClient,
    state: str | None = None,
    *,
    session_id: str | None = None,
    account_id: str | None = None,
) -> Mapping[str, str]:
    """Classify a structured summary into a dispute category and legal strategy.

    When ``session_id`` and ``account_id`` are provided the result is cached
    using those identifiers and a hash of ``summary``.
    """

    if session_id and account_id:
        cached = _cache_get(session_id, account_id, summary, state, RULES_VERSION)
        if cached:
            return cached

    category, confident = _heuristic_category(summary)
    if not confident:
        prompt = (
            "Classify the following structured credit dispute summary into one of "
            "the categories: not_mine, inaccurate_reporting, identity_theft, goodwill. "
            "Return only JSON with a 'category' field. Summary: "
            f"{summary}"
        )
        try:
            resp = ai_client.response_json(
                prompt=prompt,
                response_format={"type": "json_object"},
            )
            content = resp.output[0].content[0].text
            data, _ = parse_json(content)
            data = data or {}
            category = data.get("category") or category
        except Exception:
            pass

    mapping = _RULE_MAP.get(category, _RULE_MAP["inaccurate_reporting"]).copy()
    result = {"category": category, **mapping}
    if state and state in _STATE_HOOKS:
        result["state_hook"] = _STATE_HOOKS[state]
    logger.info("Summary classification: %s -> %s", summary.get("account_id"), result)
    if session_id and account_id:
        _cache_set(session_id, account_id, summary, state, RULES_VERSION, result)
    return result


def classify_client_summaries(
    summaries: list[Mapping[str, Any]],
    ai_client: AIClient,
    state: str | None = None,
    *,
    session_id: str | None = None,
) -> Mapping[str, Mapping[str, str]]:
    """Classify multiple summaries in a single request.

    ``summaries`` should contain an ``account_id`` field for each entry.  The
    return value maps each ``account_id`` to its classification result.  Any
    summaries missing from the batch response fall back to individual
    classification via :func:`classify_client_summary`.
    """

    results: dict[str, Mapping[str, str]] = {}
    to_classify: list[Mapping[str, Any]] = []
    for summary in summaries:
        acc_id = str(summary.get("account_id", ""))
        if session_id and acc_id:
            cached = _cache_get(session_id, acc_id, summary, state, RULES_VERSION)
            if cached:
                results[acc_id] = cached
                continue
        to_classify.append(summary)

    if to_classify:
        prompt = (
            "Classify the following structured credit dispute summaries into the "
            "categories: not_mine, inaccurate_reporting, identity_theft, goodwill. "
            "Return a JSON object keyed by account_id with a 'category' field for "
            "each entry. Summaries: "
            f"{to_classify}"
        )
        data: Mapping[str, Any] | None = None
        try:
            resp = ai_client.response_json(
                prompt=prompt, response_format={"type": "json_object"}
            )
            content = resp.output[0].content[0].text
            data, _ = parse_json(content)
        except Exception:
            data = None

        for summary in to_classify:
            acc_id = str(summary.get("account_id", ""))
            item = data.get(acc_id) if isinstance(data, Mapping) else None
            category = item.get("category") if isinstance(item, Mapping) else None
            if not category:
                # Fall back to single-item classification which also handles
                # caching and heuristics.
                results[acc_id] = classify_client_summary(
                    summary,
                    ai_client,
                    state,
                    session_id=session_id,
                    account_id=acc_id,
                )
                continue

            mapping = _RULE_MAP.get(category, _RULE_MAP["inaccurate_reporting"]).copy()
            result = {"category": category, **mapping}
            if state and state in _STATE_HOOKS:
                result["state_hook"] = _STATE_HOOKS[state]
            results[acc_id] = result
            if session_id and acc_id:
                _cache_set(session_id, acc_id, summary, state, RULES_VERSION, result)

    return results


__all__ = [
    "classify_client_summary",
    "classify_client_summaries",
    "invalidate_summary_cache",
    "summary_hash",
    "ClassificationRecord",
    "cache_hits",
    "cache_misses",
    "cache_evictions",
    "reset_cache",
    "RULES_VERSION",
]
