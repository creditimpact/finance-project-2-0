import hashlib
import json
import random
import time
from typing import Any, Dict

from backend.telemetry.metrics import emit_counter

_MIN_TTL_SEC = 14 * 24 * 60 * 60
_MAX_TTL_SEC = 30 * 24 * 60 * 60

_CACHE: Dict[str, tuple[Dict[str, Any], float]] = {}


def _now() -> float:
    return time.time()


def _hash_payload(
    stage2: Dict[str, Any],
    stage2_5: Dict[str, Any],
    stage3: Dict[str, Any],
    model_version: str,
    prompt_version: int,
    schema_version: int,
) -> str:
    payload = json.dumps(
        [stage2, stage2_5, stage3, model_version, prompt_version, schema_version],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached_strategy(
    stage2: Dict[str, Any],
    stage2_5: Dict[str, Any],
    stage3: Dict[str, Any],
    model_version: str,
    *,
    prompt_version: int,
    schema_version: int,
) -> Dict[str, Any] | None:
    key = _hash_payload(
        stage2, stage2_5, stage3, model_version, prompt_version, schema_version
    )
    item = _CACHE.get(key)
    if not item:
        emit_counter("strategy.cache_miss")
        return None
    data, expires = item
    if _now() > expires:
        _CACHE.pop(key, None)
        emit_counter("strategy.cache_miss")
        return None
    emit_counter("strategy.cache_hit")
    return data


def store_cached_strategy(
    stage2: Dict[str, Any],
    stage2_5: Dict[str, Any],
    stage3: Dict[str, Any],
    model_version: str,
    result: Dict[str, Any],
    *,
    prompt_version: int,
    schema_version: int,
) -> None:
    ttl = random.randint(_MIN_TTL_SEC, _MAX_TTL_SEC)
    expires = _now() + ttl
    key = _hash_payload(
        stage2, stage2_5, stage3, model_version, prompt_version, schema_version
    )
    _CACHE[key] = (result, expires)
    emit_counter("strategy.cache_store")


def reset_cache() -> None:
    _CACHE.clear()
