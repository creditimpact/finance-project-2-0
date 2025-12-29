import random
import time
from typing import Any, Dict, Tuple

from backend.telemetry.metrics import emit_counter

_MIN_TTL_SEC = 14 * 24 * 60 * 60
_MAX_TTL_SEC = 30 * 24 * 60 * 60

CacheKey = Tuple[str, str, str, str, int]
CacheValue = Tuple[Dict[str, Any], float]

_CACHE: Dict[CacheKey, CacheValue] = {}


def _now() -> float:
    return time.time()


def get_cached_analysis(
    doc_fingerprint: str,
    bureau: str,
    prompt_hash: str,
    model_version: str,
    schema_version: int,
) -> Dict[str, Any] | None:
    key = (doc_fingerprint, bureau, prompt_hash, model_version, schema_version)
    item = _CACHE.get(key)
    if not item:
        emit_counter("analysis.cache_miss")
        return None
    data, expires = item
    if _now() > expires:
        _CACHE.pop(key, None)
        emit_counter("analysis.cache_miss")
        return None
    emit_counter("analysis.cache_hit")
    return data


def store_cached_analysis(
    doc_fingerprint: str,
    bureau: str,
    prompt_hash: str,
    model_version: str,
    result: Dict[str, Any],
    schema_version: int,
) -> None:
    ttl = random.randint(_MIN_TTL_SEC, _MAX_TTL_SEC)
    expires = _now() + ttl
    key = (doc_fingerprint, bureau, prompt_hash, model_version, schema_version)
    _CACHE[key] = (result, expires)
    emit_counter("analysis.cache_store")


def reset_cache() -> None:
    _CACHE.clear()
