from __future__ import annotations

"""Redis client helper with an in-memory fallback."""

import logging
import os
import threading
import time
from typing import Any, Iterable

try:  # pragma: no cover - optional dependency in tests
    import redis as _redis_lib
except Exception:  # pragma: no cover - allow running without redis installed
    _redis_lib = None  # type: ignore[assignment]


log = logging.getLogger(__name__)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    text = str(value).strip().lower()
    return text not in {"", "0", "false", "no", "off"}


class _InMemoryRedis:
    """Very small subset of the Redis API backed by a dictionary."""

    def __init__(self) -> None:
        self._values: dict[str, tuple[Any, float | None]] = {}
        self._lock = threading.Lock()

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [key for key, (_, ttl) in self._values.items() if ttl is not None and ttl <= now]
        for key in expired:
            self._values.pop(key, None)

    def set(self, key: str, value: Any, *, nx: bool | None = None, ex: int | None = None) -> bool:
        with self._lock:
            self._purge_expired()
            if _coerce_bool(nx) and key in self._values:
                return False
            expiry = time.monotonic() + ex if ex else None
            self._values[key] = (value, expiry)
            return True

    def delete(self, *keys: Iterable[str] | str) -> int:
        removed = 0
        with self._lock:
            self._purge_expired()
            if not keys:
                return 0
            # ``delete`` accepts either individual keys or an iterable.
            if len(keys) == 1 and isinstance(keys[0], Iterable) and not isinstance(keys[0], (str, bytes, bytearray)):
                candidates: Iterable[str] = keys[0]
            else:
                candidates = keys  # type: ignore[assignment]
            for key in candidates:
                if isinstance(key, bytes):
                    decoded = key.decode("utf-8", "ignore")
                else:
                    decoded = str(key)
                if decoded in self._values:
                    removed += 1
                    self._values.pop(decoded, None)
        return removed


class _RedisProxy:
    """Proxy exposing ``set``/``delete`` with graceful fallback."""

    def __init__(self) -> None:
        self._client = None
        self._fallback = _InMemoryRedis()
        self._lock = threading.Lock()

    @staticmethod
    def _redis_url() -> str:
        for env in ("REDIS_URL", "CACHE_REDIS_URL", "CELERY_BROKER_URL"):
            value = os.getenv(env)
            if value:
                return value
        return "redis://localhost:6379/0"

    def _ensure_client(self):  # type: ignore[no-untyped-def]
        if _redis_lib is None:
            return None
        with self._lock:
            if self._client is not None:
                return self._client
            try:
                client = _redis_lib.Redis.from_url(self._redis_url())
                # ``ping`` eagerly detects unavailable servers to trigger fallback.
                client.ping()
            except Exception:  # pragma: no cover - fallback path
                log.warning("REDIS_CLIENT_FALLBACK", exc_info=True)
                return None
            self._client = client
            return self._client

    def set(self, key: str, value: Any, *, nx: bool | None = None, ex: int | None = None) -> bool:
        client = self._ensure_client()
        if client is not None:
            try:
                result = client.set(key, value, nx=nx, ex=ex)
                return bool(result)
            except Exception:  # pragma: no cover - fallback path
                log.warning("REDIS_CLIENT_SET_FALLBACK", exc_info=True)
                with self._lock:
                    self._client = None
        return self._fallback.set(key, value, nx=nx, ex=ex)

    def delete(self, *keys: Iterable[str] | str) -> int:
        client = self._ensure_client()
        if client is not None:
            try:
                return int(client.delete(*keys))
            except Exception:  # pragma: no cover - fallback path
                log.warning("REDIS_CLIENT_DELETE_FALLBACK", exc_info=True)
                with self._lock:
                    self._client = None
        return self._fallback.delete(*keys)


redis = _RedisProxy()

__all__ = ["redis"]
