"""LLM service integration used for AI adjudication."""

from __future__ import annotations

import json
import time
from typing import Any

from backend.config import (
    AI_MODEL_ID,
    AI_REQUEST_TIMEOUT_S,
    AI_MAX_TOKENS,
    AI_TEMPERATURE_DEFAULT,
)
from backend.core.case_store.telemetry import emit


def get_ai_client():
    """Return the configured AI client.

    The indirection keeps ``run_llm_prompt`` easily patchable for tests while
    deferring the actual client import until runtime.
    """

    from services.ai_client import get_ai_client as _get  # local import

    return _get()


def run_llm_prompt(
    system_text: str,
    user_text: Any,
    *,
    temperature: float = AI_TEMPERATURE_DEFAULT,
    timeout_s: int = AI_REQUEST_TIMEOUT_S,
    model: str = AI_MODEL_ID,
    max_tokens: int = AI_MAX_TOKENS,
    **legacy_kwargs: Any,
) -> str:
    """Call the underlying AI client and return the raw string response.

    Parameters largely mirror the OpenAI chat completion API. ``timeout_s`` can
    also be provided via the legacy ``timeout`` keyword for backwards
    compatibility with existing call sites.

    The function emits minimal telemetry and propagates any exception from the
    client so the caller can handle fallbacks.
    """

    # Support older call sites using ``timeout`` instead of ``timeout_s``
    if "timeout" in legacy_kwargs:
        timeout_s = legacy_kwargs["timeout"]

    client = get_ai_client()

    if not isinstance(user_text, str):
        try:
            user_payload = json.dumps(user_text, separators=(",", ":"))
        except Exception:
            user_payload = str(user_text)
    else:
        user_payload = user_text

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_payload},
    ]

    t0 = time.perf_counter()
    try:
        resp = client.chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout_s,
            max_tokens=max_tokens,
        )

        if hasattr(resp, "choices"):
            content = resp.choices[0].message.content
        else:  # pragma: no cover - defensive for dict-like clients
            content = resp["choices"][0]["message"]["content"]

        latency_ms = (time.perf_counter() - t0) * 1000.0
        emit(
            "ai_llm_call",
            model=model,
            temperature=temperature,
            latency_ms=round(latency_ms, 3),
        )
        return content
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        emit(
            "ai_llm_call_error",
            model=model,
            latency_ms=round(latency_ms, 3),
            error=type(exc).__name__,
        )
        raise


__all__ = ["run_llm_prompt"]

