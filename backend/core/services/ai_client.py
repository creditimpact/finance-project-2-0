from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from openai import OpenAI

from backend.core.services.ai_config import AIConfig


logger = logging.getLogger(__name__)
class AIClientProtocolError(RuntimeError):
    """Raised when the OpenAI response violates the expected protocol."""


class ChatCompletionResult(tuple, Mapping[str, Any]):
    """Tuple-like wrapper that also exposes a mapping interface."""

    def __new__(
        cls,
        raw_response: Any,
        parsed_payload: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> "ChatCompletionResult":
        obj = super().__new__(cls, (raw_response, parsed_payload))
        obj._metadata = metadata
        return obj

    # Mapping interface -------------------------------------------------
    def __getitem__(self, key: Any) -> Any:  # type: ignore[override]
        if isinstance(key, int):
            return super().__getitem__(key)
        return self._metadata[key]

    def __iter__(self) -> Iterator[Any]:
        return iter(self._metadata)

    def __len__(self) -> int:
        return len(self._metadata)

    def get(self, key: Any, default: Any = None) -> Any:
        return self._metadata.get(key, default)

    def keys(self) -> Iterable[Any]:
        return self._metadata.keys()

    def items(self) -> Iterable[Tuple[Any, Any]]:
        return self._metadata.items()

    def values(self) -> Iterable[Any]:
        return self._metadata.values()

    def __contains__(self, key: object) -> bool:  # type: ignore[override]
        return key in self._metadata

    # Convenience tuple accessors --------------------------------------
    @property
    def raw(self) -> Any:
        return self[0]

    @property
    def parsed(self) -> Dict[str, Any]:
        return self[1]


class AIClient:
    """Thin wrapper around the OpenAI client used throughout the codebase."""

    def __init__(self, config: AIConfig):
        api_key = (config.api_key or "").strip()
        if not api_key:
            logger.error(
                "AI_CLIENT_CREDENTIAL_ERROR model=%s base_url=%s detail=missing_api_key",
                config.chat_model,
                (config.base_url or "https://api.openai.com/v1"),
            )
            raise RuntimeError(
                "AI client requires an OpenAI API key. Set OPENAI_API_KEY before starting."
            )

        base_url = (config.base_url or "").strip() or "https://api.openai.com/v1"

        logger.info(
            "AI_CLIENT_READY model=%s response_model=%s base_url=%s key_present=yes",
            config.chat_model,
            config.response_model,
            base_url,
        )

        config.api_key = api_key
        config.base_url = base_url

        self.config = config
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    # --- OpenAI API helpers -------------------------------------------------
    def chat_completion(
        self,
        *,
        messages: List[Dict[str, Any]],
        model: str | None = None,
        temperature: float = 0,
        top_p: float = 1,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        """Proxy to ``chat.completions.create``."""

        model = model or self.config.chat_model

        extra_headers = kwargs.pop("extra_headers", None)
        if extra_headers:
            sanitized: Dict[str, str] = {}
            for k, v in extra_headers.items():
                cleaned = v.encode("ascii", "ignore").decode()
                if cleaned:
                    sanitized[k] = cleaned
            if sanitized:
                kwargs["extra_headers"] = sanitized

        # Enforce deterministic parameters regardless of caller provided values.
        if "frequency_penalty" in kwargs:
            kwargs.pop("frequency_penalty")
        if "presence_penalty" in kwargs:
            kwargs.pop("presence_penalty")
        if "top_p" in kwargs:
            kwargs.pop("top_p")

        frequency_penalty = 0
        presence_penalty = 0
        top_p_value = top_p

        is_note_style_request = bool(kwargs.pop("_note_style_request", False))
        if is_note_style_request:
            kwargs["response_format"] = {"type": "json_object"}
            kwargs.pop("tools", None)
            kwargs.pop("tool_choice", None)
        else:
            using_tools = bool(kwargs.get("tools"))
            if not using_tools:
                kwargs.setdefault("response_format", {"type": "json_object"})

        # derive intent from kwargs, do not rely on external flags
        wants_json_object = False
        rf = kwargs.get("response_format")
        if isinstance(rf, dict) and rf.get("type") == "json_object":
            wants_json_object = True

        resp = self._client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p_value,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        )

        usage = getattr(resp, "usage", None)
        prompt_tokens: int | None = None
        response_tokens: int | None = None
        total_tokens: int | None = None
        if usage is not None:
            if isinstance(usage, Mapping):
                prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
                response_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
                total_tokens = usage.get("total_tokens")
            else:
                prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(
                    usage, "input_tokens", None
                )
                response_tokens = getattr(usage, "completion_tokens", None) or getattr(
                    usage, "output_tokens", None
                )
                total_tokens = getattr(usage, "total_tokens", None)

        choice = resp.choices[0]
        message = choice.message

        def _coerce_json_value(value: Any) -> Any:
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, Mapping):
                return {
                    str(key): _coerce_json_value(sub_value) for key, sub_value in value.items()
                }
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return [_coerce_json_value(item) for item in value]
            return str(value)

        def _payload_to_text(payload: Any) -> str | None:
            if payload is None:
                return None
            if isinstance(payload, str):
                return payload
            if isinstance(payload, (bytes, bytearray)):
                return payload.decode("utf-8", "ignore")
            if isinstance(payload, Mapping) or (
                isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray))
            ):
                coerced = _coerce_json_value(payload)
                try:
                    return json.dumps(coerced, ensure_ascii=False)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    return str(coerced)
            return str(payload)

        def _parse_json_object(
            text: str,
            *,
            context: str,
            allow_recovery: bool,
        ) -> Dict[str, Any]:
            try:
                parsed_payload = json.loads(text)
            except json.JSONDecodeError as exc:
                if not allow_recovery:
                    raise AIClientProtocolError(
                        f"{context} did not contain valid JSON object"
                    ) from exc

                from backend.util.json_tools import try_fix_to_json

                fixed = try_fix_to_json(text)
                if fixed is None:
                    raise AIClientProtocolError(
                        f"{context} did not contain valid JSON object"
                    ) from exc

                fixed_len = len(json.dumps(fixed, ensure_ascii=False))
                logger.warning(
                    "NOTE_STYLE_JSON_FIXED context=%s raw_len=%d fixed_len=%d",
                    context,
                    len(text),
                    fixed_len,
                )
                return fixed

            if not isinstance(parsed_payload, dict):
                raise AIClientProtocolError(
                    f"{context} did not produce a JSON object"
                )

            return parsed_payload

        content_payload = getattr(message, "content", None)

        raw_content = None
        mode: str | None = None
        parsed_json: Dict[str, Any] | None = None
        raw_tool_arguments = None

        tool_calls = getattr(message, "tool_calls", None)

        if wants_json_object:
            raw_content = _payload_to_text(content_payload)
            if raw_content is None:
                raise AIClientProtocolError(
                    "json_object expected in message.content but got None (tool call?)"
                )
            try:
                parsed = json.loads(raw_content)
            except json.JSONDecodeError as exc:
                raise AIClientProtocolError(
                    "json_object response was not valid JSON"
                ) from exc
            except Exception as exc:  # pragma: no cover - defensive
                raise AIClientProtocolError(
                    f"json_object parsing failed: {exc}"
                ) from exc
            if not isinstance(parsed, dict):
                raise AIClientProtocolError("json_object expected a JSON object")
            parsed_json = parsed
            mode = "content"
        else:
            raw_content = _payload_to_text(content_payload)

        content_text = (raw_content or "").strip() if isinstance(raw_content, str) else None
        if not wants_json_object:
            if content_text:
                parsed_json = _parse_json_object(
                    content_text,
                    context="message.content",
                    allow_recovery=True,
                )
                mode = "content"
            elif tool_calls:
                first_call = tool_calls[0]
                arguments = getattr(getattr(first_call, "function", object()), "arguments", None)
                raw_tool_arguments = _payload_to_text(arguments)
                arguments_text = (
                    raw_tool_arguments.strip() if isinstance(raw_tool_arguments, str) else None
                )
                if not arguments_text:
                    raise AIClientProtocolError("Tool call arguments missing JSON payload")
                parsed_json = _parse_json_object(
                    arguments_text,
                    context="message.tool_calls[0].function.arguments",
                    allow_recovery=False,
                )
                mode = "tool"

        if parsed_json is None:
            if wants_json_object:
                raise AIClientProtocolError("json_object parsing produced no payload")
            raise AIClientProtocolError("No content or tool_calls in response")

        content_text_value = None
        if mode == "content" and not wants_json_object:
            content_text_value = content_text if content_text else None

        metadata = {
            "mode": mode,
            "content_json": parsed_json if mode == "content" else None,
            "content_text": content_text_value,
            "tool_json": parsed_json if mode == "tool" else None,
            "json": parsed_json,
            "raw": resp,
            "openai": resp,
            "raw_content": raw_content,
            "raw_tool_arguments": raw_tool_arguments,
        }

        if any(value is not None for value in (prompt_tokens, response_tokens, total_tokens)):
            logger.info(
                "AI_CLIENT_CHAT_USAGE model=%s prompt_tokens=%s response_tokens=%s total_tokens=%s",
                model,
                prompt_tokens if prompt_tokens is not None else "?",
                response_tokens if response_tokens is not None else "?",
                total_tokens if total_tokens is not None else "?",
            )

        return ChatCompletionResult(resp, parsed_json, metadata)

    def response_json(
        self,
        *,
        prompt: str,
        model: str | None = None,
        response_format: Dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Proxy to ``responses.create`` for JSON output."""

        model = model or self.config.response_model
        return self._client.responses.create(
            model=model,
            input=prompt,
            response_format=response_format,
            **kwargs,
        )


def build_ai_client(config: AIConfig) -> AIClient:
    """Return an :class:`AIClient` instance from ``config``."""

    return AIClient(config)


def get_default_ai_client() -> AIClient:  # pragma: no cover - backwards compat
    """Deprecated helper that previously returned a global client.

    The application now requires explicit :class:`AIClient` injection at all
    call sites. This stub remains temporarily to surface a clearer error if
    legacy code attempts to use the old global accessor.
    """

    raise RuntimeError(
        "get_default_ai_client() has been removed. Build an AIClient via"
        " services.ai_client.build_ai_client and pass it explicitly."
    )


class _StubAIClient:
    """Fallback AI client used when configuration is missing."""

    def chat_completion(self, *a: Any, **kw: Any):  # pragma: no cover - minimal stub
        raise RuntimeError("AI client is not configured")

    def response_json(self, *a: Any, **kw: Any):  # pragma: no cover - minimal stub
        raise RuntimeError("AI client is not configured")


def get_ai_client() -> AIClient | _StubAIClient:
    """Return a configured :class:`AIClient` or a safe stub.

    If environment configuration is missing or invalid, a no-op client is
    returned which raises ``RuntimeError`` on use. This keeps the application
    dependency-injection friendly while providing a clear failure mode during
    development.
    """

    from backend.api.config import get_ai_config

    try:
        cfg = get_ai_config()
        return build_ai_client(cfg)
    except Exception as exc:  # pragma: no cover - best effort fallback
        logger.warning("Using stub AI client due to configuration error: %s", exc)
        return _StubAIClient()
