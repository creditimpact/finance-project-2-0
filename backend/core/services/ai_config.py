"""Lightweight configuration dataclasses for AI services."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AIConfig:
    """Configuration payload shared between the API config and AI client."""

    api_key: str
    base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4"
    response_model: str = "gpt-4.1-mini"
    timeout: float | None = None
    max_retries: int = 0


__all__ = ["AIConfig"]

