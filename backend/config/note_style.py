"""Environment-backed configuration for the note_style stage.

The ``NOTE_STYLE_RESPONSE_MODE`` flag controls how note_style requests ask the
model to return JSON. It now has two explicit options:

``"content"`` (default)
    Enforce plain JSON responses delivered in ``assistant.content`` by setting
    ``response_format={"type": "json_object"}`` and disabling tool calls.

``"tool"``
    Require tool calls using the configured tool schema and parse responses via
    tool arguments.

The flag is parsed into :class:`NoteStyleResponseMode` for stronger typing and
to ensure downstream callers always receive a validated value.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Final

from . import _coerce_positive_int, _warn_default, env_bool


class NoteStyleResponseMode(str, Enum):
    """Valid response modes for note_style requests."""

    CONTENT = "content"
    TOOL = "tool"


_DEFAULT_RESPONSE_MODE: Final[NoteStyleResponseMode] = NoteStyleResponseMode.CONTENT


def _normalize_candidate(value: str) -> NoteStyleResponseMode | None:
    normalized = value.strip().lower()
    if not normalized:
        return None

    for mode in NoteStyleResponseMode:
        if normalized == mode.value:
            return mode

    # Backwards compatibility with legacy aliases.
    if normalized in {"json", "json_object"}:
        return NoteStyleResponseMode.CONTENT

    return None


def _coerce_response_mode() -> NoteStyleResponseMode:
    raw = os.getenv("NOTE_STYLE_RESPONSE_MODE")
    if raw is None:
        return _DEFAULT_RESPONSE_MODE

    candidate = _normalize_candidate(raw)
    if candidate is None:
        reason = "empty" if not str(raw).strip() else "invalid_choice"
        _warn_default("NOTE_STYLE_RESPONSE_MODE", raw, _DEFAULT_RESPONSE_MODE.value, reason)
        return _DEFAULT_RESPONSE_MODE

    return candidate


def _coerce_retry_count() -> int:
    return _coerce_positive_int("NOTE_STYLE_RETRY_COUNT", 2, min_value=0)


NOTE_STYLE_RESPONSE_MODE: Final[NoteStyleResponseMode] = _coerce_response_mode()
NOTE_STYLE_RETRY_COUNT: Final[int] = _coerce_retry_count()
NOTE_STYLE_STRICT_SCHEMA: Final[bool] = env_bool("NOTE_STYLE_STRICT_SCHEMA", True)
NOTE_STYLE_ALLOW_TOOL_CALLS: Final[bool] = env_bool(
    "NOTE_STYLE_ALLOW_TOOL_CALLS", False
)
NOTE_STYLE_ALLOW_TOOLS: Final[bool] = env_bool("NOTE_STYLE_ALLOW_TOOLS", False)


__all__ = [
    "NoteStyleResponseMode",
    "NOTE_STYLE_RESPONSE_MODE",
    "NOTE_STYLE_RETRY_COUNT",
    "NOTE_STYLE_STRICT_SCHEMA",
    "NOTE_STYLE_ALLOW_TOOL_CALLS",
    "NOTE_STYLE_ALLOW_TOOLS",
]

