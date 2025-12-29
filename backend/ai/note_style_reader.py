"""Read helpers for note_style stage outputs."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from backend.core.ai.paths import (
    ensure_note_style_account_paths,
    ensure_note_style_paths,
)


log = logging.getLogger(__name__)


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        env_root = os.getenv("RUNS_ROOT")
        return Path(env_root) if env_root else Path("runs")
    return Path(runs_root)


def _load_result_payload(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("NOTE_STYLE_METADATA_READ_FAILED path=%s", path, exc_info=True)
        return None

    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None

    try:
        payload = json.loads(lines[0])
    except json.JSONDecodeError:
        log.warning("NOTE_STYLE_METADATA_PARSE_FAILED path=%s", path, exc_info=True)
        return None

    if isinstance(payload, Mapping):
        return payload

    return None


def _sanitize_emphasis(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return []

    seen: set[str] = set()
    ordered: list[str] = []
    for entry in values:
        text = str(entry or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def get_style_metadata(
    sid: str,
    account_id: str,
    *,
    runs_root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Return sanitized tone/topic/emphasis metadata for ``account_id``.

    When the note_style result for ``sid``/``account_id`` is missing or invalid,
    ``None`` is returned.
    """

    runs_root_path = _resolve_runs_root(runs_root)
    paths = ensure_note_style_paths(runs_root_path, sid, create=False)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)

    payload = _load_result_payload(account_paths.result_file)
    if not isinstance(payload, Mapping):
        return None

    analysis = payload.get("analysis")
    if not isinstance(analysis, Mapping):
        analysis = payload.get("extractor")
        if not isinstance(analysis, Mapping):
            return None

    tone_raw = analysis.get("tone")
    tone = str(tone_raw).strip() if tone_raw is not None else ""

    context = analysis.get("context_hints")
    topic_raw: Any = None
    if isinstance(context, Mapping):
        topic_raw = context.get("topic")
    topic = str(topic_raw).strip() if topic_raw is not None else ""

    emphasis = _sanitize_emphasis(analysis.get("emphasis"))

    return {
        "tone": tone or "neutral",
        "topic": topic or "other",
        "emphasis": emphasis,
    }


__all__ = ["get_style_metadata"]
