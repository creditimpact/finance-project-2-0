"""Helpers for persisting note_style analysis results."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Mapping

from backend.core.ai.paths import (
    NoteStyleAccountPaths,
    ensure_note_style_account_paths,
    ensure_note_style_paths,
)

from .validator import coerce_text, validate_analysis_payload


log = logging.getLogger(__name__)


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        env_value = os.getenv("RUNS_ROOT")
        return Path(env_value) if env_value else Path("runs")
    return Path(runs_root)


def _write_jsonl(path: Path, payload: Mapping[str, Any]) -> int:
    text = json.dumps(payload, ensure_ascii=False)
    data = (text + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(data)
    return len(data)


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        if not math.isfinite(value) or value < 0:
            return None
        return int(value)
    text = coerce_text(value, preserve_case=True)
    if not text:
        return None
    digits = "".join(ch for ch in text if ch.isdigit())
    if not digits:
        try:
            numeric = float(text)
        except ValueError:
            return None
        if math.isnan(numeric) or math.isinf(numeric) or numeric < 0:
            return None
        return int(numeric)
    return int(digits)


def _extract_note_metrics(
    *,
    note_metrics: Mapping[str, Any] | None,
    pack_payload: Mapping[str, Any] | None,
) -> dict[str, int]:
    candidate: Mapping[str, Any] | None = None
    if isinstance(note_metrics, Mapping):
        candidate = note_metrics
    elif isinstance(pack_payload, Mapping):
        metrics_payload = pack_payload.get("note_metrics")
        if isinstance(metrics_payload, Mapping):
            candidate = metrics_payload

    if isinstance(candidate, Mapping):
        char_len = _coerce_int(candidate.get("char_len"))
        word_len = _coerce_int(candidate.get("word_len"))
        if char_len is not None and word_len is not None:
            return {"char_len": int(char_len), "word_len": int(word_len)}

    note_text: str | None = None
    if isinstance(pack_payload, Mapping):
        text_candidate = pack_payload.get("note_text")
        if not isinstance(text_candidate, str):
            context = pack_payload.get("context")
            if isinstance(context, Mapping):
                context_candidate = context.get("note_text")
                if isinstance(context_candidate, str):
                    text_candidate = context_candidate
        if isinstance(text_candidate, str) and text_candidate.strip():
            note_text = text_candidate

    if note_text is not None:
        char_len = len(note_text)
        word_len = len(note_text.split())
        return {"char_len": char_len, "word_len": word_len}

    raise ValueError("note_metrics with char_len and word_len is required")


def _ensure_account_paths(
    *,
    sid: str,
    account_id: str,
    runs_root: Path | str | None,
    create: bool,
) -> NoteStyleAccountPaths:
    runs_root_path = _resolve_runs_root(runs_root)
    paths = ensure_note_style_paths(runs_root_path, sid, create=create)
    return ensure_note_style_account_paths(paths, account_id, create=create)


def write_result(
    sid: str,
    account_id: str,
    analysis: Mapping[str, Any],
    *,
    runs_root: Path | str | None = None,
    note_metrics: Mapping[str, Any] | None = None,
    pack_payload: Mapping[str, Any] | None = None,
) -> Path:
    """Persist the sanitized ``analysis`` payload for ``account_id``."""

    sid_text = coerce_text(sid, preserve_case=True)
    if not sid_text:
        raise ValueError("sid is required")

    account_id_text = coerce_text(account_id, preserve_case=True)
    if not account_id_text:
        raise ValueError("account_id is required")

    normalized_analysis = validate_analysis_payload(analysis)
    metrics_payload = _extract_note_metrics(
        note_metrics=note_metrics, pack_payload=pack_payload
    )

    account_paths = _ensure_account_paths(
        sid=sid_text, account_id=account_id_text, runs_root=runs_root, create=True
    )

    payload = {
        "sid": sid_text,
        "account_id": account_id_text,
        "analysis": normalized_analysis,
        "note_metrics": metrics_payload,
    }

    bytes_written = _write_jsonl(account_paths.result_file, payload)

    log.info(
        "NOTE_STYLE_RESULT_WRITTEN sid=%s account_id=%s path=%s bytes=%d",
        sid_text,
        account_id_text,
        account_paths.result_file.resolve().as_posix(),
        bytes_written,
    )

    return account_paths.result_file


def _format_raw_payload(raw_payload: Any) -> str:
    if isinstance(raw_payload, (bytes, bytearray)):
        try:
            text = raw_payload.decode("utf-8", errors="ignore")
        except Exception:
            text = ""
    elif isinstance(raw_payload, str):
        text = raw_payload
    else:
        try:
            text = json.dumps(raw_payload, ensure_ascii=False, indent=2)
        except Exception:
            text = repr(raw_payload)
    text = text.rstrip("\n")
    return f"{text}\n" if text else ""


def write_failure_dump(
    sid: str,
    account_id: str,
    raw_payload: Any,
    *,
    runs_root: Path | str | None = None,
) -> Path:
    """Persist ``raw_payload`` to the debug results directory."""

    sid_text = coerce_text(sid, preserve_case=True)
    if not sid_text:
        raise ValueError("sid is required")

    account_id_text = coerce_text(account_id, preserve_case=True)
    if not account_id_text:
        raise ValueError("account_id is required")

    account_paths = _ensure_account_paths(
        sid=sid_text, account_id=account_id_text, runs_root=runs_root, create=True
    )

    text = _format_raw_payload(raw_payload)
    account_paths.result_raw_file.write_text(text, encoding="utf-8")

    log.info(
        "NOTE_STYLE_RESULT_RAW_WRITTEN sid=%s account_id=%s path=%s",
        sid_text,
        account_id_text,
        account_paths.result_raw_file.resolve().as_posix(),
    )

    return account_paths.result_raw_file


__all__ = ["write_result", "write_failure_dump"]
