from __future__ import annotations

import os
import re
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

__all__ = [
    "normalize_stage_path",
    "normalize_worker_path",
    "coerce_stage_path",
    "sanitize_stage_path_value",
    "ensure_frontend_review_dirs",
    "get_frontend_review_paths",
]

_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:")


def sanitize_stage_path_value(value: Any) -> str | None:
    """Return a sanitized representation of ``value`` suitable for workers."""

    if value is None:
        return None

    try:
        text = os.fspath(value)
    except TypeError:
        text = str(value)

    sanitized = str(text).strip()
    if not sanitized:
        return None

    sanitized = sanitized.replace("\\", "/")
    drive_match = _WINDOWS_DRIVE_PATTERN.match(sanitized)
    if drive_match:
        sanitized = sanitized[drive_match.end():]

    return sanitized


def normalize_stage_path(run_root: Path | str, raw: Any) -> Path:
    """Return a resolved path for a worker given a manifest or ENV value."""

    run_root_path = Path(run_root).resolve()
    sanitized = sanitize_stage_path_value(raw)
    if not sanitized:
        raise ValueError("path value must be a non-empty string")

    candidate = Path(sanitized)

    if candidate.is_absolute():
        drive_match = _WINDOWS_DRIVE_PATTERN.match(str(raw or ""))
        if drive_match:
            parts = [part for part in candidate.parts if part not in {"", ".", "/"}]
            lowered = [part.lower() for part in parts]
            run_root_name = run_root_path.name.lower()
            if run_root_name and run_root_name in lowered:
                idx = lowered.index(run_root_name)
                parts = parts[idx + 1 :]
            candidate = run_root_path.joinpath(*parts)
        try:
            return candidate.resolve()
        except OSError:
            return candidate

    candidate = run_root_path / candidate

    try:
        return candidate.resolve()
    except OSError:
        return candidate


def normalize_worker_path(base_dir: Path | str, raw: os.PathLike[str] | str) -> Path:
    """Return a normalized worker path for manifest or environment values."""

    base_path = Path(base_dir).resolve()

    try:
        text = os.fspath(raw)
    except TypeError:
        text = str(raw)

    sanitized = str(text).strip()
    if not sanitized:
        raise ValueError("path value must be a non-empty string")

    sanitized = sanitized.replace("\\", "/")

    drive_match = _WINDOWS_DRIVE_PATTERN.match(sanitized)
    if drive_match:
        sanitized = sanitized[drive_match.end():]

    candidate = Path(sanitized)
    if not candidate.is_absolute():
        candidate = base_path / candidate

    try:
        return candidate.resolve()
    except OSError:
        return candidate


def coerce_stage_path(
    run_root: Path | str, raw: Any, *, fallback: Path | str
) -> Path:
    """Return ``fallback`` when ``raw`` cannot be normalized."""

    fallback_path = Path(fallback).resolve()
    try:
        return normalize_stage_path(run_root, raw)
    except ValueError:
        return fallback_path


_frontend_review = import_module("backend.core.frontend_review_paths")
frontend_review = _frontend_review
sys.modules[__name__ + ".frontend_review"] = _frontend_review

get_frontend_review_paths = _frontend_review.get_frontend_review_paths
ensure_frontend_review_dirs = _frontend_review.ensure_frontend_review_dirs

__all__.append("frontend_review")
