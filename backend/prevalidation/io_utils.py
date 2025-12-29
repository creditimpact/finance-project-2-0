"""I/O helpers for the pre-validation pipeline."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _load_json_dict(path: Path) -> dict[str, Any]:
    """Load JSON content from ``path`` returning a mapping."""

    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    if not raw:
        return {}

    payload = json.loads(raw)
    if not isinstance(payload, dict):  # pragma: no cover - defensive
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _fsync_directory(path: Path) -> None:
    """Ensure ``path`` directory metadata is flushed to disk."""

    try:
        dir_fd = os.open(path, os.O_RDONLY)
    except FileNotFoundError:  # pragma: no cover - defensive
        return
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def atomic_merge_json(path: str, key: str, value: dict[str, Any]) -> None:
    """Load JSON if exists, set ``obj[key] = value`` and atomically persist."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = _load_json_dict(target)
    payload[key] = value

    tmp_path = target.with_suffix(target.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as tmp_file:
        json.dump(payload, tmp_file, ensure_ascii=False, indent=2)
        tmp_file.flush()
        os.fsync(tmp_file.fileno())

    os.replace(tmp_path, target)
    _fsync_directory(target.parent)


__all__ = ["atomic_merge_json"]

