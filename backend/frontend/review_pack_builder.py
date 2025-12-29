"""Compatibility helpers for frontend review pack generation."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from backend.frontend.packs.generator import generate_frontend_packs_for_run


log = logging.getLogger(__name__)

_TEXT_SENTINELS = {"", "--", "—", "unknown", "Unknown"}


def _is_missing_text(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in _TEXT_SENTINELS
    return False


def get_first(data: Mapping[str, Any], *keys: str) -> Any | None:
    """Return the first non-empty value found in ``data`` for ``keys``."""

    for key in keys:
        if key not in data:
            continue
        value = data[key]
        if _is_missing_text(value):
            continue
        return value
    return None


def build_review_packs(sid: str, manifest: Any) -> dict[str, Any]:
    """Delegate manifest-driven builds to the legacy frontend pack generator."""

    manifest_path = getattr(manifest, "path", None)
    if manifest_path is None and isinstance(manifest, Mapping):
        manifest_path = manifest.get("path")

    if manifest_path is None:
        raise ValueError("manifest path is required to resolve run directory")

    run_dir = Path(manifest_path).resolve().parent
    runs_root = run_dir.parent

    log.info("FRONTEND: delegating manifest build sid=%s", sid)

    return generate_frontend_packs_for_run(sid, runs_root=runs_root, force=True)


__all__ = ["build_review_packs", "get_first"]
