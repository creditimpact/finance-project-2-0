from __future__ import annotations

import copy
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.core.io.json_io import _atomic_write_json

logger = logging.getLogger(__name__)


def _now_iso_utc() -> str:
    """Return the current time in ISO-8601 UTC format."""

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_nested_dict(container: dict[str, Any], *keys: str) -> dict[str, Any]:
    current: dict[str, Any] = container
    for key in keys:
        value = current.get(key)
        if not isinstance(value, dict):
            value = {}
            current[key] = value
        current = value
    return current


def register_date_convention_in_manifest(
    sid: str,
    run_root: str | os.PathLike[str],
    out_path_rel: str,
    detector_meta: dict[str, Any] | None = None,
) -> None:
    """Register the date convention artifact inside ``manifest.json`` for ``sid``."""

    run_root_path = Path(run_root)
    if not run_root_path.is_absolute():
        try:
            run_root_path = run_root_path.resolve()
        except FileNotFoundError:
            run_root_path = run_root_path.absolute()

    manifest_path = run_root_path / "manifest.json"

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest: dict[str, Any] = json.load(handle)
    except FileNotFoundError:
        manifest = {"sid": sid, "created_at": _now_iso_utc()}
    except json.JSONDecodeError:
        logger.error("MANIFEST_DATE_CONVENTION_INVALID sid=%s path=%s", sid, manifest_path, exc_info=True)
        return

    if not isinstance(manifest, dict):
        logger.error("MANIFEST_DATE_CONVENTION_UNEXPECTED sid=%s path=%s", sid, manifest_path)
        return

    original_snapshot = copy.deepcopy(manifest)

    traces = _ensure_nested_dict(manifest, "artifacts", "traces")
    prevalidation = _ensure_nested_dict(manifest, "prevalidation")

    file_rel_candidate = out_path_rel.replace("/", os.sep)
    file_rel = os.path.normpath(file_rel_candidate)
    file_abs = os.path.normpath(os.path.join(str(run_root_path), file_rel))

    if traces.get("date_convention") != file_abs:
        traces["date_convention"] = file_abs

    if traces.get("date_convention_rel") != file_rel:
        traces["date_convention_rel"] = file_rel

    detector_meta = detector_meta or {}

    existing_dc = prevalidation.get("date_convention")
    new_dc: dict[str, Any] = dict(existing_dc) if isinstance(existing_dc, dict) else {}

    fields_to_update: dict[str, Any] = {
        "convention": detector_meta.get("convention"),
        "month_language": detector_meta.get("month_language"),
        "confidence": detector_meta.get("confidence"),
        "file_rel": file_rel,
        "file_abs": file_abs,
        "detector_version": detector_meta.get("detector_version"),
    }

    content_changed = False
    for key, value in fields_to_update.items():
        if value is None and key not in {"file_rel", "file_abs"}:
            continue
        if new_dc.get(key) != value:
            new_dc[key] = value
            content_changed = True

    existing_created_at = None
    if isinstance(existing_dc, dict):
        existing_created_at = existing_dc.get("created_at")

    if content_changed or existing_created_at is None:
        desired_created_at = _now_iso_utc()
    else:
        desired_created_at = existing_created_at

    if desired_created_at is not None and new_dc.get("created_at") != desired_created_at:
        new_dc["created_at"] = desired_created_at
        content_changed = True

    if not isinstance(existing_dc, dict) or existing_dc != new_dc:
        prevalidation["date_convention"] = new_dc

    if manifest == original_snapshot:
        return

    _atomic_write_json(manifest_path, manifest)


__all__ = ["register_date_convention_in_manifest"]

