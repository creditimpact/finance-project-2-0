"""Date convention loader for tradeline_check.

Resolves the global date_convention trace via the manifest and returns
an always-present block suitable for attaching to per-bureau payloads.
Non-blocking: logs warnings and falls back to deterministic defaults.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Mapping

log = logging.getLogger(__name__)

DATE_CONVENTION_VERSION = "date_convention_v1"
DEFAULT_REL_PATH = "traces/date_convention.json"


def _default_block(file_abs: str | None, file_rel: str | None) -> dict[str, Any]:
    return {
        "version": DATE_CONVENTION_VERSION,
        "scope": "unknown",
        "convention": "unknown",
        "month_language": "unknown",
        "confidence": 0.0,
        "evidence_counts": {},
        "detector_version": "unknown",
        "source": {
            "file_abs": file_abs,
            "file_rel": file_rel or DEFAULT_REL_PATH,
            "created_at": None,
        },
    }


def _safe_load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        if isinstance(data, Mapping):
            return data
    except FileNotFoundError:
        log.warning("DATE_CONVENTION_FILE_MISSING path=%s", path)
    except Exception as exc:  # pragma: no cover - defensive
        log.warning("DATE_CONVENTION_FILE_READ_FAILED path=%s error=%s", path, exc, exc_info=True)
    return None


def load_date_convention(account_dir: Path) -> dict[str, Any]:
    """Load date_convention trace via manifest; always returns a block.

    Fallbacks are deterministic and never raise.
    """

    manifest_path = account_dir.parent.parent.parent / "manifest.json"
    file_abs: str | None = None
    file_rel: str | None = DEFAULT_REL_PATH
    trace_payload: Mapping[str, Any] | None = None
    created_at: str | None = None

    manifest = _safe_load_json(manifest_path)
    if manifest is None:
        return _default_block(file_abs, file_rel)

    artifacts = manifest.get("artifacts") if isinstance(manifest, Mapping) else None
    traces = artifacts.get("traces") if isinstance(artifacts, Mapping) else None
    base_dirs = manifest.get("base_dirs") if isinstance(manifest, Mapping) else None

    if isinstance(traces, Mapping):
        abs_candidate = traces.get("date_convention")
        rel_candidate = traces.get("date_convention_rel")
        if isinstance(abs_candidate, str) and abs_candidate.strip():
            file_abs = abs_candidate.strip()
        if isinstance(rel_candidate, str) and rel_candidate.strip():
            file_rel = rel_candidate.strip()

    if file_abs is None and isinstance(base_dirs, Mapping):
        traces_dir = base_dirs.get("traces_dir")
        if isinstance(traces_dir, str) and traces_dir.strip() and file_rel:
            file_abs = str(Path(traces_dir).resolve() / file_rel)

    if file_abs:
        trace_payload = _safe_load_json(Path(file_abs))
    else:
        log.warning("DATE_CONVENTION_PATH_UNRESOLVED manifest=%s", manifest_path)

    if isinstance(trace_payload, Mapping):
        inner = trace_payload.get("date_convention")
        if isinstance(inner, Mapping):
            scope = str(inner.get("scope") or "unknown")
            convention = str(inner.get("convention") or "unknown")
            month_language = str(inner.get("month_language") or "unknown")
            confidence = inner.get("confidence")
            detector_version = str(inner.get("detector_version") or "unknown")
            evidence_counts = inner.get("evidence_counts") if isinstance(inner.get("evidence_counts"), Mapping) else {}
            try:
                confidence_val = float(confidence)
            except (TypeError, ValueError):
                confidence_val = 0.0

            block = _default_block(file_abs, file_rel)
            block.update(
                {
                    "scope": scope,
                    "convention": convention,
                    "month_language": month_language,
                    "confidence": confidence_val,
                    "evidence_counts": dict(evidence_counts),
                    "detector_version": detector_version,
                }
            )
            if isinstance(trace_payload, Mapping):
                created_at_val = trace_payload.get("created_at")
                if isinstance(created_at_val, str) and created_at_val.strip():
                    created_at = created_at_val.strip()
            block["source"]["created_at"] = created_at
            return block

    return _default_block(file_abs, file_rel)
