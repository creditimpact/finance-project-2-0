from __future__ import annotations

from pathlib import Path
import datetime
import json
import re

try:
    from backend.core.utils.json_utils import _json_safe as __json_safe
except Exception:
    def __json_safe(x):
        return x

TRACES_DIR = Path("traces")


def _sanitize(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip())
    return name[:80] or "item"


def write_text_trace(text: str, *, session_id: str, prefix: str) -> str:
    """Write a UTF-8 text trace under ``traces/`` and return its path.

    The file name pattern is ``{session_id}-{prefix}-{timestamp}.txt``.
    """

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{_sanitize(session_id)}-{_sanitize(prefix)}-{ts}.txt"
    path = TRACES_DIR / fname
    path.write_text(text or "", encoding="utf-8")
    return str(path)


def write_json_trace(obj: dict, *, session_id: str, prefix: str) -> str:
    """Write a JSON trace under ``traces/`` and return its path.

    The file name pattern is ``{session_id}-{prefix}-{timestamp}.json``.
    """

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"{_sanitize(session_id)}-{_sanitize(prefix)}-{ts}.json"
    path = TRACES_DIR / fname
    path.write_text(json.dumps(__json_safe(obj or {}), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)
