from __future__ import annotations

from pathlib import Path
import datetime


def dump_text(text: str, session_id: str | None = None, prefix: str = "extracted") -> str:
    """Write a text dump to disk for debugging and return the file path.

    The output directory is ``traces/`` at the repository root. The filename
    encodes the optional ``session_id``, the ``prefix`` (default: ``extracted``)
    and a timestamp so multiple dumps do not overwrite each other.
    """

    out_dir = Path("traces")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    sid = (session_id or "no-session").replace(":", "_").replace("/", "_")
    out_path = out_dir / f"{sid}-{prefix}-{ts}.txt"
    out_path.write_text(text or "", encoding="utf-8")
    return str(out_path)

