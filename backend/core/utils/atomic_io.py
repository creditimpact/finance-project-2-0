from __future__ import annotations

import json
import os
import tempfile
from typing import Any


def atomic_write_json(path: str, data: Any, ensure_ascii: bool = False) -> None:
    """Atomically write JSON to ``path`` (tmp file + rename).

    Creates parent directories as needed. Ensures that readers never observe
    a partially written file.
    """

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    dir_name = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", dir=dir_name, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=ensure_ascii)
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

